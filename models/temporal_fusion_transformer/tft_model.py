import torch
import torch.nn as nn
import torch.nn.functional as F


class GRN(nn.Module):
    def __init__(self, d_in, d_out=None, d_context=None, dropout=0.1):
        super().__init__()
        d_out = d_out or d_in
        self.fc1  = nn.Linear(d_in,  d_out)
        self.fc2  = nn.Linear(d_out, d_out)
        self.ctx  = nn.Linear(d_context, d_out, bias=False) if d_context else None
        self.gate = nn.Linear(d_out, d_out * 2)
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Linear(d_in, d_out, bias=False) if d_in != d_out else nn.Identity()

    def forward(self, x, context=None):
        h = self.fc1(x)
        if self.ctx is not None and context is not None:
            h = h + self.ctx(context)
        h  = F.elu(h)
        h  = self.fc2(h)
        h  = self.drop(h)
        g  = self.gate(h)
        g1, g2 = g.chunk(2, dim=-1)
        h  = torch.sigmoid(g1) * g2
        return self.norm(self.skip(x) + h)


class VarSelNet(nn.Module):
    def __init__(self, n_vars, d, d_context=None, dropout=0.1):
        super().__init__()
        self.grns     = nn.ModuleList([GRN(d, dropout=dropout) for _ in range(n_vars)])
        self.selector = GRN(n_vars * d, d_out=d, d_context=d_context, dropout=dropout)
        self.softmax  = nn.Linear(d, n_vars)
        self.d        = d
        self.n_vars   = n_vars

    def forward(self, x, context=None):
        B = x.size(0)
        processed = torch.stack(
            [self.grns[i](x[:, i]) for i in range(self.n_vars)], dim=1
        )
        flat    = x.reshape(B, -1)
        sel_out = self.selector(flat, context)
        weights = F.softmax(self.softmax(sel_out), dim=-1)
        out     = (processed * weights.unsqueeze(-1)).sum(dim=1)
        return out, weights


class InterpretableAttention(nn.Module):
    def __init__(self, d, n_heads=1, dropout=0.0):
        super().__init__()
        self.d_k   = d // n_heads
        self.heads = n_heads
        self.wq    = nn.ModuleList([nn.Linear(d, self.d_k, bias=False) for _ in range(n_heads)])
        self.wk    = nn.ModuleList([nn.Linear(d, self.d_k, bias=False) for _ in range(n_heads)])
        self.wv    = nn.Linear(d, self.d_k, bias=False)
        self.wo    = nn.Linear(self.d_k, d, bias=False)
        self.drop  = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, T, _ = q.shape
        V = self.wv(v)
        attn_weights, head_outs = [], []
        for i in range(self.heads):
            Q  = self.wq[i](q)
            K  = self.wk[i](k)
            sc = torch.bmm(Q, K.transpose(1, 2)) / (self.d_k ** 0.5)
            if mask is not None:
                sc = sc.masked_fill(mask, float("-inf"))
            w  = F.softmax(sc, dim=-1)
            w  = self.drop(w)
            attn_weights.append(w)
            head_outs.append(torch.bmm(w, V))
        attn = torch.stack(attn_weights, dim=0).mean(0)
        out  = self.wo(torch.stack(head_outs, dim=0).mean(0))
        return out, attn


class TFTModel(nn.Module):
    def __init__(self, n_obs, n_known, n_static, d=64, n_heads=1,
                 n_lstm_layers=1, dropout=0.3, encoder_len=63):
        super().__init__()
        self.d           = d
        self.n_obs       = n_obs
        self.n_known     = n_known
        self.encoder_len = encoder_len

        self.obs_proj   = nn.ModuleList([nn.Linear(1, d) for _ in range(n_obs)])
        self.known_proj = nn.ModuleList([nn.Linear(1, d) for _ in range(n_known)])

        self.obs_sel   = VarSelNet(n_obs,   d, d_context=d, dropout=dropout)
        self.known_sel = VarSelNet(n_known, d, d_context=d, dropout=dropout)

        self.lstm      = nn.LSTM(d, d, num_layers=n_lstm_layers, batch_first=True,
                                 dropout=dropout if n_lstm_layers > 1 else 0.0)
        self.lstm_gate = nn.Linear(d, d * 2)
        self.lstm_norm = nn.LayerNorm(d)

        self.enrich_grn = GRN(d, d_context=d, dropout=dropout)

        self.attn      = InterpretableAttention(d, n_heads, dropout=0.0)
        self.attn_gate = nn.Linear(d, d * 2)
        self.attn_norm = nn.LayerNorm(d)

        self.pw_grn  = GRN(d, dropout=dropout)
        self.pw_gate = nn.Linear(d, d * 2)
        self.pw_norm = nn.LayerNorm(d)

        self.drop    = nn.Dropout(dropout)
        self.out_p50 = nn.Linear(d, 1)
        self.out_p10 = nn.Linear(d, 1)
        self.out_p90 = nn.Linear(d, 1)

    def _glu(self, x, layer):
        g1, g2 = layer(x).chunk(2, dim=-1)
        return torch.sigmoid(g1) * g2

    def forward(self, obs, known, static=None):
        B     = obs.size(0)
        T_obs = obs.size(1)
        T_kn  = known.size(1)

        cs = torch.zeros(B, self.d, device=obs.device)
        ce = torch.zeros(B, self.d, device=obs.device)
        ch = torch.zeros(B, self.d, device=obs.device)
        cc = torch.zeros(B, self.d, device=obs.device)

        obs_emb  = torch.stack(
            [self.obs_proj[i](obs[:, :, i:i+1]) for i in range(self.n_obs)], dim=2
        )
        obs_flat = obs_emb.reshape(B * T_obs, self.n_obs, self.d)
        ctx_obs  = cs.unsqueeze(1).expand(-1, T_obs, -1).reshape(B * T_obs, self.d)
        obs_sel, obs_w = self.obs_sel(obs_flat, ctx_obs)
        obs_sel  = obs_sel.view(B, T_obs, self.d)

        kn_emb  = torch.stack(
            [self.known_proj[i](known[:, :, i:i+1]) for i in range(self.n_known)], dim=2
        )
        kn_flat = kn_emb.reshape(B * T_kn, self.n_known, self.d)
        ctx_kn  = cs.unsqueeze(1).expand(-1, T_kn, -1).reshape(B * T_kn, self.d)
        kn_sel, kn_w = self.known_sel(kn_flat, ctx_kn)
        kn_sel  = kn_sel.view(B, T_kn, self.d)

        h0 = ch.unsqueeze(0).expand(self.lstm.num_layers, -1, -1).contiguous()
        c0 = cc.unsqueeze(0).expand(self.lstm.num_layers, -1, -1).contiguous()

        enc_out, (hn, cn) = self.lstm(obs_sel, (h0, c0))
        dec_out, _        = self.lstm(kn_sel,  (hn, cn))
        lstm_out = torch.cat([enc_out, dec_out], dim=1)

        combined = torch.cat([obs_sel, kn_sel], dim=1)
        phi      = self.lstm_norm(combined + self._glu(lstm_out, self.lstm_gate))

        theta = self.enrich_grn(phi, ce.unsqueeze(1).expand(-1, phi.size(1), -1))

        T_total = theta.size(1)
        mask    = torch.triu(
            torch.ones(T_total, T_total, device=obs.device), diagonal=1
        ).bool().unsqueeze(0).expand(B, -1, -1)

        beta, attn_weights = self.attn(theta, theta, theta, mask)
        delta = self.attn_norm(theta + self._glu(beta, self.attn_gate))

        psi     = self.pw_grn(delta)
        psi_hat = self.pw_norm(phi + self._glu(psi, self.pw_gate))

        last = self.drop(psi_hat[:, -1])

        p50 = self.out_p50(last).squeeze(-1)
        p10 = self.out_p10(last).squeeze(-1)
        p90 = self.out_p90(last).squeeze(-1)

        return p50, p10, p90, attn_weights, obs_w