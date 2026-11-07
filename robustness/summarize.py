import os
import pandas as pd

RESDIR = r"D:\MBFM_project\results\robustness"

r1 = pd.read_csv(os.path.join(RESDIR, "r1_no_covid.csv"),    index_col=0)["QLIKE"].rename("R1_no_covid")
r2 = pd.read_csv(os.path.join(RESDIR, "r2_5day_rv.csv"),     index_col=0)["QLIKE"].rename("R2_5day_rv")
r3 = pd.read_csv(os.path.join(RESDIR, "r3_market_only.csv"), index_col=0)["QLIKE"].rename("R3_market_only")
r4 = pd.read_csv(os.path.join(RESDIR, "r4_5day_ahead.csv"),  index_col=0)["QLIKE"].rename("R4_5day_ahead")

base = pd.read_csv(r"D:\MBFM_project\results\table4_main_accuracy.csv",index_col=0)["QLIKE"].rename("Baseline")
summary = pd.concat([base, r1, r2, r3, r4], axis=1, join="outer")
garch_har = ["GARCH(1,1)", "EGARCH", "GJR-GARCH", "HAR-RV", "HAR-RV-X"]
summary.loc[summary.index.isin(garch_har), "R3_market_only"] = float("nan")

print("\n=== TABLE 7: Robustness QLIKE ===")
print(summary.round(6).to_string())
summary.to_csv(os.path.join(RESDIR, "table7_robustness.csv"))
print(f"\nsaved table7_robustness.csv")
