import os, runpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FCAST = os.environ.get("MBFM_FCAST", r"D:\MBFM_project\results\forecasts")
OLD_QLIKE = {"garch11": 0.154288, "egarch": 0.131604, "gjr_garch": 0.158456}   # old fixed-forecast run

for script in ["garch11.py", "egarch.py", "gjr_garch.py"]:
    print("\n" + "#" * 60 + "\n# " + script + "\n" + "#" * 60)
    runpy.run_path(os.path.join(HERE, script), run_name="__main__")

print("\n" + "=" * 60 + "\nSUMMARY: distinct forecast values per model (old = 15)")
for k, old in OLD_QLIKE.items():
    f = np.load(os.path.join(FCAST, k + ".npy"))
    print(f"  {k:10s} distinct = {len(np.unique(np.round(f[~np.isnan(f)], 10))):5d}   old QLIKE = {old}")
print("done")

