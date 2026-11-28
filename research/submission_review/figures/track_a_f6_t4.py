"""Build Figure F6 and Table T4 only from the frozen historical FY2010–24 overlay."""
import hashlib
from pathlib import Path
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

source, out = map(Path, sys.argv[1:])
raw = source.read_bytes()
git_blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
assert git_blob == "d26ee005a9543b8b3b84eef8fb11106226eb5247"
df = pd.read_csv(source)
assert len(df) == df.disasterNumber.nunique() == 912 and df.fyDeclared.between(2010, 2024).all()
bands = ["0-100K", "100K-1M", "1M-50M", "50-200M", "200-500M", "500M+"]
display = ["0–<100K", "100K–<1M", "1M–<50M", "50M–<200M", "200M–<500M", "500M+"]
assert set(df.actual_band) == set(df.router_pred) == set(bands)
matrix = pd.crosstab(df.actual_band, df.router_pred).reindex(index=bands, columns=bands)
n = matrix.sum(axis=1)
recall = np.diag(matrix) / n
assert np.trace(matrix) == 830
columns = ["frozen_stack_prediction", "accepted_low_band_prediction", "challenger_stack_prediction"]
errors = df[columns].sub(df.target, axis=0).abs()
mae = errors.mean()
assert np.allclose(mae, [5131359.538876612, 5127644.080431649, 5014819.011614494], rtol=0, atol=1e-6)
shares = errors.groupby(df.actual_band).sum().reindex(bands).div(errors.sum()) * 100
assert round(shares.iloc[-1,-1],2)==73.88 and round(shares.iloc[3:,-1].sum(),2)==91.81
report = matrix.copy()
report["actual_n"] = n
report["recall_pct"] = 100 * recall
report["overlay_absolute_error_share_pct"] = shares.iloc[:,-1]
out.mkdir(parents=True, exist_ok=True)
report.to_csv(out/"F6-T4-band-evidence.csv",float_format="%.9f")
pd.DataFrame({"amount_version":columns,"mae_usd":mae.values}).to_csv(out/"T4-amount-versions.csv",index=False,float_format="%.9f")
mpl.rcParams.update({"font.family":"DejaVu Sans","svg.hashsalt":"track-a-f6-20261128"})
fig,(ax,bar)=plt.subplots(1,2,figsize=(12,5.7),gridspec_kw={"width_ratios":[1.24,1]})
fig.subplots_adjust(left=.15,right=.96,bottom=.30,top=.82,wspace=.52)
fig.suptitle("Frozen Track A: historical routes and dollar error",fontsize=17,weight="bold",x=.51,y=.96)
ax.set_title("A  Route agreement (case counts)",loc="left",fontsize=11,weight="bold",pad=13)
row_fraction=matrix.div(n,axis=0).to_numpy()
ax.imshow(row_fraction,cmap="Blues",vmin=0,vmax=1,aspect="auto")
ax.set_xticks(range(6),display,rotation=30,ha="right",fontsize=8)
ax.set_yticks(range(6),[f"{b}  (n={v})" for b,v in zip(display,n)],fontsize=9)
ax.set_xlabel("Archived predicted band (USD)",fontsize=9)
ax.set_ylabel("Observed funding band (USD)",fontsize=9)
for (y,x),value in np.ndenumerate(matrix.to_numpy()):
    ax.text(x,y,str(value),ha="center",va="center",color="white" if row_fraction[y,x]>.56 else "#27364b",fontsize=9)
bar.set_title("B  Share of total absolute error",loc="left",fontsize=11,weight="bold",pad=13)
fs=shares.iloc[:,-1].to_numpy()
bar.barh(range(6),fs,color=["#88aabf"]*5+["#1e587c"],height=.63)
bar.set_yticks(range(6),display,fontsize=9); bar.invert_yaxis()
bar.set_xlim(0,88); bar.set_xticks([0,20,40,60,80],["0%","20%","40%","60%","80%"],fontsize=8)
bar.grid(axis="x",alpha=.18); bar.set_axisbelow(True)
bar.spines[["top","right","left"]].set_visible(False)
bar.tick_params(axis="y",length=0)
for i,v in enumerate(fs): bar.text(v+1.1,i,f"{v:.1f}%",va="center",fontsize=9,color="#27364b")
fig.text(.15,.12,"912 FY2010–24 cases; 830 archived routes agree. Color in A encodes within-row proportion.",fontsize=9)
fig.text(.15,.08,"Four routing rules used observed amounts for eligibility. This is not target-blind accuracy.",fontsize=9,weight="bold")
fig.text(.15,.04,"B uses the research-only historical amount overlay; the five $500M+ cases contribute 73.9% of absolute error.",fontsize=8.5)
fig.savefig(out/"F6-track-a.svg",metadata={"Date":None})
fig.savefig(out/"F6-track-a.png",dpi=220)
print(f"F6, T4 complete: {np.trace(matrix)}/{len(df)}; $500M+ share {fs[-1]:.4f}%")
