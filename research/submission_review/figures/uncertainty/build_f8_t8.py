"""Compare fixed archived historical intervals; no training or new target data."""
from pathlib import Path
import argparse
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

p=argparse.ArgumentParser()
p.add_argument("repo",type=Path)
p.add_argument("out",type=Path)
args=p.parse_args()
root=args.repo/"research/results/uncertainty"
out=args.out
out.mkdir(parents=True,exist_ok=True)
inputs={"cqr_case_intervals.csv":"a58ca45e4e2aac067c7e70b1aa87501d57be4980791bf66a1eed5b9e718ad28d",
        "forward_prediction_intervals.csv":"dff977a755417ee88fe45bcca71488d1930733d8085d1c760b136a81b616771e"}
for name,digest in inputs.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name

a=pd.read_csv(root/"cqr_case_intervals.csv").rename(columns={"actual":"amount"})
b=pd.read_csv(root/"forward_prediction_intervals.csv")
b=b.query("variant == 'nested_metadata' and method == 'dollar'").rename(
    columns={"target":"amount","lower_usd":"lower","upper_usd":"upper","width_usd":"width"})
keys=["disasterNumber","fyDeclared","level"]
cols=keys+["amount","lower","upper","width","covered"]
for t in (a,b):
    assert set(t.level)=={.8,.9} and t.fyDeclared.between(2020,2024).all()
    assert len(t)==616 and not t.duplicated(keys).any()
    assert (t.lower<=t.upper).all() and np.allclose(t.width,t.upper-t.lower,rtol=1e-7,atol=1e-7)
    assert ((t.lower<=t.amount)&(t.amount<=t.upper)).equals(t.covered.astype(bool))
assert b.last_calibration_year.lt(b.fyDeclared).all()
pairs=a[cols].merge(b[cols],on=keys,suffixes=("_cqr","_residual"),validate="one_to_one")
assert len(pairs)==616 and np.array_equal(np.rint(100*pairs.amount_cqr),np.rint(100*pairs.amount_residual))
assert pairs.groupby("level").disasterNumber.nunique().eq(308).all()
report=[]
for level,t in pairs.groupby("level"):
    tail=t.amount_cqr.ge(50_000_000)
    assert tail.sum()==9
    for method in ["residual","cqr"]:
        for scope,g in [("all",t),("under_50m",t[~tail]),("50m_or_more",t[tail])]:
            y=g["amount_"+method]; lo=g["lower_"+method]; hi=g["upper_"+method]
            width=hi-lo
            above=(y>hi);below=(y<lo)
            score=width+2/(1-level)*((lo-y).clip(lower=0)+(y-hi).clip(lower=0))
            report.append({"nominal":level,"pipeline":method,"scope":scope,
                "n":len(g),"covered":int((~(above|below)).sum()),"above_upper":int(above.sum()),
                "below_lower":int(below.sum()),"median_width_usd":width.median(),"mean_interval_score_usd":score.mean()})
d=pd.DataFrame(report)
assert d.query("scope=='50m_or_more'").covered.tolist()==[0,5,0,7]
assert d.query("scope=='all'").covered.tolist()==[260,275,281,293]
assert d.query("scope=='50m_or_more' and nominal==.9").above_upper.tolist()==[9,2]
d.to_csv(out/"T8-matched-intervals.csv",index=False,float_format="%.8f",lineterminator="\n")
plt.rcParams.update({"font.family":"DejaVu Sans","svg.hashsalt":"f8-thesis"})
fig,axes=plt.subplots(1,2,figsize=(12.6,6.6),gridspec_kw={"width_ratios":[1.05,1.15]})
fig.subplots_adjust(left=.25,right=.94,top=.76,bottom=.22,wspace=.25)
colors={"residual":"#176b78","cqr":"#ad521c"}
fig.suptitle("Archived interval coverage and width",fontweight="bold",fontsize=18,y=.97)
fig.text(.5,.915,"Same 308 FY2020–24 declarations; the $50M+ groups use observed amounts",ha="center",fontsize=11)
labels=[]; ys=[]; j=0
for level in [.8,.9]:
    for scope,title in [("all","all cases"),("50m_or_more","$50M+")]:
        y=j; j+=1;ys.append(y);labels.append(f"{int(level*100)}% nominal · {title}")
        for k,method in enumerate(["residual","cqr"]):
            row=d.query("nominal==@level and scope==@scope and pipeline==@method").iloc[0]
            yy=y+(-.13 if k==0 else .13)
            pct=100*row.covered/row.n
            axes[0].scatter(pct,yy,s=42,color=colors[method],zorder=3)
            axes[0].text(pct+2,yy,f"{row.covered}/{row.n}",va="center",fontsize=8,color=colors[method])
            v=row.median_width_usd/1e6
            axes[1].scatter(v,yy,s=42,color=colors[method],zorder=3)
            axes[1].text(v*1.2,yy,f"{v:,.2f}",va="center",fontsize=8,color=colors[method])
        axes[0].vlines(100*level,y-.31,y+.31,color="#677380",alpha=.75,linestyle="--",linewidth=1)
for ax in axes:
    ax.set_ylim(-.5,3.52);ax.invert_yaxis()
    ax.set_yticks(ys)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(axis="y",length=0)
    ax.grid(axis="x",alpha=.15)
axes[0].set_yticklabels(labels)
axes[1].set_yticklabels([])
axes[0].set_xlim(-4,116)
axes[1].set_xscale("log"); axes[1].set_xlim(.1,10000)
axes[0].set_title("A  Inclusive endpoint coverage",loc="left",fontsize=12,fontweight="bold")
axes[1].set_title("B  Median interval width",loc="left",fontsize=12,fontweight="bold")
axes[0].set_xlabel("Observed coverage (%)");axes[1].set_xlabel("USD millions, log scale")
fig.legend([Line2D([0],[0],color=colors[m],marker="o",linestyle="") for m in ("residual","cqr")],
           ["Nested metadata / dollar residual","Fixed-split CQR"],loc="upper center",bbox_to_anchor=(.55,.885),ncol=2,frameon=False)
fig.text(.092,.09,"Dashed segments indicate nominal targets. Counts use the same cohort; each upper group has only nine cases.",fontsize=9)
fig.text(.092,.055,"Different historical training and calibration pipelines. Not independent deployment validation or a causal comparison.",fontsize=9,fontweight="bold")
fig.savefig(out/"F8-matched-intervals.svg",metadata={"Date":None})
fig.savefig(out/"F8-matched-intervals.png",dpi=200)
print(d.to_string(index=False))
