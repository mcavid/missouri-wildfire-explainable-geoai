"""Regenerate all 13 paper figures from processed data, without fitting models."""
from pathlib import Path
import ast
import inspect
import json
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator
from PIL import Image

from project import ROOT
import review_common as common

OUT=ROOT/"figures/generated"
DATA=OUT/"data"
MAPPING={
    "R1_study_area_coordinates":"Figure_01",
    "R5_susceptibility_coordinates":"Figure_05",
    "A1_nested_spatial_errors":"Figure_07",
    "A2_spatial_shap":"Figure_08",
    "S3_reporting_seasonality":"Figure_S1",
    "S1_validation_robustness":"Figure_S2",
    "S4_nested_roc_pr":"Figure_S3",
    "S5_population_road_dependence":"Figure_S4",
    "S2_shap_fold_stability":"Figure_S5",
}
RECORDS={}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig,stem,detail):
    name=MAPPING.get(stem,stem)
    # Ignore locator ticks outside the view: Matplotlib does not render them.
    # Preserve axis limits and all ticks that are actually visible.
    for ax in fig.axes:
        if ax.axison:
            for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
                low, high = sorted(limits)
                axis.set_major_locator(FixedLocator([t for t in axis.get_majorticklocs() if low <= t <= high]))
    fig.canvas.draw();renderer=fig.canvas.get_renderer();w,h=fig.canvas.get_width_height()
    for obj in fig.findobj(Text):
        if obj.get_visible() and obj.get_text().strip():
            box=obj.get_window_extent(renderer)
            if box.x0 < -2 or box.y0 < -2 or box.x1 > w+2 or box.y1 > h+2:
                raise ValueError(f"{name}: text outside canvas: {obj.get_text()}")
    png,pdf=OUT/(name+".png"),OUT/(name+".pdf")
    fig.savefig(png,dpi=600);fig.savefig(pdf,metadata={"Title":name,"Creator":"Missouri wildfire processed-input figure pipeline"})
    if name=="Figure_S1":
        detail={"inputs":[common.input_record(ROOT/"results"/s) for s in ["nifc_annual_record_counts_2015_2026.csv","nifc_created_by_system_2015_2026.csv","monthly_wildfire_distribution_2021_2025.csv"]],"method":"Plot saved reporting counts/proportions; check their arithmetic. Raw incident queries are not executed.","raw_incident_queries_executed":False}
    RECORDS[name]={"detail":detail,"png_pixels":list(Image.open(png).size),"dpi":600,"outputs":[{"path":p.relative_to(ROOT).as_posix(),"sha256":sha(p)} for p in [png,pdf]],"models_fitted":False,"shap_recomputed":False}
    plt.close(fig);print("Generated",name,flush=True)


def table(frame,name):
    # The saved-table reporting adapter must not claim fresh raw SQL validation.
    cols=[c for c in frame if not c.endswith("_match_raw")]
    frame.loc[:,cols].to_csv(DATA/name,index=False)
    return DATA/name


def saved_reporting(function):
    ns=dict(function.__globals__)
    ns["verify_raw_reporting"]=lambda *args:{"raw_incident_queries_executed":False}
    def input_record(path):
        p=Path(path)
        return common.input_record(p) if p.is_file() else {"source_acquisition_file":p.relative_to(ROOT).as_posix(),"read_during_generation":False}
    ns["input_record"]=input_record;ns["save_table"]=table;ns["save_figure"]=save
    exec(compile(ast.parse(inspect.getsource(function)),__file__,"exec"),ns)
    ns[function.__name__]()


def rank_figure(function):
    tree=ast.parse(inspect.getsource(function));node=tree.body[0]
    node.body=[n for n in node.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=="text" and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=="fig")]
    for n in ast.walk(tree):
        if isinstance(n,ast.Constant) and n.value=="Outer fold (saved label)":n.value="Fold (saved label)"
        elif isinstance(n,ast.Constant) and n.value=="Outer fold":n.value="Fold"
    ns=dict(function.__globals__);ns["save_figure"]=save
    exec(compile(ast.fix_missing_locations(tree),__file__,"exec"),ns);ns[function.__name__]()


def main():
    OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(exist_ok=True)
    common.OUT=OUT;common.DATA=DATA;common.save_figure=save;common.save_table=table
    # Import after the output adapter so their bound globals write only generated files.
    import review_spatial as spatial
    import review_diagnostics as diagnostic
    import review_supporting as supporting
    import make_publication_figures as publication
    spatial.study_area_coordinates()
    for n in [2,3,4,6]:
        fig,detail=getattr(publication,f"figure{n}")();save(fig,f"Figure_{n:02d}",detail)
    spatial.susceptibility_coordinates()
    nested,pooled,folds,roc,pr,paths=diagnostic.prepare_diagnostics()
    diagnostic.spatial_errors(nested,pooled,paths)
    spatial.spatial_shap()
    saved_reporting(supporting.s3_reporting)
    supporting.s1_robustness()
    diagnostic.roc_pr(nested,pooled,folds,roc,pr,paths)
    spatial.population_road_dependence()
    rank_figure(supporting.s2_shap_stability)
    expected={f"Figure_{i:02d}" for i in range(1,9)}|{f"Figure_S{i}" for i in range(1,6)}
    if set(RECORDS)!=expected:raise ValueError(f"Unexpected figure set: {set(RECORDS)}")
    (OUT/"figure_manifest.json").write_text(json.dumps({"figures":RECORDS,"processed_inputs_only":True,"models_fitted":False},indent=2,default=common.json_default)+"\n")
    print("All 13 paper figures regenerated from processed inputs.",flush=True)


if __name__=="__main__":main()
