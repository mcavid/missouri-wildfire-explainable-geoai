"""Verify the publication checkout without training models or rewriting results."""
from pathlib import Path
import ast
import hashlib
import json
import re
import sys
import yaml
from PIL import Image
import pandas as pd
import numpy as np

from project import ROOT,FEATURES
import verify_publication as original


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    records=[]
    package_path = ROOT / "PACKAGE_MANIFEST.json"
    if package_path.is_file():
        package = json.loads(package_path.read_text())
        for item in package["files"]:
            p = ROOT / item["path"]
            if not p.is_file() or p.stat().st_size != item["bytes"] or digest(p) != item["sha256"]:
                raise ValueError("Package file changed: " + item["path"])
        records.append({"name":"complete_local_package_integrity","files":len(package["files"]),"status":"passed"})
    manifest=json.loads((ROOT/"RESEARCH_MANIFEST.json").read_text())
    for item in manifest["files"]:
        p=ROOT/item["path"]
        if not p.is_file() or p.stat().st_size!=item["bytes"] or digest(p)!=item["sha256"]:raise ValueError("Research artifact changed: "+item["path"])
    records.append({"name":"frozen_artifact_integrity","files":len(manifest["files"]),"status":"passed"})
    for name,check in [("notebook_and_code",original.verify_notebook),("model_matrix_and_folds",original.verify_data_and_folds),("saved_validation_metrics",original.verify_validation),("full_data_shap",original.verify_shap)]:
        records.append({"name":name,"status":"passed","details":check()});print("PASS",name,flush=True)
    importance=pd.read_csv(ROOT/"results/shap_spatial_fold_stability.csv",float_precision="round_trip").set_index("Unnamed: 0")
    folds=[f"Fold_{i}" for i in range(1,6)];ranks=importance[folds].rank(ascending=False,method="average")
    assert np.allclose(ranks.mean(axis=1),importance.Mean_Rank,rtol=0,atol=1e-12)
    assert np.array_equal((ranks<=10).sum(axis=1),importance.Top10_Count)
    assert np.allclose(importance[folds].mean(axis=1),importance.Mean_SHAP,rtol=0,atol=1e-12)
    saved=pd.read_csv(ROOT/"results/shap_fold_rank_correlations.csv",float_precision="round_trip").set_index("Unnamed: 0")
    assert np.allclose(ranks.corr(),saved.loc[folds,folds],rtol=0,atol=1e-12)
    records.append({"name":"fold_summary_rank_arithmetic","status":"passed","features":33,"folds":5})
    pairs=[]
    for label in [f"Figure_{i:02d}" for i in range(1,9)]+[f"Figure_S{i}" for i in range(1,6)]:
        png,pdf=ROOT/"figures/paper"/(label+".png"),ROOT/"figures/paper"/(label+".pdf")
        with Image.open(png) as image:
            assert min(image.size)>1000 and min(image.info.get("dpi",(0,0)))>=599
            pixels=list(image.size)
        assert pdf.read_bytes().startswith(b"%PDF")
        pairs.append({"figure":label,"pixels":pixels})
    records.append({"name":"13_paper_figure_pairs","status":"passed","figures":pairs})
    citation=yaml.safe_load((ROOT/"CITATION.cff").read_text())
    assert citation["cff-version"]=="1.2.0" and citation["license"]=="MIT"
    assert [(x["given-names"],x["family-names"]) for x in citation["authors"]]==[("Mehdi","Javid"),("Reda","Amer")]
    assert citation["preferred-citation"]["type"]=="unpublished" and "doi" not in citation
    workflow=yaml.load((ROOT/".github/workflows/verify.yml").read_text(),Loader=yaml.BaseLoader)
    assert workflow["permissions"]=={"contents":"read"} and set(workflow["on"])=={"push","pull_request","workflow_dispatch"}
    for step in workflow["jobs"]["verify"]["steps"]:
        if "uses" in step:assert re.search(r"@[a-f0-9]{40}$",step["uses"])
    records.append({"name":"citation_license_and_read_only_ci","status":"passed"})
    for script in (ROOT/"scripts").glob("*.py"):ast.parse(script.read_text())
    report={"status":"passed","checks":records,"models_fitted":False,"raw_inputs_read":False}
    out=ROOT/"results/checks";out.mkdir(parents=True,exist_ok=True)
    (out/"repository_verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print("PASS: repository verification complete; no models fitted",flush=True)


if __name__=="__main__":main()
