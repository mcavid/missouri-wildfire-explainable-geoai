"""Optional reproduction of recorded nested RF; frozen artifacts are not replaced."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score,f1_score
from project import ROOT,REPRO,load_model_data

GRID={"n_estimators":[300,500],"max_depth":[8,10,None],"min_samples_leaf":[1,3],"max_features":["sqrt"],"class_weight":["balanced"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--workers",type=int,default=4);args=parser.parse_args()
    if args.workers<1:parser.error("--workers must be positive")
    frame,X,y,groups=load_model_data()
    outer=pd.read_csv(REPRO/"outer_fold_assignments.csv").sort_values("row_index")
    inner=pd.read_csv(REPRO/"inner_fold_assignments.csv")
    if not np.array_equal(outer.grid_id,frame.grid_id):raise ValueError("Outer identities changed")
    out=ROOT/"results/new_runs/nested_rf";out.mkdir(parents=True,exist_ok=True)
    if any(out.glob("*.csv")):raise FileExistsError("Preserve/move the existing new run before refitting")
    scores=np.zeros(len(frame));predictions=np.zeros(len(frame),dtype=int);metrics=[];winners=[]
    for fold in range(1,6):
        test=np.flatnonzero(outer.outer_fold.to_numpy()==fold);train=np.flatnonzero(outer.outer_fold.to_numpy()!=fold)
        assignment=inner.loc[inner.outer_fold==fold].sort_values("outer_train_row_position")
        if not np.array_equal(assignment.row_index,train):raise ValueError("Inner identities changed")
        membership=assignment.inner_validation_fold.to_numpy()
        splits=[(np.flatnonzero(membership!=k),np.flatnonzero(membership==k)) for k in range(1,5)]
        for a,b in splits:
            if set(groups.iloc[train[a]])&set(groups.iloc[train[b]]):raise ValueError("Inner blocks overlap")
        search=GridSearchCV(RandomForestClassifier(random_state=42,n_jobs=-1),GRID,scoring="roc_auc",cv=splits,n_jobs=args.workers,refit=True)
        search.fit(X.iloc[train],y.iloc[train]);model=search.best_estimator_
        probability=model.predict_proba(X.iloc[test])[:,1];prediction=model.predict(X.iloc[test])
        scores[test]=probability;predictions[test]=prediction
        metrics.append({"fold":fold,"roc_auc":roc_auc_score(y.iloc[test],probability),"accuracy":accuracy_score(y.iloc[test],prediction),"precision":precision_score(y.iloc[test],prediction),"recall":recall_score(y.iloc[test],prediction),"f1":f1_score(y.iloc[test],prediction)})
        winners.append({"fold":fold,"inner_best_auc":search.best_score_,**search.best_params_})
        pd.DataFrame(search.cv_results_).to_csv(out/f"fold_{fold}_candidates.csv",index=False)
        print(f"Fold {fold}: ROC-AUC={metrics[-1]['roc_auc']:.9f}",flush=True)
    pd.DataFrame(metrics).to_csv(out/"fold_results.csv",index=False);pd.DataFrame(winners).to_csv(out/"best_parameters.csv",index=False)
    pd.DataFrame({"grid_id":frame.grid_id,"oof_probability":scores,"oof_prediction":predictions}).to_csv(out/"oof_predictions.csv",index=False)
    difference=float(np.max(np.abs(scores-frame.rf_nested_oof_prob.to_numpy())))
    match=bool(np.allclose(scores,frame.rf_nested_oof_prob,rtol=0,atol=1e-12) and np.array_equal(predictions,frame.rf_nested_oof_pred))
    result={"saved_oof_match":match,"maximum_probability_difference":difference,"pooled_auc":float(roc_auc_score(y,scores)),"grid_candidates":12,"outer_folds":5,"inner_folds":4,"source":"historical notebook cells 126–127, using frozen equivalent memberships"}
    (out/"comparison.json").write_text(json.dumps(result,indent=2)+"\n")
    if not match:raise RuntimeError("New run differs from the frozen results; see comparison.json")
    print("Nested RF reproduction matches the frozen OOF results.")


if __name__=="__main__":main()
