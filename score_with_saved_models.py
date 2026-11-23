#!/usr/bin/env python3
"""Score a 01-stage skim with saved k-fold models (ensemble average, as bdt_inference does for other_data_files) and write a
wScores file with the same branch layout as the applied run1_fix data file.  Also a --validate mode: score the first N entries
of an already-scored dataset and compare with its stored bdt_score (checks that the saved models are the ones that were applied).
usage: python score_with_saved_models.py -c bdt_cfg.yml --models <dir with model_fold_*.json> -i <skim.root> -o <out_wScores.root>
       python score_with_saved_models.py -c bdt_cfg.yml --models <dir> --validate <01 skim> <applied wScores> [-n 1000000]"""
import argparse, glob, yaml, numpy as np, uproot as ur
from pathlib import Path
from xgboost import Booster
from utils import read_bdt_arrays, save_bdt_arrays
from bdt_inference import predict_score_matrix
APPLIED_BRANCHES=['run','luminosityBlock','event','Bmass','Mll','trigger_OR','Bpt','L1pt','L2pt','L1eta','L2eta','Bcos','Bprob','Kpt','BsLxy',
                  'L2iso','Kiso','LKdz','LKdr','Passymetry','Kip3d','Kip3dErr','L1id','L2id','total_weight']
ap=argparse.ArgumentParser(); ap.add_argument('-c','--config',required=True); ap.add_argument('--models',required=True)
ap.add_argument('-i'); ap.add_argument('-o'); ap.add_argument('--validate',nargs=2); ap.add_argument('-n',type=int,default=None); a=ap.parse_args()
cfg=yaml.safe_load(open(a.config)); ds=cfg['datasets']; feats=cfg['model']['features']; presel=cfg['model'].get('preselection') or None
models=[]
for f in sorted(glob.glob(a.models+'/model_fold_*.json'),key=lambda s:int(Path(s).stem.split('_')[-1])):
    m=Booster(); m.load_model(f); models.append(m)   # Booster: model_predict_scores uses inplace_predict with best_iteration (sklearn wrapper loses n_classes_ on load)
print('loaded',len(models),'fold models from',a.models)
if a.validate:
    skim,applied=a.validate
    X,_,_=read_bdt_arrays(skim,ds['tree_name'],feats,None,presel,(ds['b_mass_branch'],ds['ll_mass_branch']),n_evts=a.n)
    S=predict_score_matrix(models,X); avg=S.mean(axis=1)
    stored=ur.open(applied)['Events']['bdt_score'].array(library='np',entry_stop=a.n)
    k=ur.open(skim)['Events'].arrays(['run','luminosityBlock','event'],library='np',entry_stop=a.n); k2=ur.open(applied)['Events'].arrays(['run','luminosityBlock','event'],library='np',entry_stop=a.n)
    same_key=np.mean((k['run']==k2['run'])&(k['luminosityBlock']==k2['luminosityBlock'])&(k['event']==k2['event']))
    d=avg-stored; print(f'n={len(avg)}  same event keys {same_key:.4f}  |avg-stored|: max {np.abs(d).max():.3e} mean {np.abs(d).mean():.3e}  frac within 1e-4: {np.mean(np.abs(d)<1e-4):.4f}')
    # per-fold: the stored OOF score of a training event equals ONE fold model's score; for non-training events it is the average
    best=np.min(np.abs(S-stored[:,None]),axis=1); print(f'frac matching some single fold model within 1e-5: {np.mean(best<1e-5):.4f}; frac matching ensemble avg within 1e-5: {np.mean(np.abs(d)<1e-5):.4f}')
    cut=0.998; print(f'pass fraction at {cut}: stored {np.mean(stored>cut):.5f}  ensemble {np.mean(avg>cut):.5f}')
else:
    X,_,_=read_bdt_arrays(a.i,ds['tree_name'],feats,None,presel,(ds['b_mass_branch'],ds['ll_mass_branch']),n_evts=a.n)
    S=predict_score_matrix(models,X); avg=S.mean(axis=1).astype(np.float64); print('scored',len(avg),'entries; pass@0.998:',np.mean(avg>0.998))
    save_bdt_arrays(a.i,ds['tree_name'],a.o,ds['tree_name'],APPLIED_BRANCHES,'bdt_score',avg,preselection=presel,n_evts=a.n); print('wrote',a.o)
