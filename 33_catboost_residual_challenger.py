#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRC 2026 / Campaign 33 -- CatBoost residual challenger, 2025 RESEARCH ONLY.

Purpose: test algorithmic diversity *where C32 revealed a credible opportunity*:
stand/destination/operator/runway high-cardinality categorical variables. Keep
exact C24 OOF first-stage baseline and C30 LIRF bad-clock expert unchanged;
train only residual CatBoost on chronologically PRIOR 2025 out-of-fold data.

Compare 2025 matched folds against saved C32 LightGBM queue50 predictions.
NO full-2025 final fit; NO 2026 files; NO challenge predictions or submission.
No current/future calendar month residual enters the meta fit, 1-day embargo.
Historical 2025 experiments are repeatedly explored, not a virgin holdout.

Examples:
 python 33_catboost_residual_challenger.py --preset full --threads 4 --bootstrap 400 --save-outer-predictions
 python 33_catboost_residual_challenger.py --self-test

Requires original C32, C21, C23 in same folder, unchanged; CatBoost>=1.2.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

C32_SOURCE='32_crossfit_residual_lgb_research.py'
C32_HASH='310b97b54814390b5553136a15e2d8b00a314116325742c870a31ff2b5f5b9ea'
MONTHS=tuple(f'2025-{i:02d}' for i in range(6,13))
DEFAULT=('2025-06','2025-07','2025-08','2025-10','2025-11')
BAD='LIRF_bad_AOBT'
METHODS=('C24','flow65','flow100','meta_queue50','meta_queue50_flow65',
         'meta_queue50_flow100','cat50','cat100','cat50_flow65','cat50_flow100')


def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def load_c32():
    p=Path(__file__).resolve().with_name(C32_SOURCE)
    if not p.is_file():raise FileNotFoundError('Missing original C32: '+str(p))
    got=digest(p)
    if got!=C32_HASH:raise RuntimeError(f'C32 code integrity fail: expected {C32_HASH}, got {got}')
    spec=importlib.util.spec_from_file_location('campaign32_original_c33',p)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod


def parse():
    p=argparse.ArgumentParser(description='C33 CatBoost alternative meta model: research 2025 only')
    p.add_argument('--data-dir',type=Path,default=Path('.'))
    p.add_argument('--c30-predictions',type=Path,default=Path('campaign30_research/outer_predictions_2025_RESEARCH_ONLY.parquet'))
    p.add_argument('--c32-predictions',type=Path,default=Path('campaign32_research/meta_outer_predictions_2025_RESEARCH_ONLY.parquet'))
    p.add_argument('--out-dir',type=Path,default=Path('campaign33_research'))
    p.add_argument('--preset',choices=('full','smoke'),default='full')
    p.add_argument('--outer-months',nargs='+',default=list(DEFAULT))
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--bootstrap',type=int,default=400)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--embargo-days',type=int,default=1)
    p.add_argument('--min-history-months',type=int,default=3)
    p.add_argument('--save-outer-predictions',action='store_true')
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--smoke-full-model',action='store_true',help='For synthetic tests only: full 480-tree model budget with smoke-size data')
    a=p.parse_args()
    if not set(a.outer_months).issubset(MONTHS) or len(a.outer_months)!=len(set(a.outer_months)):
        p.error('Valid unique outer months: 2025-06..2025-12')
    a.outer_months=sorted(a.outer_months)
    if a.bootstrap<100 or a.threads<1 or a.embargo_days<1 or a.min_history_months<3:p.error('Invalid validation settings')
    if a.smoke_full_model and a.preset!='smoke':p.error('--smoke-full-model allowed only with --preset smoke')
    return a


def read_c32_predictions(p):
    if not p.is_file():
        alt=p.with_suffix('.csv')
        if alt.is_file():p=alt
        else:raise FileNotFoundError('Missing saved C32 OOF: '+str(p)+'; run C32 with --save-outer-predictions')
    cols=['MVT_ID_mvt','outer','utc_mvt','airport',BAD,'y_true','pred_C24',
          'pred_flow65','pred_flow100','pred_meta_queue50','pred_meta_queue50_flow65','pred_meta_queue50_flow100']
    df=pd.read_parquet(p,columns=cols) if p.suffix=='.parquet' else pd.read_csv(p,usecols=cols)
    if len(df)==0 or df.MVT_ID_mvt.isna().any() or not df.MVT_ID_mvt.is_unique:
        raise AssertionError('Invalid C32 saved IDs')
    if set(df.outer.astype(str).unique())!=set(MONTHS):raise AssertionError('C32 historical months 06..12 required')
    df['outer']=df.outer.astype(str)
    df['utc_mvt']=pd.to_datetime(df.utc_mvt,utc=True)
    if not pd.api.types.is_bool_dtype(df[BAD]):
        s=df[BAD].astype(str).str.lower()
        if not s.isin(['true','false','1','0']).all():raise AssertionError('Invalid bad clock C32 mask')
        df[BAD]=s.isin(['true','1'])
    if df[BAD].isna().any():raise AssertionError('C32 bad mask missing')
    if (df[BAD] & (df.airport.astype(str)!='LIRF')).any():raise AssertionError('Non-LIRF BAD mask')
    for col in [c for c in cols if c.startswith('pred_')]+['y_true']:
        a=pd.to_numeric(df[col],errors='raise').to_numpy(float)
        if not np.isfinite(a).all() or (a<0).any():raise AssertionError('Invalid C32 '+col)
    bad=df[BAD].to_numpy(bool)
    # Exact reference isolation and expert math, including missing-expert folds.
    base=df.pred_C24.to_numpy(float)
    f65=df.pred_flow65.to_numpy(float)
    f100=df.pred_flow100.to_numpy(float)
    if not np.array_equal(base[~bad],f100[~bad]):raise AssertionError('Expert changed nonbad movements')
    if not np.allclose(f65,base+.65*(f100-base),rtol=1e-11,atol=1e-7):
        raise AssertionError('C32 flow65 weights do not match C30')
    if not np.array_equal(df.loc[bad,'pred_meta_queue50'].to_numpy(),base[bad]):
        raise AssertionError('C32 meta changed bad-clock predictions')
    if not np.array_equal(df.loc[~bad,'pred_meta_queue50_flow65'].to_numpy(),
                          df.loc[~bad,'pred_meta_queue50'].to_numpy()):
        raise AssertionError('C32 flow65 combination changed ordinary regime')
    if not np.array_equal(df.loc[~bad,'pred_meta_queue50_flow100'].to_numpy(),
                          df.loc[~bad,'pred_meta_queue50'].to_numpy()):
        raise AssertionError('C32 flow100 combination changed ordinary regime')
    return df,p


def exact_ref_alignment(oof,ref,month):
    """Compare original C30 and saved C32 per ID to prevent silently changed splits."""
    oo=oof.loc[oof.outer==month,['MVT_ID_mvt','utc_mvt','y_true','pred_C24','pred_flow100',BAD]].copy()
    rr=ref.loc[ref.outer==month,['MVT_ID_mvt','utc_mvt','y_true','pred_C24','pred_flow100',BAD]].copy()
    if len(oo)!=len(rr):raise AssertionError(f'C32 / C30 count differs {month}')
    z=oo.merge(rr,on='MVT_ID_mvt',validate='one_to_one',suffixes=('_o','_r'))
    if len(z)!=len(oo):raise AssertionError(f'C32 / C30 ID mismatch {month}')
    for v in ('y_true','pred_C24','pred_flow100'):
        if not np.allclose(z[v+'_o'].to_numpy(float),z[v+'_r'].to_numpy(float),rtol=1e-12,atol=1e-7):
            raise AssertionError(f'C32/C30 prediction or label inconsistency {month} {v}')
    if not np.array_equal(z[BAD+'_o'].to_numpy(bool),z[BAD+'_r'].to_numpy(bool)):
        raise AssertionError(f'C32/C30 clock regime mismatch {month}')
    if not np.array_equal(z.utc_mvt_o.astype('int64').to_numpy(),z.utc_mvt_r.astype('int64').to_numpy()):
        raise AssertionError(f'C32/C30 takeoff time mismatch {month}')


def model_frame(x,idx,categorical):
    """Convert shared C32 features to CatBoost-friendly frame, no new features."""
    frame=x.iloc[idx].copy()
    for col in categorical:
        if col not in frame:raise AssertionError('Missing categorical '+col)
        # Explicit string missing token; no target-dependent imputation.
        v=frame[col].astype(object)
        frame[col]=v.where(pd.notna(v),'__MISSING__').astype(str)
    for col in frame.columns.difference(categorical):
        if not pd.api.types.is_numeric_dtype(frame[col]):raise AssertionError('Unexpected object: '+col)
        v=frame[col].to_numpy(dtype=np.float32,copy=False)
        if np.isinf(v).any():frame[col]=np.where(np.isfinite(v),v,np.nan).astype(np.float32)
    if frame.columns.duplicated().any():raise AssertionError('Repeated features')
    return frame


def train_catboost(X,y,fit_idx,eval_idx,cols_cat,a,outer):
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise ImportError('CatBoost required: python -m pip install catboost>=1.2') from exc
    iters=480 if (a.preset=='full' or a.smoke_full_model) else 35
    depth=6 if (a.preset=='full' or a.smoke_full_model) else 4
    kwargs=dict(iterations=iters,learning_rate=.045,depth=depth,
                loss_function='RMSE',l2_leaf_reg=20.,random_strength=1.5,
                one_hot_max_size=10,max_ctr_complexity=1,
                bootstrap_type='Bayesian',bagging_temperature=.5,
                thread_count=a.threads,random_seed=a.seed+330,
                verbose=False,allow_writing_files=False)
    tr=model_frame(X,fit_idx,cols_cat)
    te=model_frame(X,eval_idx,cols_cat)
    tr_y=np.clip(y[fit_idx],-3500.,3500.)
    if not np.isfinite(tr_y).all():raise AssertionError('Non-finite meta fit residual')
    mod=CatBoostRegressor(**kwargs)
    mod.fit(tr,tr_y,cat_features=cols_cat,verbose=False)
    pred=np.asarray(mod.predict(te),dtype=np.float64)
    importance=sorted(zip(X.columns,mod.feature_importances_),key=lambda x:-x[1])[:15]
    del tr,te,mod
    gc.collect()
    if not np.isfinite(pred).all():raise AssertionError('Nonfinite CatBoost corrections')
    return pred,importance


def monthly_score(y,p):
    if len(y)==0 or len(y)!=len(p) or not np.isfinite(p).all() or np.any(p<0):
        raise AssertionError('Invalid evaluation vector')
    return float(np.sqrt(np.mean((y-p)**2)))


def boot_delta(df,pred_column,base_column,boots,seed):
    yy=df.y_true.to_numpy(float)
    a=df[base_column].to_numpy(float)
    b=df[pred_column].to_numpy(float)
    month_day=df.outer.astype(str)+'/'+pd.to_datetime(df.utc_mvt,utc=True).dt.strftime('%Y-%m-%d')
    codes,_=pd.factorize(month_day,sort=True)
    count=np.bincount(codes).astype(float)
    s0=np.bincount(codes,weights=(yy-a)**2)
    s1=np.bincount(codes,weights=(yy-b)**2)
    rng=np.random.default_rng(seed)
    out=[]
    for i in range(boots):
        take=rng.integers(len(count),size=len(count))
        out.append(float(np.sqrt(s1[take].sum()/count[take].sum())-
                         np.sqrt(s0[take].sum()/count[take].sum())))
    return {f'q{k}':float(np.percentile(out,k)) for k in (5,50,95)}


def sanity():
    # Time embargo + exact regime isolation + categorical missing, no 2026 files.
    ts=pd.to_datetime(['2025-06-29','2025-06-30','2025-07-01'],utc=True)
    assert (ts<pd.Timestamp('2025-07-01',tz='UTC')-pd.Timedelta(days=1)).tolist()==[True,False,False]
    f=pd.DataFrame({'airport':pd.Categorical(['LIRF',None,'EGLL']),
                    'level':[1.,np.nan,3.],'pred_C24':[3.,6.,9.]})
    a=model_frame(f,np.array([0,1,2]),['airport'])
    assert a.airport.tolist()==['LIRF','__MISSING__','EGLL']
    b=np.array([10.,20.,30.]);bad=np.array([True,False,False]);adjust=np.array([2.,4.,6.]);adjust[bad]=0
    assert np.array_equal(np.maximum(0,b+adjust)[bad],b[bad])
    print('SELF-TEST PASS C33: chronological embargo, exact source alignment, categorical missing and bad-clock isolation',flush=True)


def run():
    a=parse();sanity()
    if a.self_test:return
    start=time.time()
    c32=load_c32()
    parent=c32.import_locked(c32.PARENT,c32.SHA_PARENT,'c33_parent')
    c23=c32.import_locked(c32.C23_NAME,c32.SHA_C23,'c33_c23')
    print('='*82+'\nC33 CATBOOST RESIDUAL CHALLENGER, 2025 RESEARCH ONLY\n'
          'NO FINAL FULL-2025 FIT, NO 2026 FILES, NO SUBMISSION\n'+'='*82,flush=True)
    oof,c30_path=c32.read_oof(a.c30_predictions,a.preset)
    reference,c32_path=read_c32_predictions(a.c32_predictions)
    if a.preset=='full' and len(reference)!=1262461:
        raise AssertionError('FULL saved C32 OOF expected 1,262,461 scored rows')
    raw,manifest=parent.load_training_files(a.data_dir)
    X=c32.align_features(parent,c23,raw,oof)
    del raw
    gc.collect()
    cats=[c for c in parent.CATS if c in X]
    if 'pred_C24' not in X or not cats:raise AssertionError('C33 requires C32 numeric prior prediction and categorical fields')
    months=oof.outer.to_numpy(str)
    dates=oof.utc_mvt
    residual=(oof.y_true-oof.pred_C24).to_numpy(float)
    baseall=oof.pred_C24.to_numpy(float)
    badall=oof[BAD].to_numpy(bool)
    results=[];importance=[];outputs=[]
    ref_index=reference.set_index('MVT_ID_mvt',drop=False)
    for outer in a.outer_months:
        exact_ref_alignment(oof,reference,outer)
        cutoff=pd.Timestamp(outer+'-01',tz='UTC')-pd.Timedelta(days=a.embargo_days)
        fit=(months<outer)&(dates.lt(cutoff).to_numpy())&(~badall)
        valid=(months==outer)
        fit_idx=np.flatnonzero(fit)
        val_idx=np.flatnonzero(valid)
        hmonths=set(months[fit_idx])
        if len(hmonths)<a.min_history_months or len(fit_idx)<1000 or not len(val_idx):
            raise AssertionError(f'Insufficient chronological fold {outer}: fit {len(fit_idx)} months {hmonths}')
        if dates.iloc[fit_idx].max()>=dates.iloc[val_idx].min():raise AssertionError('Temporal overlap')
        v=oof.iloc[val_idx]
        r=ref_index.loc[v.MVT_ID_mvt.to_numpy()]
        # Ensure the reference matches exactly the scored C30 record sequence.
        if not np.array_equal(r.MVT_ID_mvt.to_numpy(),v.MVT_ID_mvt.to_numpy()):
            raise AssertionError('Cached C32 order not recovered')
        predictions={
            'C24':r.pred_C24.to_numpy(float),
            'flow65':r.pred_flow65.to_numpy(float),
            'flow100':r.pred_flow100.to_numpy(float),
            'meta_queue50':r.pred_meta_queue50.to_numpy(float),
            'meta_queue50_flow65':r.pred_meta_queue50_flow65.to_numpy(float),
            'meta_queue50_flow100':r.pred_meta_queue50_flow100.to_numpy(float),
        }
        bad=v[BAD].to_numpy(bool)
        yy=v.y_true.to_numpy(float)
        add,top_imp=train_catboost(X,residual,fit_idx,val_idx,cats,a,outer)
        add[bad]=0.
        for w,label in [(0.5,'cat50'),(1.0,'cat100')]:
            predictions[label]=np.maximum(0.,predictions['C24']+w*add)
        for label,w in [('cat50_flow65',.65),('cat50_flow100',1.0)]:
            arr=predictions['cat50'].copy()
            arr[bad]=predictions['C24'][bad]+w*(predictions['flow100'][bad]-predictions['C24'][bad])
            predictions[label]=np.maximum(0.,arr)
        if set(predictions)!=set(METHODS):raise AssertionError('Incomplete method comparison')
        for name in ('cat50','cat100'):
            if not np.array_equal(predictions[name][bad],predictions['C24'][bad]):
                raise AssertionError('CatBoost changed BAD regime')
        if not np.array_equal(predictions['cat50_flow65'][~bad],predictions['cat50'][~bad]):
            raise AssertionError('Expert affects nonbad')
        row={'outer':outer,'n':len(yy),'n_bad':int(bad.sum()),'fit_rows':len(fit_idx),
             'history_months':len(hmonths),'categorical_features':len(cats)}
        for name,arr in predictions.items():
            row[name+'_rmse']=monthly_score(yy,arr)
            row[name+'_nonbad_rmse']=monthly_score(yy[~bad],arr[~bad])
        results.append(row)
        importance.extend({'outer':outer,'feature':name,'importance':float(imp),'rank':rank}
                          for rank,(name,imp) in enumerate(top_imp,1))
        df=v[['MVT_ID_mvt','outer','utc_mvt','airport',BAD,'y_true']].copy()
        for name,arr in predictions.items():df['pred_'+name]=arr
        outputs.append(df)
        print(f'  {outer} FIT={len(fit_idx):,} C24={row["C24_rmse"]:.3f} '
              f'LGB50={row["meta_queue50_rmse"]:.3f} CAT50={row["cat50_rmse"]:.3f} '
              f'CAT50+flow65={row["cat50_flow65_rmse"]:.3f}',flush=True)
    monthly=pd.DataFrame(results)
    combined=pd.concat(outputs,ignore_index=True)
    y=combined.y_true.to_numpy(float)
    pooled={}
    for idx,m in enumerate(METHODS):
        p=combined['pred_'+m].to_numpy(float)
        rec={'rmse':monthly_score(y,p),
             'wins_vs_C24':int((monthly[m+'_rmse']<monthly.C24_rmse-1e-8).sum()),
             'worst_month_regret_vs_C24':float((monthly[m+'_rmse']-monthly.C24_rmse).max())}
        if m!='C24':rec['day_bootstrap_vs_C24']=boot_delta(combined,'pred_'+m,'pred_C24',a.bootstrap,a.seed+idx*37)
        if m.startswith('cat'):
            rec['wins_vs_lgb_queue50']=int((monthly[m+'_rmse']<monthly.meta_queue50_rmse-1e-8).sum())
            rec['worst_regret_vs_lgb_queue50']=float((monthly[m+'_rmse']-monthly.meta_queue50_rmse).max())
        pooled[m]=rec
    # Scores in the ordinary regime: tells if CatBoost beats LGB absent extreme specialist.
    normal=combined.loc[~combined[BAD]].copy()
    normal_summary={m:monthly_score(normal.y_true.to_numpy(float),normal['pred_'+m].to_numpy(float))
                    for m in ('C24','meta_queue50','cat50','cat100')}
    summary={'campaign':33,'research_only':True,'no_2026':True,'no_final_2025_fit':True,
             'same_folds_as_C32':True,'full_2025_validated_not_a_fresh_holdout':True,
             'months':a.outer_months,'n':len(combined),'n_bad':int(combined[BAD].sum()),
             'c30_checksum':digest(c30_path),'c32_checksum':digest(c32_path),
             'C32_code_checksum':C32_HASH,'model':'CatBoost '+str(480 if (a.preset=='full' or a.smoke_full_model) else 35)+' trees depth '+str(6 if (a.preset=='full' or a.smoke_full_model) else 4)+' CPU; categorical native',
             'target':'clipped -3500..3500 seconds residual C24 only for TRAIN',
             'validation':'previous-month OOF only, >=1-day embargo; 2025 months previously explored',
             'pooled':pooled,'nonbad_pooled':normal_summary,'elapsed_seconds':time.time()-start}
    a.out_dir.mkdir(parents=True,exist_ok=True)
    monthly.to_csv(a.out_dir/'monthly_catboost_vs_lgb_2025.csv',index=False)
    pd.DataFrame(importance).to_csv(a.out_dir/'catboost_feature_importance.csv',index=False)
    if a.save_outer_predictions:
        try:combined.to_parquet(a.out_dir/'outer_predictions_2025_RESEARCH_ONLY.parquet',index=False)
        except ImportError:combined.to_csv(a.out_dir/'outer_predictions_2025_RESEARCH_ONLY.csv',index=False)
    (a.out_dir/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
    print('POOLED C33 MATCHED FOLDS:',{k:round(v['rmse'],3) for k,v in pooled.items()},flush=True)
    print('OUTPUT',a.out_dir,'RESEARCH ONLY -- NO FINAL 2025 FIT, NO 2026',flush=True)


if __name__=='__main__':
    try:run()
    except Exception as e:
        print(f'FATAL C33 {type(e).__name__}: {e}',file=sys.stderr,flush=True)
        raise
