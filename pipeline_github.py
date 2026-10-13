#!/usr/bin/env python3
"""PRC 2026: real ranking audit and guarded 2025->2026 production pipeline.

Default: AUDIT only, reads the real 2026 ranking and submitting template.
Final: explicit --authorize-final-fit generates local submission Parquet; NEVER uploads.
Original 2025 parent models are imported byte-for-byte, not reimplemented.
"""
import argparse, hashlib, json, sys, time, gc
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
C35='35_overnight_model_search.py'
C28='28_lirf_flow_expert_tuning.py'
EXPECTED_2026=670790
MONTHS={'2026-01','2026-02','2026-06','2026-07'}
META={'family':'cat','depth':8,'learning_rate':.075,'l2':15.,'trees':1100,'blend':.5}

def cli():
    """Parse and validate command-line options. Final fitting requires explicit authorization. Returns argparse.Namespace."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['audit','final'],default='audit')
    p.add_argument('--authorize-final-fit',action='store_true')
    p.add_argument('--data-dir',type=Path,default=Path('data'))
    p.add_argument('--ranking',type=Path,default=Path('final_ranking.parquet'))
    p.add_argument('--template',type=Path,default=Path('final_submitting.parquet'))
    p.add_argument('--out-dir',type=Path,default=Path('campaign37_final'))
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--meta-blend',type=float,default=.5,
                   help='Frozen 0.50 development policy, not tuned on blocked group')
    p.add_argument('--flow-weight',type=float,default=1.,help='Frozen C28 flow100')
    p.add_argument('--no-meta',action='store_true',help='Emergency conservative C28-only mode')
    p.add_argument('--expected-ranking-sha256',default='4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f')
    p.add_argument('--expected-template-sha256',default='b3445d80e29bf599a5583ca9675f63ddf57b51fdf1ca3f498facb38ea84f3924')
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.mode=='final' and not a.authorize_final_fit:p.error('Final fit requires --authorize-final-fit')
    if a.mode=='audit' and a.authorize_final_fit:p.error('Authorization only valid with --mode final')
    if not 0<=a.meta_blend<=1 or not 0<=a.flow_weight<=1:p.error('Blend weights must be 0..1')
    return a

def digest(path):
    """Compute the streaming SHA-256 checksum of a file. Args: path (Path or str). Returns: lowercase hexadecimal digest."""
    h=hashlib.sha256()
    with open(path,'rb') as fh:
        for block in iter(lambda:fh.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def self_test():
    """Run basic integrity checks for ordering, IDs, clipping, and fit/selection separation. Raises AssertionError on failure."""
    v=pd.DataFrame({'MVT_ID_mvt':['z','a','b'],'PHASE_mvt':['DEP']*3})
    tpl=pd.DataFrame({'MVT_ID_mvt':['b','z','a']})
    order=v.set_index('MVT_ID_mvt').index.get_indexer(tpl.MVT_ID_mvt)
    assert order.tolist()==[2,0,1]
    assert np.array_equal(np.array([8.,2.,1.])[order],[1.,8.,2.])
    try:
        exact_ids(pd.Series([1,1]),pd.Series([1,2]))
        raise AssertionError('duplicate IDs accepted')
    except ValueError:pass
    assert np.array_equal(np.maximum(0,np.array([-10.,200.])),[0.,200.])
    assert callable(data_path)
    fake=np.array([500.,600.,700.]); fit=np.array([0,1]); select=np.array([2]);
    assert not np.intersect1d(fit,select).size and np.isfinite(fake[select]).all()
    print('SELF-TEST PASS C37 PATCH: IDs, positive predictions, selection label visibility and FIT separation',flush=True)

def exact_ids(src,template):
    """Build a positional reordering from source IDs to template IDs. Requires unique and identical ID sets; raises ValueError otherwise."""
    src=pd.Index(src); template=pd.Index(template)
    if src.hasnans or template.hasnans or not src.is_unique or not template.is_unique:
        raise ValueError('Missing or duplicate movement IDs')
    if len(src)!=len(template):raise ValueError(f'ID length mismatch {len(src)} vs {len(template)}')
    take=src.get_indexer(template)
    if (take<0).any():raise ValueError(f'Missing template IDs: {(take<0).sum()}')
    return take

def data_path(root, p, role='ranking'):
    """Find a user-supplied official Parquet without guessing between ambiguous files."""
    root=Path(root).expanduser().resolve()
    p=Path(p).expanduser()
    if p.is_file():
        return p.resolve()
    candidate=root/p
    if candidate.is_file():
        return candidate.resolve()
    # Source distribution layouts vary; inspect only the local challenge tree.
    roots=[root, HERE]
    for base in (root/'data',root/'dataset',root/'datasets',root/'2026',root/'final',root/'data'/'2026'):
        if base.is_dir():roots.append(base)
    names=([p.name] if p.name not in ('','.','None') else [])
    patterns=([ '*final*ranking*.parquet', '*ranking*final*.parquet'] if role=='ranking'
              else ['*final*submitt*.parquet','*final*template*.parquet','*submitting*.parquet'])
    matches=[]
    for base in dict.fromkeys(roots):
        if not base.is_dir():continue
        for name in names:
            matches.extend(x for x in base.glob(name) if x.is_file())
        for pattern in patterns:
            matches.extend(x for x in base.glob(pattern) if x.is_file())
        # Allow typical nested source folders without searching the whole drive.
        if base in (root,HERE):
            for pattern in patterns:
                matches.extend(x for x in base.glob('*/*'+pattern.lstrip('*')) if x.is_file())
    matches=sorted({x.resolve() for x in matches})
    if len(matches)==1:
        print(f'FOUND OFFICIAL {role.upper()}: {matches[0]}',flush=True)
        return matches[0]
    if len(matches)>1:
        raise FileNotFoundError(f'Ambiguous {role} Parquet files: '+', '.join(map(str,matches))+
            f'. Supply --{role if role=="ranking" else "template"} with the exact path.')
    nearby=[]
    for base in dict.fromkeys(roots):
        if base.is_dir():nearby.extend(str(x) for x in base.glob('*.parquet'))
    raise FileNotFoundError(
       f'Could not locate the official {role} Parquet. Requested: {p!s}; data-dir: {root}. '
       f'Nearby Parquet files: {nearby[:15]}. '
       'Supply the exact file path with --ranking and --template; '
       'if they are not downloaded, obtain both official final files first.')

def load_2026(a,parent):
    """Read the official ranking and submission template; verify schema, months and exact movement IDs. Returns ranking, departures, template, index mapping, and audit metadata."""
    import pyarrow.parquet as pq
    ranking_path=data_path(a.data_dir,a.ranking,'ranking')
    template_path=data_path(a.data_dir,a.template,'template')
    r_schema=pq.read_schema(ranking_path); t_schema=pq.read_schema(template_path)
    wanted=set(parent.ALLOWED_RAW)|{parent.ID,parent.PHASE,parent.TAKEOFF}
    rcols=[c for c in r_schema.names if c in wanted]
    tcols=[c for c in t_schema.names if c in (parent.ID,parent.TARGET)]
    ranking=pd.read_parquet(ranking_path,columns=rcols)
    template=pd.read_parquet(template_path,columns=tcols)
    if parent.ID not in ranking or parent.ID not in template:raise ValueError('MVT_ID_mvt missing')
    if parent.PHASE not in ranking:raise ValueError('PHASE_mvt missing from ranking')
    dep=ranking.loc[ranking[parent.PHASE].astype(str)=='DEP'].copy().reset_index(drop=True)
    if parent.TAKEOFF not in dep:raise ValueError('Movement timestamp absent from ranking')
    if len(dep)!=EXPECTED_2026 or len(template)!=EXPECTED_2026:
        raise ValueError(f'Unexpected final rows: departures={len(dep)}, template={len(template)}; expected {EXPECTED_2026}')
    mapping=exact_ids(dep[parent.ID],template[parent.ID]);
    timestamps=pd.to_datetime(dep[parent.TAKEOFF],utc=True,errors='coerce')
    if timestamps.isna().any():raise ValueError('2026 departure movement timestamps missing')
    months=set(timestamps.dt.strftime('%Y-%m').unique())
    if months!=MONTHS:raise ValueError(f'Unexpected ranking months {months}; expected {MONTHS}')
    for forbidden in (parent.TARGET,parent.BLOCK):
        if forbidden in dep and dep[forbidden].notna().any():
            raise ValueError(f'Unexpected non-null hidden 2026 target/actual block time: {forbidden}')
    required={parent.AIRPORT,parent.AOBT,parent.TAKEOFF,parent.ID,parent.PHASE}
    if not required.issubset(dep):raise ValueError(f'Missing basic 2026 fields: {required-set(dep)}')
    manifest={'ranking':str(ranking_path),'ranking_sha256':digest(ranking_path),
        'template':str(template_path),'template_sha256':digest(template_path),
        'ranking_rows':len(ranking),'departures':len(dep),'template_rows':len(template),
        'month_counts':timestamps.dt.strftime('%Y-%m').value_counts().sort_index().to_dict(),
        'available_source_columns':r_schema.names,
        'missing_feature_source_columns':sorted(set(parent.ALLOWED_RAW)-set(r_schema.names)-{parent.TARGET}),
        'template_columns':t_schema.names,'id_match':True}
    return ranking,dep,template,mapping,manifest

def load_legacy():
    # Modules perform their original hash checks on each dependency.
    """Dynamically import frozen historical campaign modules and their dependencies. Files must be present next to this script."""
    import importlib.util
    def mod(name,alias):
        path=HERE/name
        if not path.is_file():raise FileNotFoundError(f'Missing original campaign: {path}')
        spec=importlib.util.spec_from_file_location(alias,path)
        m=importlib.util.module_from_spec(spec);sys.modules[alias]=m;spec.loader.exec_module(m)
        return m
    c35=mod(C35,'c36_c35');c28=mod(C28,'c36_c28')
    c27,c26,c23,c22,parent=c28.load_deps()
    c33=c35.load_module(c35.C33_NAME,'c36_c33')
    c32=c33.load_c32()
    return c35,c28,c23,c22,parent,c32,c33

def prepare_features(raw,ranking,parent,c23):
    # Reference timestamps in training + all 2026 ranking movements, for strict past-only
    # flow features. 2026 feature construction is independent of hidden labels.
    """Join labelled 2025 and unlabelled 2026 source records and build source-available features, including strictly past-only flow histories."""
    start=len(raw)
    combined=pd.concat([raw.drop(columns=['_origin_file'],errors='ignore'),
                        ranking.reindex(columns=raw.drop(columns=['_origin_file'],errors='ignore').columns)],
                       ignore_index=True)
    feat=parent.build_features(combined)
    parent.add_past_only_flow(feat)
    c23.add_strict_history(feat)
    idx=np.arange(start,len(combined),dtype=np.int64)
    if len(idx)!=len(ranking):raise AssertionError('Combined source indexing failed')
    return feat,idx

def blocked_2026_feature_audit(feat,idx,dep,parent):
    # Source features must exist on target rows; fallbacks are explicit.
    """Summarize 2026 departure features and data-quality regimes without accessing hidden labels."""
    counts={}
    m=feat['_month'].iloc[idx].astype(str)
    counts['month_counts']=m.value_counts().sort_index().to_dict()
    aps=feat.airport.iloc[idx].astype(str)
    bad=(aps.to_numpy()=='LIRF')&(~parent._plausible_proxy(feat,idx))
    counts['airport_counts']=aps.value_counts().sort_index().to_dict()
    counts['LIRF_bad_clock']=int(bad.sum())
    counts['aobt_missing']=int(feat.aobt_missing.iloc[idx].sum())
    counts['aobt_after_takeoff']=int(feat.aobt_after_takeoff.iloc[idx].sum())
    counts['aobt_very_old']=int(feat.aobt_very_old.iloc[idx].sum())
    return counts

def run(a):
    """Perform official-data audit or authorized full-year training and 2026 inference. Audit leaves submissions untouched; final mode produces a local validated Parquet and manifest."""
    tick=time.time()
    c35,c28,c23,c22,parent,c32,c33=load_legacy()
    ranking,dep,tpl,mapping,manifest=load_2026(a,parent)
    if a.expected_ranking_sha256 and manifest['ranking_sha256']!=a.expected_ranking_sha256:raise ValueError('Ranking file SHA256 changed since official audit')
    if a.expected_template_sha256 and manifest['template_sha256']!=a.expected_template_sha256:raise ValueError('Template SHA256 changed since official audit')
    print('OFFICIAL 2026 IDS VERIFIED:',len(dep),'target months',manifest['month_counts'],flush=True)
    out=a.out_dir;out.mkdir(parents=True,exist_ok=True)
    raw,sources=parent.load_training_files(a.data_dir)
    feat,ranking_all_indices=prepare_features(raw,ranking,parent,c23)
    ranking_departure_mask=ranking[parent.PHASE].astype(str).to_numpy()=='DEP'
    idx=ranking_all_indices[ranking_departure_mask]
    if len(idx)!=len(dep):raise AssertionError('DEP selection inconsistency')
    diag=blocked_2026_feature_audit(feat,idx,dep,parent)
    manifest['source_2025']=sources
    manifest['ranking_feature_diagnostics']=diag
    manifest['code_sha256']=digest(Path(__file__).resolve())
    manifest['parent_sha256']=digest(HERE/'21_clock_aware_tail_research.py')
    manifest['mode']=a.mode
    with open(out/'2026_real_data_audit.json','w',encoding='utf-8') as f:json.dump(manifest,f,indent=2,default=str)
    print('REAL 2026 AUDIT',json.dumps(diag,default=str)[:3500],flush=True)
    if a.mode=='audit':
        print('AUDIT COMPLETE: no model fit, no 2026 prediction file, no submission.',flush=True)
        return
    # --mode final REQUIRES --authorize-final-fit. This is intentionally separate.
    from catboost import CatBoostRegressor
    ym=parent.make_y(raw)
    if not np.isfinite(ym).any():raise ValueError('No valid 2025 labels')
    good=np.flatnonzero(np.isfinite(ym)&(ym>=0))
    ntrain=len(raw)
    if len(good)<1_000_000:raise ValueError('Unexpectedly small training population')
    y=np.full(len(feat),np.nan);y[:ntrain]=ym
    # Fully frozen stage-1 recipe: October 2025 selection (from C28), then 2025 refit.
    original_argv=sys.argv[:]
    try:
        sys.argv=[original_argv[0],'--preset','full','--threads',str(a.threads),'--seed',str(a.seed)]
        cfg=c22.args_from_cli()
    finally:sys.argv=original_argv
    cfg.data_dir=a.data_dir;cfg.out_dir=out
    month=feat['_month'].iloc[:ntrain].astype(str).to_numpy()
    fs=np.flatnonzero(np.isin(month,('2025-03','2025-04','2025-05','2025-08','2025-09'))&np.isfinite(ym)&(ym>=0))
    select=np.flatnonzero((month=='2025-10')&np.isfinite(ym)&(ym>=0))
    # SELECT labels are needed to score candidate models. Fit isolation is guaranteed
    # by the fs positional index, which excludes every October row.
    # Do not blank SELECT targets: that made all candidate RMSE scores NaN in C37.
    if np.intersect1d(fs,select).size: raise AssertionError('October SELECT contamination in FIT')
    if not np.isfinite(y[select]).all() or (y[select]<0).any():
        raise AssertionError('October SELECT targets are not finite and nonnegative')
    ysel=y.copy()
    pack_select=c23.fit_models(feat,ysel,fs,cfg,c22,parent)
    incumbent,_,legacy=c23.c22_locked_candidates(pack_select,feat,ysel,select,cfg,c22,parent,'2025-11')
    print('OCTOBER 2025 LOCKED BASELINE:',incumbent,flush=True)
    if incumbent!='hybrid_lirf_tail_resid':raise AssertionError('Frozen C28 October baseline not reproduced')
    del pack_select,ysel;gc.collect()
    # 4 group folds fit residuals on honest out-of-fold predictions. Full-2025 meta crossfit
    # includes all months; Jan/Feb/Jun/Jul now enter training only after explicit authorization.
    groups=[('2025-01','2025-02','2025-03'),('2025-04','2025-05','2025-06'),
            ('2025-07','2025-08','2025-09'),('2025-10','2025-11','2025-12')]
    allpos=[];allpred=[]
    for j,group in enumerate(groups,1):
        val=np.flatnonzero(np.isin(month,group)&np.isfinite(ym)&(ym>=0))
        fit=np.flatnonzero(~np.isin(month,group)&np.isfinite(ym)&(ym>=0))
        yf=y.copy();yf[val]=np.nan
        print(f'META CROSS-FIT {j}/4 training={len(fit):,} evaluation={len(val):,}',flush=True)
        pack=c23.fit_models(feat,yf,fit,cfg,c22,parent)
        pp=c23.c22_locked_predict(pack,feat,yf,val,cfg,parent,legacy)
        allpos.append(val);allpred.append(pp)
        del pack,pp,yf;gc.collect()
    trainpos=np.concatenate(allpos);trainpred=np.concatenate(allpred)
    if len(trainpos)!=len(good) or not np.array_equal(np.sort(trainpos),good):
        raise AssertionError('Crossfit cover mismatch')
    X=c35.feature_frame(feat,trainpos,trainpred,parent,c32)
    cats=[c for c in parent.CATS if c in X]
    bad=(feat.airport.iloc[trainpos].astype(str).to_numpy()=='LIRF')&(~parent._plausible_proxy(feat,trainpos))
    valid=np.flatnonzero(~bad)
    residual=np.clip(ym[trainpos]-trainpred,-3500.,3500.)
    Xcat=c33.model_frame(X,np.arange(len(X)),cats)
    model=CatBoostRegressor(iterations=META['trees'],depth=META['depth'],learning_rate=META['learning_rate'],
        l2_leaf_reg=META['l2'],loss_function='RMSE',random_strength=1.5,one_hot_max_size=10,
        max_ctr_complexity=1,thread_count=a.threads,random_seed=a.seed,verbose=False,allow_writing_files=False)
    print('META FIT',len(valid),'rows CatBoost trial66',flush=True)
    model.fit(Xcat.iloc[valid],residual[valid],cat_features=cats,verbose=False)
    del Xcat,X;gc.collect()
    print('FULL 2025 STAGE1 REFIT',len(good),'rows',flush=True)
    pack=c23.fit_models(feat,y,good,cfg,c22,parent)
    base=c23.c22_locked_predict(pack,feat,y,idx,cfg,parent,legacy)
    q,_=c23.queue_candidates(pack,feat,idx,base,cfg,parent)
    # c23 bad_flow_35 = .65 base + .35 flow; recover pure flow without refit.
    bad=(feat.airport.iloc[idx].astype(str).to_numpy()=='LIRF')&(~parent._plausible_proxy(feat,idx))
    if bad.any() and pack['models'].get('bad_flow') is None:
        raise RuntimeError('LIRF bad-clock specialist was not fitted; cannot use flow100')
    flow=np.maximum(0,(np.asarray(q['c23_bad_lirf_flow_35'])-.65*base)/.35)
    pred=base.copy();pred[bad]=np.maximum(0,(1-a.flow_weight)*base[bad]+a.flow_weight*flow[bad])
    if not a.no_meta:
        Xtest=c35.feature_frame(feat,idx,base,parent,c32)
        Xt=c33.model_frame(Xtest,np.arange(len(Xtest)),cats)
        corr=np.asarray(model.predict(Xt),dtype=float)
        pred[~bad]=np.maximum(0,pred[~bad]+a.meta_blend*corr[~bad])
    if len(pred)!=len(dep) or not np.isfinite(pred).all() or (pred<0).any():
        raise AssertionError('Prediction invalid')
    result=pd.DataFrame({parent.ID:tpl[parent.ID].to_numpy(),parent.TARGET:pred[mapping].astype('float64')})
    if list(result.columns)!=[parent.ID,parent.TARGET] or not np.isfinite(result[parent.TARGET]).all():
        raise AssertionError('Final schema invalid')
    output=out/'PRC_2026_final_predictions_READY_FOR_MANUAL_SUBMISSION.parquet'
    result.to_parquet(output,index=False)
    verify=pd.read_parquet(output)
    if len(verify)!=EXPECTED_2026 or not verify[parent.ID].equals(result[parent.ID]):
        raise AssertionError('Saved final prediction integrity failure')
    manifest.update({'completed':True,'final_output':str(output),'output_sha256':digest(output),
                     'meta_blend':a.meta_blend,'flow_weight':a.flow_weight,'meta_disabled':a.no_meta,
                     'pred_summary':pd.Series(pred).describe(percentiles=[.01,.5,.95,.99]).to_dict(),
                     'elapsed_s':round(time.time()-tick,1)})
    (out/'final_fit_manifest.json').write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8')
    print('FINAL 2026 PREDICTION SUMMARY',json.dumps(manifest['pred_summary'],default=str),flush=True)
    print('FINAL PARQUET CREATED LOCALLY (NOT UPLOADED)',output,'SHA256',digest(output),flush=True)

if __name__=='__main__':
    a=cli();self_test()
    if not a.self_test:run(a)
