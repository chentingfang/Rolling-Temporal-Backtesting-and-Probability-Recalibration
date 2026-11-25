"""Reproduce the manuscript's archived-forecast analyses.

Run: python reproduce.py
No original fitted estimator, Excel workbook, or precomputed interval is required.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import metrics as mt

HERE = Path(__file__).resolve().parent
BUNDLED_INPUT = HERE / 'data' / 'anonymized_prediction_archive.csv'
BUNDLED_SHA256 = '0cafa218445c8b71804448767afce09d5698a50b73eda52e4c6d284f039c8e39'
DEFAULT_SEED = 20260719
DEFAULT_BOOTSTRAP = 2000
MODELS = {'Logistic':'logistic', 'SVM':'svm', 'Random forest':'rf', 'XGBoost':'xgb'}
QS = (.01, .02, .05, .10)
METRICS = ('auc', 'ap', 'ap_skill', 'brier', 'brier_skill')
VERSION = '1.0.0'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    # Convert nonfinite values to JSON null; never emit nonstandard NaN tokens.
    def clean(value):
        if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
        if isinstance(value, (list,tuple)): return [clean(v) for v in value]
        if isinstance(value, (float,np.floating)): return float(value) if np.isfinite(value) else None
        if isinstance(value, np.integer): return int(value)
        if isinstance(value, np.bool_): return bool(value)
        return value
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_and_validate(path, out):
    d = pd.read_csv(path, dtype={'firm_id':str})
    keys = ['firm_id','horizon','year']
    numeric = ['feature_year','year','horizon','y'] + [f'{prefix}_{slug}' for slug in MODELS.values() for prefix in ['p','threshold','yhat']]
    missing = sorted(set(keys+numeric)-set(d.columns))
    errors = []
    if missing:
        errors.append({'row':None, 'field':','.join(missing), 'reason':'missing_columns'})
    else:
        def flag(mask, field, reason):
            for idx in d.index[np.asarray(mask)]:
                errors.append({'row':int(idx)+2, 'firm_id':d.at[idx,'firm_id'],
                               'horizon':d.at[idx,'horizon'], 'year':d.at[idx,'year'],
                               'field':field, 'reason':reason})
        flag(d.firm_id.isna() | d.firm_id.fillna('').str.strip().eq(''), 'firm_id','missing_identifier')
        for col in numeric:
            d[col] = pd.to_numeric(d[col], errors='coerce')
            flag(~np.isfinite(d[col]),col,'missing_or_nonfinite')
        flag(d.duplicated(keys,keep=False),'+'.join(keys),'duplicate_key')
        flag(~d.horizon.isin([1,2]),'horizon','invalid_horizon')
        flag(~d.y.isin([0,1]),'y','invalid_binary_outcome')
        for col in ['feature_year','year']:
            flag(d[col] != np.floor(d[col]),col,'noninteger_year')
        flag(d.year-d.feature_year != d.horizon,'feature_year','horizon_mismatch')
        flag(~d.year.isin(range(2020,2025)),'year','outside_paper_period')
        for slug in MODELS.values():
            for prefix in ['p','threshold']:
                col=f'{prefix}_{slug}'
                flag(~d[col].between(0,1),col,'outside_probability_range')
            flag(~d['yhat_'+slug].isin([0,1]),'yhat_'+slug,'invalid_binary_prediction')
            flag((d['p_'+slug]>=d['threshold_'+slug]).astype(int)!=d['yhat_'+slug],
                 'yhat_'+slug,'threshold_prediction_mismatch')
            n=d.groupby(['horizon','year'])['threshold_'+slug].transform('nunique')
            flag(n!=1,'threshold_'+slug,'nonunique_annual_threshold')
        cells=set(map(tuple,d[['horizon','year']].drop_duplicates().to_numpy()))
        for h in [1,2]:
            for year in range(2020,2025):
                if (h,year) not in cells:
                    errors.append({'row':None,'horizon':h,'year':year,'field':'year','reason':'missing_annual_cell'})
    if errors:
        pd.DataFrame(errors).to_csv(out/'validation_errors.csv',index=False,encoding='utf-8-sig')
        raise ValueError('Input validation failed. See validation_errors.csv; no automatic row deletion was applied.')
    for col in ['feature_year','year','horizon','y']:
        d[col] = d[col].astype(int)
    d=d.sort_values(keys[1:]+keys[:1],kind='stable').reset_index(drop=True)
    return d, dict(records=len(d), unique_firms=d.firm_id.nunique(), duplicate_keys=0,
                   missing_required_values=0, probability_range_errors=0, threshold_mismatches=0,
                   years=sorted(d.year.unique().tolist()), horizons=[1,2],
                   note='Wide archive has one shared outcome per row. Original pre-join source alignment cannot be re-audited.')


def annual_analysis(d):
    annual, bins, capacities, policies = [], [], [], []
    for (h,t), g in d.groupby(['horizon','year'],sort=True):
        y,ids=g.y.to_numpy(),g.firm_id.to_numpy()
        for model,slug in MODELS.items():
            key=dict(horizon=int(h),year=int(t),model=model)
            p=g['p_'+slug].to_numpy()
            b=mt.reliability_bins(y,p)
            bins.extend([{**key,**row} for row in b])
            cal=mt.calibration(y,p)
            annual.append({**key,'n':len(g),'events':int(y.sum()),'prevalence':float(y.mean()),
                           'mean_probability':float(p.mean()), **mt.scores(y,p),**cal,
                           'ece10':sum(x['n']*abs(x['observed']-x['predicted']) for x in b)/len(y)})
            policies.append({**key,'threshold':float(g['threshold_'+slug].iloc[0]),
                             **mt.classification(y,g['yhat_'+slug].to_numpy())})
            for q in QS:
                row,_=mt.capacity(y,p,ids,q)
                capacities.append({**key,**row})
    return [pd.DataFrame(x) for x in [annual,bins,capacities,policies]]


def recalibration(d):
    rows, audits, lists, predictions = [], [], [], []
    for h in [1,2]:
        hd=d[d.horizon==h]
        for model,slug in MODELS.items():
            for year in range(2021,2025):
                history=hd[hd.year<year]; g=hd[hd.year==year]
                y,p,ids=g.y.to_numpy(),g['p_'+slug].to_numpy(),g.firm_id.to_numpy()
                key=dict(horizon=h,year=year,model=model)
                fit=mt.fit_offset(history.y.to_numpy(),history['p_'+slug].to_numpy())
                updated=mt.sigmoid(mt.logit(p)+fit['alpha']) if fit['converged'] else p.copy()
                raw, rec = mt.scores(y,p), mt.scores(y,updated)
                rows.append({**key, **fit, 'history_years':';'.join(map(str,sorted(history.year.unique()))),
                             'history_n':len(history),'history_events':int(history.y.sum()),'test_n':len(g),
                             'prevalence':float(y.mean()),'raw_mean':float(p.mean()),'updated_mean':float(updated.mean()),
                             'raw_brier':raw['brier'],'updated_brier':rec['brier'],
                             'brier_improvement':raw['brier']-rec['brier'],
                             'raw_log_loss':raw['log_loss'],'updated_log_loss':rec['log_loss'],
                             'log_loss_improvement':raw['log_loss']-rec['log_loss'],
                             'raw_auc':raw['auc'],'updated_auc':rec['auc'],
                             'updated_citl':mt.fit_offset(y,updated)['alpha']})
                old_order,new_order=mt.ranked_indices(p,ids),mt.ranked_indices(updated,ids)
                audits.append({**key,'raw_auc':raw['auc'],'updated_auc':rec['auc'],
                               'delta_auc':rec['auc']-raw['auc'],
                               'auc_equal_within_1e_12':bool(np.isclose(raw['auc'],rec['auc'],atol=1e-12,rtol=0,equal_nan=False)),
                               'full_order_identical':bool(np.array_equal(old_order,new_order)),
                               'logit_clipped_records':int(np.sum((p<mt.LOGIT_EPS)|(p>1-mt.LOGIT_EPS))),
                               'unique_raw_scores':len(np.unique(p)), 'unique_updated_scores':len(np.unique(updated)),
                               'fit_status':fit['status']})
                for q in QS:
                    a,ii=mt.capacity(y,p,ids,q);b,jj=mt.capacity(y,updated,ids,q)
                    old,new=set(ids[ii]),set(ids[jj])
                    lists.append({**key,'q':q,'k':a['k'],'same_membership':old==new,
                                  'same_order':bool(np.array_equal(ids[ii],ids[jj])),
                                  'symmetric_difference':len(old^new),
                                  'raw_events':a['captured'],'updated_events':b['captured']})
                predictions.append(pd.DataFrame({'firm_id':ids,'horizon':h,'year':year,'model':model,
                                                 'y':y,'raw_probability':p,'recalibrated_probability':updated,
                                                 'alpha':fit['alpha'],'fit_status':fit['status']}))
    return pd.DataFrame(rows),pd.DataFrame(audits),pd.DataFrame(lists),pd.concat(predictions,ignore_index=True)


def paired_bootstrap(d, annual, seed, b):
    """Same RNG call order as original year/class resampling; all intervals recomputed."""
    yearly, results = [], []
    for h in [1,2]:
        rng=np.random.default_rng(seed+h)
        gs={int(t):g for t,g in d[d.horizon==h].groupby('year',sort=True)}
        years=sorted(gs)
        arrays={t:(g.y.to_numpy(),np.column_stack([g['p_'+s].to_numpy() for s in MODELS.values()])) for t,g in gs.items()}
        draws=np.empty((b,3,len(METRICS)))
        for replicate in range(b):
            sampled=rng.choice(years,size=len(years),replace=True)
            accum=[]
            for t in sampled:
                y,ps=arrays[int(t)]
                pos,neg=np.flatnonzero(y==1),np.flatnonzero(y==0)
                if not len(pos) or not len(neg):
                    raise ValueError('Paired stratified bootstrap requires both classes in every year.')
                idx=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
                vals=np.array([[s[k] for k in METRICS] for s in [mt.scores(y[idx],ps[idx,j]) for j in range(4)]])
                delta=vals[2]-vals[[0,1,3]]
                delta[:,METRICS.index('brier')]*=-1
                accum.append(delta)
            draws[replicate]=np.mean(accum,axis=0)
            if (replicate+1)%500==0:
                print(f'  horizon {h}: {replicate+1}/{b} paired bootstrap replicates',flush=True)
        for j,model in enumerate(['Logistic','SVM','XGBoost']):
            for k,metric in enumerate(METRICS):
                deltas=[]
                for t in years:
                    ref=annual[(annual.horizon==h)&(annual.year==t)&(annual.model=='Random forest')].iloc[0][metric]
                    comp=annual[(annual.horizon==h)&(annual.year==t)&(annual.model==model)].iloc[0][metric]
                    value=comp-ref if metric=='brier' else ref-comp
                    deltas.append(value)
                    yearly.append(dict(horizon=h,year=t,reference='Random forest',comparator=model,metric=metric,difference=value))
                v=draws[:,j,k]
                results.append(dict(horizon=h,reference='Random forest',comparator=model,metric=metric,
                                    difference_favoring_reference=float(np.mean(deltas)),
                                    ci_low=float(np.quantile(v,.025)),ci_high=float(np.quantile(v,.975)),
                                    bootstrap_fraction_positive=float(np.mean(v>0)),replicates=b,seed=seed+h))
    return pd.DataFrame(yearly),pd.DataFrame(results)


def recalibration_bootstrap(r,seed,b):
    rng=np.random.default_rng(seed+99)
    rows=[]
    for h in [1,2]:
        for model in MODELS:
            g=r[(r.horizon==h)&(r.model==model)].sort_values('year')
            if not g.converged.all():
                # Do not silently mix failed fits with successful updates in inference.
                continue
            idx=rng.choice(len(g),size=(b,len(g)),replace=True)
            for metric in ['brier','log_loss']:
                v=g[metric+'_improvement'].to_numpy(); draws=v[idx].mean(axis=1)
                rows.append(dict(horizon=h,model=model,metric=metric,mean_improvement=float(v.mean()),
                                 ci_low=float(np.quantile(draws,.025)),ci_high=float(np.quantile(draws,.975)),
                                 bootstrap_fraction_positive=float(np.mean(draws>0)),replicates=b,seed=seed+99,
                                 note='Four target-year blocks; intercepts not refitted inside bootstrap; not firm-clustered.'))
    return pd.DataFrame(rows)


def manuscript_checks(d,a,r,audit,topk,p):
    """Regression checks of selected published values, never inputs to estimation."""
    checks=[]
    def check(name,observed,expected,tolerance):
        checks.append(dict(check=name,observed=observed,expected=expected,tolerance=tolerance,
                           passed=bool(abs(observed-expected)<=tolerance)))
    check('records',len(d),21245,0);check('unique_firms',d.firm_id.nunique(),3377,0)
    for h,auc,bs in [(1,.9117870985730123,.01299938844991735),(2,.846304186397796,.016391692040576854)]:
        check(f'RF mean annual AUC h={h}',a[(a.horizon==h)&(a.model=='Random forest')].auc.mean(),auc,1e-10)
        check(f'XGBoost mean Brier improvement h={h}',r[(r.horizon==h)&(r.model=='XGBoost')].brier_improvement.mean(),bs,1e-10)
    check('AUC audit rows',len(audit),32,0)
    check('max absolute AUC change',audit.delta_auc.abs().max(),0,1e-12)
    check('changed Top-k memberships',int((~topk.same_membership).sum()),0,0)
    paired_targets = [
        (1, 'auc', (.016, -.008, .047), .0005000001),
        (2, 'auc', (.035, -.001, .078), .0005000001),
        (1, 'brier', (.0134, .0105, .0163), .0000500001),
        (2, 'brier', (.0145, .0103, .0187), .0000500001),
    ]
    for h,metric,targets,tolerance in paired_targets:
        row=p[(p.horizon==h)&(p.comparator=='XGBoost')&(p.metric==metric)].iloc[0]
        estimates=(row.difference_favoring_reference,row.ci_low,row.ci_high)
        for label,actual,target in zip(('difference','CI low','CI high'),estimates,targets):
            check(f'RF vs XGBoost {metric.upper()} h={h} {label}',actual,target,tolerance)
    return pd.DataFrame(checks)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=BUNDLED_INPUT)
    parser.add_argument('--output',type=Path,default=HERE/'results')
    parser.add_argument('--seed',type=int,default=DEFAULT_SEED)
    parser.add_argument('--bootstrap',type=int,default=DEFAULT_BOOTSTRAP)
    parser.add_argument('--no-figures',action='store_true')
    args=parser.parse_args()
    if args.bootstrap<1:parser.error('--bootstrap must be positive')
    if args.seed<0:parser.error('--seed must be nonnegative')
    input_path=args.input.resolve()
    if not input_path.is_file():parser.error(f'Input CSV does not exist: {input_path}')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    # A new run must not leave old successful artifacts masquerading as new results.
    if any(out.iterdir()):
        parser.error('Output folder is not empty. Use --output with a new folder to preserve earlier runs.')
    input_hash=sha(input_path)
    matches_bundled=input_hash==BUNDLED_SHA256
    verify_reference=matches_bundled and args.seed==DEFAULT_SEED and args.bootstrap==DEFAULT_BOOTSTRAP
    config=dict(version=VERSION,status='running',started_at_utc=datetime.now(timezone.utc).isoformat(),
                input_file=input_path.name,input_sha256=input_hash,
                bundled_input_sha256_matches=matches_bundled,seed=args.seed,bootstrap=args.bootstrap,
                manuscript_checks_applicable=verify_reference,manuscript_checks_all_passed=None,
                python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,
                source_hashes={p.name:sha(p) for p in sorted(HERE.glob('*.py'))},
                analysis_mode='retrospective_target_year_order',
                prospective_information_availability_verified=False,
                timing_limitation='Archive lacks forecast_issue_at and outcome_available_at. Earlier target year does not prove availability at forecast origin.',
                weights='Equal annual weights for reported summaries; pooled historical rows for intercept fitting.',
                seeds={'paired_h1':args.seed+1,'paired_h2':args.seed+2,'recalibration_year_blocks':args.seed+99},
                bootstrap_limitation='Year resampling then event-stratified within-year resampling; no cross-year firm clustering.',
                numerical_settings={'logit_clip':mt.LOGIT_EPS,'logloss_clip':mt.LOGLOSS_EPS,'bisection_tolerance':1e-12,
                                    'bisection_max_iterations':200,'bisection_bracket':[-50,50]},
                missing_value_policy='NaN written as blank in CSV / null in JSON; not zero.',
                tie_policy='Grouped thresholds for AP; average-tie-equivalent AUC; descending score then ascending firm_id for capacity.')
    write_json(out/'run_manifest.json',config)
    try:
        if input_path==BUNDLED_INPUT.resolve() and not matches_bundled:
            raise ValueError('Bundled input SHA-256 does not match the release checksum. Restore the original CSV.')
        print('1/6 Validating input archive',flush=True)
        d,validation=read_and_validate(input_path,out)
        write_json(out/'validation.json',validation)
        print('2/6 Annual scores, calibration diagnostics and capacity',flush=True)
        annual,bins,cap,policies=annual_analysis(d)
        print('3/6 Historical recalibration and actual AUC / Top-k checks',flush=True)
        recal,audit,topk,predictions=recalibration(d)
        print('4/6 Recomputing paired bootstrap intervals from CSV',flush=True)
        differences,paired=paired_bootstrap(d,annual,args.seed,args.bootstrap)
        inference=recalibration_bootstrap(recal,args.seed,args.bootstrap)
        sample=d.groupby(['horizon','year']).y.agg(records='size',events='sum',event_rate='mean').reset_index()
        schedule=pd.DataFrame([dict(evaluation_year=t,history_years='none' if t==2020 else f'2020-{t-1}',
                                    role='baseline only' if t==2020 else 'historical intercept update') for t in range(2020,2025)])
        summary=recal.groupby(['horizon','model'],sort=False).agg(
            raw_brier=('raw_brier','mean'),recalibrated_brier=('updated_brier','mean'),
            brier_improvement=('brier_improvement','mean'),log_loss_improvement=('log_loss_improvement','mean'),
            evaluation_years=('year','size'),successful_updates=('converged','sum')).reset_index()
        summary['averaging']='equal target-year weights, 2021-2024'
        annual_summary=annual.groupby(['horizon','model'],sort=False)[list(METRICS)+['log_loss']].mean().reset_index()
        annual_summary['averaging']='equal target-year weights, 2020-2024'
        print('5/6 Exporting tables and supplementary audits',flush=True)
        frames={'annual_metrics':annual,'annual_summary':annual_summary,'calibration_bins':bins,'capacity':cap,
                'threshold_policies':policies,'recalibration':recal,'paired_annual_differences':differences,
                'paired_comparisons':paired,'recalibration_inference':inference,
                'Table_1_Validation_Sample':sample,'Table_2_Recalibration_Schedule':schedule,
                'Table_3_Recalibration_Summary':summary,'Table_S1_AUC_Audit':audit,'Table_S2_TopK_Audit':topk}
        if verify_reference:
            checks=manuscript_checks(d,annual,recal,audit,topk,paired)
            frames['manuscript_checks']=checks
            config['manuscript_checks_all_passed']=bool(checks.passed.all())
        for name,frame in frames.items():frame.to_csv(out/(name+'.csv'),index=False,encoding='utf-8-sig',float_format='%.15g')
        predictions.to_csv(out/'recalibrated_predictions.csv.gz',index=False,
                           compression={'method':'gzip','mtime':0},float_format='%.17g')
        if config['manuscript_checks_all_passed'] is False:
            raise ValueError('A manuscript regression check failed. See manuscript_checks.csv.')
        print('6/6 Rendering five figures',flush=True)
        if not args.no_figures:
            import plot_figures
            plot_figures.make_figures(out,annual,cap,recal,paired,policies)
            import matplotlib
            config['matplotlib']=matplotlib.__version__
        config.update(status='complete',completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      output_hashes={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*'))
                                     if p.is_file() and p.name!='run_manifest.json'})
        write_json(out/'run_manifest.json',config)
        print(f'Completed: {out}',flush=True)
    except Exception as exc:
        config.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        write_json(out/'run_manifest.json',config)
        raise


if __name__=='__main__':
    main()
