"""Paired equal-event comparisons and observable censoring on fresh random streams."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,pathlib,argparse,csv,json,time,concurrent.futures,platform
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from core import plant,noises,errors,shape,ages,at_ages,quantile
from extensions import primitive_shape,calibrate_shape,rollout,certificate_metrics


def one(job):
    name,law,rep=job;T,L,ntrain,ncal,ntest=100,16,400,999,2000
    seed=[20261120,['integrator','oscillator','planar'].index(name),['gaussian','correlated_t','switching'].index(law),rep]
    rt,rc,re,rm=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    A,sw,sv=plant(name)
    wt,vt=noises(rt,ntrain,T,sw,sv,law);Et=errors(A,wt,vt,L);S=shape(Et)
    ws=np.sqrt(np.mean(wt*wt,axis=(0,1)));vs=np.sqrt(np.mean(vt*vt,axis=(0,1)))
    innovation_scale=np.sqrt(np.mean((Et[:,1:,1]+vt[:,1:])**2,axis=(0,1)))
    shapes={'all_window':S,'primitive_calibrated':primitive_shape(A,ws,vs,L),
            'constant':np.tile(np.max(S,axis=0),(L+1,1)),
            'max_shape':np.nanmax(np.abs(Et),axis=(0,1)),
            'tail_shape':np.quantile(np.nanmax(np.abs(Et),axis=1),.95,axis=0)}
    del wt,vt,Et
    wc,vc=noises(rc,ncal,T,sw,sv,law);Ec=errors(A,wc,vc,L)
    bounds={};archive={'seed':np.asarray(seed),'innovation_scale':innovation_scale}
    for method,scale in shapes.items():
        bounds[method],cs=calibrate_shape(Ec,scale)
        archive['shape_'+method]=scale;archive['cal_'+method]=cs
    ac=ages(rm,ncal,T,L,'iid')
    cs=np.max(np.abs(at_ages(Ec,ac))/S[ac],axis=(1,2))
    bounds['observed_mask']=quantile(cs,.05)*S;archive['cal_observed_mask']=cs
    ps=np.maximum(np.max(np.abs(wc)/ws,axis=(1,2)),np.max(np.abs(vc)/vs,axis=(1,2)))
    bounds['primitive']=quantile(ps,.05)*shapes['primitive_calibrated'];archive['cal_primitive']=ps
    del Ec,wc,vc
    w,v=noises(re,ntest,T,sw,sv,law);E=errors(A,w,v,L)
    states,estimates,age=rollout(A,w,v,L,innovation_scale)
    actual_error=states-estimates
    identity_error=float(np.max(np.abs(actual_error-at_ages(E,age))))
    assert identity_error<1e-12,identity_error
    archive['gate_age_hist']=np.bincount(age.ravel(),minlength=L+1)
    archive['identity_error']=np.asarray(identity_error)
    rows=[]
    for method,bound in bounds.items():
        archive['bound_'+method]=bound
        for policy in ['all_schedules','censor']:
            ratios=np.abs(E)/bound if policy=='all_schedules' else np.abs(actual_error)/bound[age]
            maximum=np.nanmax(ratios,axis=tuple(range(1,ratios.ndim)))
            success=maximum<=1+1e-12
            archive['covered_'+policy+'_'+method]=np.packbits(success)
            row={'plant':name,'law':law,'rep':rep,'method':method,'policy':policy,'ntest':ntest,
                 'coverage':float(np.mean(success)),'covered':int(np.sum(success)),
                 'width_uniform_age':float(np.mean(2*bound/sv))}
            if policy=='censor':
                row.update(certificate_metrics(states,estimates,age,bound))
                row['drop_fraction']=float(np.mean(age[:,1:]>0))
                row['mean_age']=float(np.mean(age))
            rows.append(row)
    return rows,archive,job


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);p.add_argument('--output',default='results');args=p.parse_args()
    out=ROOT/args.output;(out/'extensions_raw').mkdir(parents=True,exist_ok=True)
    jobs=[(p,l,r) for p in ['integrator','oscillator','planar'] for l in ['gaussian','correlated_t','switching'] for r in range(30)]
    start=time.perf_counter();rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,(rr,archive,(p,l,r)) in enumerate(pool.map(one,jobs),1):
            rows.extend(rr);np.savez_compressed(out/'extensions_raw'/f'{p}_{l}_r{r:02d}.npz',**archive)
            if i%30==0:print(f'extensions {i}/{len(jobs)}; {time.perf_counter()-start:.1f} s',flush=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (out/'extensions_metrics.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(rows)
    meta={'jobs':len(jobs),'test_episodes':540000,'workers':args.workers,'elapsed_seconds':time.perf_counter()-start,'python':platform.python_version(),'numpy':np.__version__,'seed_prefix':20261120,'gate_threshold':2.5,'training_temporal_quantile':.95,'maximum_identity_error':max(float(np.load(f)['identity_error']) for f in sorted((out/'extensions_raw').glob('*.npz')))}
    (out/'extensions_environment.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2),flush=True)
if __name__=='__main__':main()
