"""Run the fixed, seeded benchmark. All numerical results are generated locally."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import argparse,json,pathlib,sys,time,platform,concurrent.futures,csv,hashlib
import numpy as np
from scipy.linalg import expm
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent/'src'))
from core import *
ROOT=pathlib.Path(__file__).resolve().parent
POLICIES=['iid','bursty','severe','periodic','greedy']


def certification(A,w,v,ee,bounds,age):
    n,T,d=w.shape;positions=np.arange(0,d,2)
    if d==2:
        if abs(A[1,0])<1e-14:B=np.array([[.005],[.1]])
        else:
            M=np.array([[0.,1.,0.],[-1.5,-.3,1.],[0.,0.,0.]])
            B=expm(.1*M)[:2,2:]
    else:B=np.array([[.005,0],[.1,0],[0,.005],[0,.1]])
    x=np.zeros((n,d));x[:,positions]=.85
    states=np.empty((n,T+1,d))
    for t in range(T+1):
        states[:,t]=x
        if t==T:break
        estimate=x-ee[:,t]
        reference=.96+.1*np.sin(2*np.pi*t/T)
        u=np.clip(4*(reference-estimate[:,positions])-2.5*estimate[:,positions+1],-2.,2.)
        x=x@A.T+u@B.T+w[:,t]
    estimated=states-ee
    safe=np.all(np.abs(states[:,:,positions])<=1,axis=2)
    out={}
    for meth,b in bounds.items():
        certified=np.all(np.abs(estimated[:,:,positions])+b[age][:,:,positions]<=1,axis=2)
        out[meth]={'false_certificate':np.mean(np.any(certified&~safe,axis=1)),
                   'certified_fraction':np.mean(certified),
                   'safe_certified_fraction':np.sum(certified&safe)/max(1,np.sum(safe)),
                   'true_safe_fraction':np.mean(safe)}
    return out


def one(job):
    start=time.perf_counter()
    group,plantname,law,rep,T,L,ntrain,ncal,ntest=job
    pi=['integrator','oscillator','planar'].index(plantname);li=['gaussian','correlated_t','switching'].index(law)
    seed=[20261119,pi,li,rep,T,L,ntrain,ncal,ntest,0 if group=='main' else 1]
    children=np.random.SeedSequence(seed).spawn(5)
    rt,rc,re,rmc,rme=[np.random.default_rng(s) for s in children]
    A,sw,sv=plant(plantname)
    wt,vt=noises(rt,ntrain,T,sw,sv,law);Et=errors(A,wt,vt,L);S=shape(Et)
    ws=np.sqrt(np.mean(wt*wt,axis=(0,1)));vs=np.sqrt(np.mean(vt*vt,axis=(0,1)))
    del wt,vt,Et
    wc,vc=noises(rc,ncal,T,sw,sv,law);Ec=errors(A,wc,vc,L)
    ac=ages(rmc,ncal,T,L,'iid')
    bounds,scores=calibrate(A,wc,vc,Ec,S,ac,.05)
    ps=add_baselines(bounds,A,wc,vc,Ec,S,ws,vs,.05)
    del Ec,wc,vc
    w,v=noises(re,ntest,T,sw,sv,law);E=errors(A,w,v,L)
    rows=[];archive={'scale':S,'cal_all':scores['all'],'cal_observed':scores['observed'],'cal_primitive':ps}
    for name,b in bounds.items():archive['bound_'+name]=b
    for policy in POLICIES+['all_schedules']:
        if policy=='all_schedules':
            ee=E; aa=None; cert={};counts=np.array([T+1-a for a in range(L+1)])
        else:
            aa=ages(rme,ntest,T,L,policy,E,S);ee=at_ages(E,aa)
            counts=np.bincount(aa.ravel(),minlength=L+1)
            cert=certification(A,w,v,ee,bounds,aa) if group=='main' else {}
            archive['age_hist_'+policy]=counts
        if policy=='all_schedules':archive['test_score_'+policy]=np.nanmax(np.abs(ee)/(S if aa is None else S[aa]),axis=tuple(range(1,ee.ndim)))
        for meth,b in bounds.items():
            ratio=np.abs(ee)/(b if aa is None else b[aa])
            trialmax=np.nanmax(ratio,axis=tuple(range(1,ee.ndim)))
            success=trialmax<=1+1e-12
            widths=np.mean(2*b/sv,axis=1)
            row=dict(group=group,plant=plantname,law=law,rep=rep,T=T,L=L,ntrain=ntrain,ncal=ncal,ntest=ntest,method=meth,policy=policy,
                     coverage=float(np.mean(success)),covered=int(np.sum(success)),width=float(np.average(widths[counts>0],weights=counts[counts>0])),
                     width_uniform_age=float(np.mean(widths)),q_all=float(bounds['all_window'][0,0]/S[0,0]),q_observed=float(bounds['observed_mask'][0,0]/S[0,0]))
            row.update(cert.get(meth,{}));rows.append(row)
            archive['covered_'+policy+'_'+meth]=np.packbits(success)
    archive['seed']=np.array(seed);archive['seconds']=np.array(time.perf_counter()-start)
    return rows,archive,job


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--workers',type=int,default=10);pa.add_argument('--mode',choices=['main','sensitivity','smoke'],default='main');pa.add_argument('--output',default='results');args=pa.parse_args()
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True);(out/'raw').mkdir(exist_ok=True)
    jobs=[]
    if args.mode=='main':
        for p in ['integrator','oscillator','planar']:
            for law in ['gaussian','correlated_t','switching']:
                for r in range(30):jobs.append(('main',p,law,r,100,16,400,999,2000))
    elif args.mode=='sensitivity':
        for law in ['gaussian','switching']:
            for L,ncal in [(4,999),(8,999),(32,999),(16,199),(16,499),(16,1999)]:
                for r in range(20):jobs.append(('sensitivity','integrator',law,r,100,L,400,ncal,2000))
    else:
        jobs=[('smoke','integrator','gaussian',0,20,4,50,99,100)]
    import threading, psutil
    peak=[0]; stop=threading.Event()
    def monitor():
        proc=psutil.Process()
        while not stop.is_set():
            total=0
            for pr in [proc]+proc.children(recursive=True):
                try:total+=pr.memory_info().rss
                except psutil.Error:pass
            peak[0]=max(peak[0],total);stop.wait(.25)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    t=time.perf_counter();allrows=[];dur=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,(rows,archive,job) in enumerate(pool.map(one,jobs),1):
            allrows.extend(rows);dur.append(float(archive['seconds']))
            g,p,l,r,T,L,ntr,nc,nte=job
            np.savez_compressed(out/'raw'/f'{g}_{p}_{l}_r{r:02d}_L{L}_n{nc}.npz',**archive)
            if i%10==0 or i==len(jobs):print(f'{args.mode}: {i}/{len(jobs)} jobs, {time.perf_counter()-t:.1f} s',flush=True)
    fields=list(dict.fromkeys(k for r in allrows for k in r))
    with (out/f'{args.mode}_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(allrows)
    stop.set();thread.join()
    import scipy,psutil
    meta={'peak_sum_rss_bytes':peak[0],'mode':args.mode,'jobs':len(jobs),'workers':args.workers,'elapsed_seconds':time.perf_counter()-t,'job_seconds_mean':np.mean(dur),'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'system':platform.system(),'machine':platform.machine(),'logical_cpus':os.cpu_count(),'physical_memory_bytes':psutil.virtual_memory().total,'configurations':jobs,'seed_scheme':'SeedSequence([20261119,plant_id,law_id,rep,T,L,ntrain,ncal,ntest,group_id]); five spawned streams'}
    (out/f'{args.mode}_environment.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({k:v for k,v in meta.items() if k!='configurations'},indent=2),flush=True)
if __name__=='__main__':main()
