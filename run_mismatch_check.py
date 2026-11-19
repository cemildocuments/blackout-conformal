"""Dynamics-mismatch negative control on fresh, paired noise episodes."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import pathlib,sys,csv,json,argparse,concurrent.futures,numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from core import plant,noises
from extensions import rollout,certificate_metrics

def one(job):
    law,rep=job;T,L=100,16;A,sw,sv=plant('integrator')
    rng=np.random.default_rng(np.random.SeedSequence([20261121,['gaussian','correlated_t','switching'].index(law),rep]))
    w,v=noises(rng,2000,T,sw,sv,law)
    record=np.load(ROOT/'results/raw'/f'main_integrator_{law}_r{rep:02d}_L16_n999.npz')
    bound=record['bound_all_window'];rows=[]
    for delta in [0,.005,.01,.02]:
        true=A.copy();true[0,0]+=delta
        x,xhat,age=rollout(A,w,v,L,np.ones(2),policy='periodic',A_true=true)
        good=np.max(np.abs(x-xhat)/bound[age],axis=(1,2))<=1+1e-12
        row={'law':law,'rep':rep,'delta':delta,'ntest':2000,'coverage':float(good.mean()),'covered':int(good.sum())}
        row.update(certificate_metrics(x,xhat,age,bound));rows.append(row)
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);args=p.parse_args()
    jobs=[(law,r) for law in ['gaussian','correlated_t','switching'] for r in range(30)];rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(one,jobs):rows.extend(r)
    with (ROOT/'results/mismatch_metrics.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    print('Completed dynamics-mismatch negative control: 90 paired noise batches, 180000 distinct noise episodes.',flush=True)
if __name__=='__main__':main()
