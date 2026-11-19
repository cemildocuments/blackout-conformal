"""Negative control: increase both noise amplitudes after calibration."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import sys,pathlib,csv,numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent/'src'))
from core import *
P=pathlib.Path(__file__).resolve().parent
rows=[]
for law in ['gaussian','correlated_t','switching']:
 for rep in range(30):
  z=np.load(P/'results/raw'/f'main_integrator_{law}_r{rep:02d}_L16_n999.npz')
  A,sw,sv=plant('integrator');r=np.random.default_rng(np.random.SeedSequence([80713,['gaussian','correlated_t','switching'].index(law),rep]))
  w,v=noises(r,2000,100,sw,sv,law);E=errors(A,w,v,16)
  scores=np.nanmax(np.abs(E)/z['bound_all_window'],axis=(1,2,3))
  for factor in [1.,1.25,1.5,2.]:
   rows.append(dict(law=law,rep=rep,factor=factor,coverage=float(np.mean(scores*factor<=1)),ntest=2000))
with (P/'results/shift_metrics.csv').open('w',newline='') as f:
 wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
print('Noise-law shift check complete.',flush=True)
