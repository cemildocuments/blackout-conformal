"""Optional independent float32 GPU cross-check of all primary test decisions."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import sys,pathlib,json,time,numpy as np,torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent/'src'))
from core import *
P=pathlib.Path(__file__).resolve().parent
if torch.backends.mps.is_available():device='mps'
elif torch.cuda.is_available():device='cuda'
else:raise RuntimeError('A supported GPU is required for this optional check.')
start=time.perf_counter();delta=0.;different=0;n=0
with torch.inference_mode():
 for fn in sorted((P/'results/raw').glob('main_*.npz')):
  z=np.load(fn);seed=z['seed'];p=['integrator','oscillator','planar'][seed[1]];law=['gaussian','correlated_t','switching'][seed[2]]
  re=np.random.default_rng(np.random.SeedSequence(seed).spawn(5)[2])
  A,sw,sv=plant(p);w,v=noises(re,2000,100,sw,sv,law)
  at=torch.tensor(A,dtype=torch.float32,device=device);wt=torch.tensor(w,dtype=torch.float32,device=device);vt=torch.tensor(v,dtype=torch.float32,device=device)
  E=torch.zeros((2000,101,17,len(A)),device=device);E[:,:,0]=-vt
  for a in range(1,17):E[:,a:,a]=E[:,a-1:-1,a-1]@at.T+wt[:,a-1:]
  S=torch.tensor(z['scale'],dtype=torch.float32,device=device)
  score=torch.amax(torch.abs(E)/S,dim=(1,2,3)).cpu().numpy()
  delta=max(delta,float(np.max(np.abs(score-z['test_score_all_schedules']))))
  q=z['bound_all_window'][0,0]/z['scale'][0,0]
  ref=np.unpackbits(z['covered_all_schedules_all_window'])[:2000].astype(bool)
  different+=int(np.sum((score<=q)!=ref));n+=2000
 result={'device_backend':device,'torch':torch.__version__,'episodes':n,'binary_decision_differences':different,'maximum_standardized_score_difference':delta,'elapsed_seconds':time.perf_counter()-start}
(P/'results/gpu_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if different:raise RuntimeError('Coverage decisions differed across numerical implementations.')
