import sys,pathlib,itertools,unittest,numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from core import *
class ScientificChecks(unittest.TestCase):
 def test_errors_equal_explicit_dynamics(self):
  r=np.random.default_rng(7);A,sw,sv=plant('planar');n,T,L=7,11,5
  w,v=noises(r,n,T,sw,sv,'gaussian');E=errors(A,w,v,L)
  x=r.normal(size=(n,T+1,4));u=r.normal(size=(n,T,4))
  for t in range(T):x[:,t+1]=x[:,t]@A.T+u[:,t]+w[:,t]
  for s in range(T+1):
   pred=x[:,s]+v[:,s]
   for a in range(min(L,T-s)+1):
    np.testing.assert_allclose(x[:,s+a]-pred,E[:,s+a,a],rtol=1e-10,atol=1e-12)
    if s+a<T:pred=pred@A.T+u[:,s+a]
 def test_exhaustive_masks_equal_window_maximum(self):
  r=np.random.default_rng(9);A,sw,sv=plant('integrator');T,L=7,3
  w,v=noises(r,6,T,sw,sv,'switching');E=errors(A,w,v,L);S=shape(E)
  best=np.zeros(6)
  for mask in itertools.product([False,True],repeat=T):
   aa=[0]
   for drop in mask:aa.append(aa[-1]+1 if drop else 0)
   if max(aa)>L:continue
   age=np.tile(aa,(6,1));sc=np.max(np.abs(at_ages(E,age))/S[age],axis=(1,2));best=np.maximum(best,sc)
  np.testing.assert_allclose(best,np.nanmax(np.abs(E)/S,axis=(1,2,3)))
 def test_quantile_boundary(self):
  self.assertTrue(np.isinf(quantile(np.arange(18),.05)))
  self.assertEqual(quantile(np.arange(19),.05),18)
  self.assertEqual(quantile(np.arange(999),.05),949)
 def test_primitive_envelope(self):
  r=np.random.default_rng(1);A,sw,sv=plant('oscillator');w,v=noises(r,50,30,sw,sv,'correlated_t');E=errors(A,w,v,8);S=shape(E)
  b={};ps=add_baselines(b,A,w,v,E,S,sw,sv,.05)
  good=ps<=quantile(ps,.05)
  self.assertTrue(np.all(np.nanmax(np.abs(E[good])/b['primitive'],axis=(1,2,3))<=1+1e-12))
 def test_age_cap(self):
  r=np.random.default_rng(3)
  for k in ['iid','bursty','severe','periodic']:
   a=ages(r,100,100,8,k);self.assertTrue(np.all(a<=8));self.assertTrue(np.all(a[:,0]==0))
if __name__=='__main__':unittest.main()
