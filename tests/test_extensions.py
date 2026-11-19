import unittest,sys,pathlib,numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from core import plant,noises,errors,at_ages,shape
from extensions import rollout,calibrate_shape,primitive_shape

class ExtensionChecks(unittest.TestCase):
    def test_observable_gate_matches_window_error_and_its_information_rule(self):
        rng=np.random.default_rng(152);A,sw,sv=plant('planar');w,v=noises(rng,13,25,sw,sv,'switching')
        scale=np.array([.03,.05,.03,.05]);x,xh,age=rollout(A,w,v,5,scale)
        E=errors(A,w,v,5)
        np.testing.assert_allclose(x-xh,at_ages(E,age),rtol=1e-11,atol=1e-12)
        # Independently reconstruct predictions from actual applied controls.
        from extensions import input_matrix
        B=input_matrix(A);pos=np.array([0,2])
        for tt in range(1,26):
            prev=xh[:,tt-1];ref=.96+.1*np.sin(2*np.pi*(tt-1)/25)
            u=np.clip(4*(ref-prev[:,pos])-2.5*prev[:,pos+1],-2,2)
            pred=prev@A.T+u@B.T
            innovation=x[:,tt]+v[:,tt]-pred
            reject=(np.max(np.abs(innovation)/scale,axis=1)>2.5)&(age[:,tt-1]<5)
            np.testing.assert_array_equal(age[:,tt]>0,reject)
        self.assertLessEqual(age.max(),5)
    def test_global_shape_scaling_cancels(self):
        rng=np.random.default_rng(921);A,sw,sv=plant('integrator');w,v=noises(rng,99,20,sw,sv,'gaussian')
        E=errors(A,w,v,5);S=shape(E)
        b,_=calibrate_shape(E,S);g,_=calibrate_shape(E,4.817*S)
        np.testing.assert_allclose(b,g,rtol=1e-14,atol=1e-14)
    def test_propagated_shape_is_primitive_triangle_bound(self):
        A,sw,sv=plant('oscillator');h=primitive_shape(A,sw,sv,7)
        for a in range(8):
            expected=np.abs(np.linalg.matrix_power(A,a))@sv
            for j in range(a):expected+=np.abs(np.linalg.matrix_power(A,j))@sw
            np.testing.assert_allclose(h[a],expected,rtol=1e-14)
    def test_mismatch_discrepancy_equals_accumulated_model_error(self):
        rng=np.random.default_rng(37);A,sw,sv=plant('integrator');true=A.copy();true[0,0]+=.01
        w,v=noises(rng,11,22,sw,sv,'gaussian');x,xh,age=rollout(A,w,v,5,np.ones(2),policy='periodic',A_true=true)
        effective=w+x[:,:-1]@(true-A).T
        E=errors(A,effective,v,5)
        np.testing.assert_allclose(x-xh,at_ages(E,age),rtol=1e-11,atol=1e-12)
if __name__=='__main__':unittest.main()
