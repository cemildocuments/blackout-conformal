"""Training-only shape alternatives and an observable innovation-censoring rule."""
import numpy as np
from scipy.linalg import expm
from core import quantile


def primitive_shape(A, sw, sv, L):
    out=[];power=np.eye(len(A));acc=np.zeros(len(A))
    for age in range(L+1):
        out.append(np.abs(power)@sv+acc)
        acc += np.abs(power)@sw
        power=A@power
    return np.asarray(out)


def calibrate_shape(E, scale, alpha=.05):
    scores=np.nanmax(np.abs(E)/scale,axis=(1,2,3))
    return quantile(scores,alpha)*scale,scores


def input_matrix(A):
    if len(A)==4:
        return np.array([[.005,0],[.1,0],[0,.005],[0,.1]])
    if abs(A[1,0])<1e-14:
        return np.array([[.005],[.1]])
    M=np.array([[0.,1.,0.],[-1.5,-.3,1.],[0.,0.,0.]])
    return expm(.1*M)[:2,2:]


def rollout(A,w,v,L,innovation_scale,threshold=2.5,policy='censor',A_true=None):
    """Simulate a feedback trajectory and the actual observer.

    Censoring decisions use only y - predicted_state and the prior age.
    The hidden state is used by the simulator and evaluation, never the rule.
    An age-L gap always ends in reception, overriding censoring.
    """
    if A_true is None:A_true=A
    n,T,d=w.shape;B=input_matrix(A);pos=np.arange(0,d,2)
    x=np.zeros((n,d));x[:,pos]=.85;xhat=x+v[:,0]
    states=np.empty((n,T+1,d));estimates=np.empty_like(states)
    states[:,0]=x;estimates[:,0]=xhat;age=np.zeros((n,T+1),dtype=np.int16)
    for t in range(1,T+1):
        reference=.96+.1*np.sin(2*np.pi*(t-1)/T)
        u=np.clip(4*(reference-xhat[:,pos])-2.5*xhat[:,pos+1],-2,2)
        x=x@A_true.T+u@B.T+w[:,t-1]
        predicted=xhat@A.T+u@B.T
        measurement=x+v[:,t]
        if policy=='censor':
            innovation=measurement-predicted
            drop=np.max(np.abs(innovation)/innovation_scale,axis=1)>threshold
        elif policy=='periodic':drop=np.full(n,t%(L+1)!=0)
        else:raise ValueError(policy)
        drop &= age[:,t-1]<L
        age[:,t]=np.where(drop,age[:,t-1]+1,0)
        xhat=np.where(drop[:,None],predicted,measurement)
        states[:,t]=x;estimates[:,t]=xhat
    return states,estimates,age


def certificate_metrics(states,estimates,age,bound):
    pos=np.arange(0,states.shape[-1],2)
    safe=np.all(np.abs(states[:,:,pos])<=1,axis=2)
    certified=np.all(np.abs(estimates[:,:,pos])+bound[age][:,:,pos]<=1,axis=2)
    false=np.any(certified&~safe,axis=1)
    return {'false_certificate':float(np.mean(false)),
            'certified_fraction':float(np.mean(certified)),
            'safe_certified_fraction':float(np.sum(certified&safe)/max(1,np.sum(safe))),
            'true_safe_fraction':float(np.mean(safe))}
