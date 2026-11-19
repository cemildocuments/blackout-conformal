"""Counterfactual reset-error calibration for intermittent full-state sensing."""
import numpy as np
from scipy.linalg import expm


def plant(name):
    if name == 'integrator':
        A=np.array([[1.,.1],[0.,1.]])
        sw=np.array([.002,.012]); sv=np.array([.02,.035])
    elif name == 'oscillator':
        A=expm(.1*np.array([[0.,1.],[-1.5,-.3]]))
        sw=np.array([.003,.018]); sv=np.array([.02,.035])
    elif name == 'planar':
        A=np.array([[1.,.1,0.,0.],[0.,.98,0.,.015],[0.,0.,1.,.1],[0.,-.015,0.,.98]])
        sw=np.array([.002,.012,.002,.012]);sv=np.array([.02,.035,.02,.035])
    else: raise ValueError(name)
    return A,sw,sv


def noises(rng,n,T,sw,sv,law):
    d=len(sw)
    corr=.25*np.ones((d,d))+.75*np.eye(d)
    C=np.linalg.cholesky(corr)
    if law=='gaussian':
        z=rng.normal(size=(n,T,d))@C.T
    elif law=='correlated_t':
        # Stationary marginal variance 1, t_5 innovations, 100-step burn-in.
        zz=(rng.standard_t(5,size=(n,T+100,d))/np.sqrt(5/3))@C.T
        z0=np.zeros((n,d)); z=np.empty((n,T,d)); rho=.85
        for t in range(T+100):
            z0=rho*z0+np.sqrt(1-rho*rho)*zz[:,t]
            if t>=100:z[:,t-100]=z0
    elif law=='switching':
        # Persistent volatility bursts; stationary P(high)=.1, mean high run=8.
        z=np.empty((n,T,d));high=rng.random(n)<.1
        for t in range(T):
            scale=np.where(high,4.,.5)/np.sqrt(.1*16+.9*.25)
            z[:,t]=(rng.normal(size=(n,d))@C.T)*scale[:,None]
            r=rng.random(n);high=np.where(high,r>=1/8,r<1/72)
    else:raise ValueError(law)
    w=z*sw
    v=rng.normal(size=(n,T+1,d))*sv
    return w,v


def errors(A,w,v,L):
    """E[i,t,a]=x_t-xhat_t when the latest packet is from t-a.

    Invalid t<a entries are NaN. The recurrence follows a fixed reset
    origin and uses actual known controls, which cancel algebraically.
    """
    n,T,d=w.shape
    E=np.full((n,T+1,L+1,d),np.nan)
    E[:,:,0]=-v
    for a in range(1,L+1):
        E[:,a:,a]=E[:,a-1:-1,a-1]@A.T+w[:,a-1:]
    return E


def shape(E):
    return np.maximum(np.sqrt(np.nanmean(E*E,axis=(0,1))),1e-12)


def quantile(s,alpha):
    n=len(s); k=int(np.ceil((n+1)*(1-alpha)))
    if k>n:return np.inf
    return np.partition(np.asarray(s),k-1)[k-1]


def ages(rng,n,T,L,kind,E=None,S=None):
    age=np.zeros((n,T+1),dtype=np.int16)
    lost=np.zeros(n,dtype=bool)
    for t in range(1,T+1):
        old=age[:,t-1]
        if kind=='iid':drop=rng.random(n)<.1
        elif kind=='bursty':
            r=rng.random(n);lost=np.where(lost,r<.875,r<1/72)
            drop=lost.copy()
        elif kind=='severe':
            r=rng.random(n);lost=np.where(lost,r<.925,r<.15)
            drop=lost.copy()
        elif kind=='periodic':drop=np.full(n,t%(L+1)!=0)
        elif kind=='greedy':
            nxt=np.minimum(old+1,L)
            held=E[np.arange(n),t,nxt]
            score_hold=np.max(np.abs(held)/S[nxt],axis=1)
            score_reset=np.max(np.abs(E[:,t,0])/S[0],axis=1)
            drop=score_hold>=score_reset
        else:raise ValueError(kind)
        age[:,t]=np.where(drop&(old<L),old+1,0)
    return age


def at_ages(E,age):
    return E[np.arange(len(E))[:,None],np.arange(E.shape[1])[None,:],age]


def calibrate(A,w,v,E,S,age,alpha):
    Z=np.abs(E)/S
    all_scores=np.nanmax(Z,axis=(1,2,3))
    observed=np.max(np.abs(at_ages(E,age))/S[age],axis=(1,2))
    L=E.shape[2]-1
    q=quantile(all_scores,alpha)
    bounds={'all_window':q*S,'observed_mask':quantile(observed,alpha)*S}
    # Independent-episode union bound over age, retaining within-episode time dependence.
    by_age=np.nanmax(Z,axis=(1,3))
    bounds['age_union']=np.array([quantile(by_age[:,a],alpha/(L+1)) for a in range(L+1)])[:,None]*S
    # Primitive disturbance and measurement envelope, then deterministic propagation.
    return bounds,{'all':all_scores,'observed':observed,'age':by_age}


def add_baselines(bounds,A,w,v,E,S,train_w_scale,train_v_scale,alpha):
    L=E.shape[2]-1
    score=np.maximum(np.max(np.abs(w)/train_w_scale,axis=(1,2)),np.max(np.abs(v)/train_v_scale,axis=(1,2)))
    q=quantile(score,alpha)
    primitive=[]; P=np.eye(len(A)); acc=np.zeros(len(A))
    for a in range(L+1):
        primitive.append(q*(np.abs(P)@train_v_scale+acc))
        acc+=np.abs(P)@train_w_scale; P=A@P
    bounds['primitive']=np.array(primitive)
    Sconst=np.max(S,axis=0)
    qconst=quantile(np.nanmax(np.abs(E)/Sconst,axis=(1,2,3)),alpha)
    bounds['constant']=np.tile(qconst*Sconst,(L+1,1))
    # Gaussian union bound using training-only empirical variances. It is a comparator,
    # not a distribution-free guarantee (variance estimation is not corrected).
    from scipy.stats import norm
    m=len(A)*sum(E.shape[1]-a for a in range(L+1))
    bounds['gaussian_union']=norm.ppf(1-alpha/(2*m))*S
    return score
