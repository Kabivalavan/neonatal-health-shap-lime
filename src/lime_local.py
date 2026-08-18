import numpy as np
from sklearn.linear_model import Ridge

def explain_local(model, X, row_index=0, samples=1000):
    x0=X[row_index]; rng=np.random.default_rng(42)
    Z=x0+rng.normal(0,.35,(samples,X.shape[1])); Z[0]=x0
    p=model.predict_proba(Z)[:,1]; d=np.linalg.norm(Z-x0,axis=1); w=np.exp(-(d**2)/2)
    surrogate=Ridge(alpha=1.0).fit(Z,p,sample_weight=w)
    contribution=surrogate.coef_*(x0-X.mean(axis=0))
    return surrogate,contribution
