"""Portable coefficients for the externally trained standardized logistic ranker.

No trained coefficients are distributed in this repository.
"""
import numpy as np
from scipy.special import expit

class LogisticRanker:
    def __init__(self, mean, scale, coefficient, intercept,coefficient_dtype='float64'):
        self.mean=np.asarray(mean,float);self.scale=np.asarray(scale,float)
        self.coefficient=np.asarray(coefficient,dtype=coefficient_dtype);self.intercept=float(intercept)
        if any(x.shape!=(11,) for x in [self.mean,self.scale,self.coefficient]) or np.any(self.scale<=0):
            raise ValueError('Ranker requires eleven features and positive scales')
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        scaler=StandardScaler();scaler.mean_=self.mean;scaler.scale_=self.scale
        scaler.var_=self.scale**2;scaler.n_features_in_=11;scaler.n_samples_seen_=1
        model=LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=42)
        model.coef_=self.coefficient[None,:]
        model.intercept_=np.asarray([self.intercept],dtype=self.coefficient.dtype)
        model.classes_=np.asarray([0,1]);model.n_features_in_=11
        self.model=make_pipeline(scaler,model)
    def predict_proba(self,x):
        x=np.asarray(x,dtype=np.float32)
        # Delegate dtype-specific arithmetic to the same sklearn implementation.
        return self.model.predict_proba(x)
    @classmethod
    def from_dict(cls,d):
        return cls(d['mean'],d['scale'],d['coefficient'],d['intercept'],d.get('coefficient_dtype','float64'))

def fit_ranker(features,labels):
    """Fit using training structures only; use separate held-out structure folds."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    model=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=42))
    return model.fit(np.asarray(features,np.float32),np.asarray(labels))

def export_ranker(model):
    scaler,logistic=model.steps[0][1],model.steps[1][1]
    return {'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
            'coefficient':logistic.coef_[0].tolist(),'intercept':float(logistic.intercept_[0]),'coefficient_dtype':str(logistic.coef_.dtype)}
