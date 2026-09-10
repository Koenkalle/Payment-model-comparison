"""Scikit-learn logistic regression; preprocessing is fitted on training rows only."""
import json
import numpy as np
from framework.contracts import Predictions

class Model:
    def fit(self,x,y,feature_names,parameters):
        try:
            import sklearn
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError as error:raise RuntimeError('Install scikit-learn to train logistic_regression.') from error
        self.library_version=sklearn.__version__
        scaler=StandardScaler().fit(x)
        estimator=LogisticRegression(**{'max_iter':500,'random_state':42,**parameters}).fit(scaler.transform(x),y)
        self.state={'version':1,'feature_names':list(feature_names),'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'weights':estimator.coef_[0].tolist(),'bias':float(estimator.intercept_[0])}

    def predict(self,x,feature_names,explain=False):
        if list(feature_names)!=self.state['feature_names']:raise ValueError('Feature names or order differ from the fitted model.')
        if explain:raise ValueError('This plugin does not declare feature explanations.')
        margins=((x-np.asarray(self.state['mean']))/np.asarray(self.state['scale']))@np.asarray(self.state['weights'])+self.state['bias']
        return Predictions(1/(1+np.exp(-np.clip(margins,-50,50))),margins)

    def save(self,path):path.write_text(json.dumps(self.state,allow_nan=False))
    def load(self,path,feature_names):
        state=json.loads(path.read_text())
        if state.get('version')!=1 or state.get('feature_names')!=list(feature_names):raise ValueError('Incompatible logistic model schema.')
        count=len(feature_names)
        for field in ('mean','scale','weights'):
            values=np.asarray(state[field],dtype=float)
            if values.shape!=(count,) or not np.isfinite(values).all():raise ValueError('Invalid logistic model parameters.')
        if np.any(np.asarray(state['scale'])<=0) or not np.isfinite(state['bias']):raise ValueError('Invalid logistic scaling/bias.')
        self.state=state

def create():return Model()
