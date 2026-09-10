"""Official XGBoost binary classifier. No dependency on payment schemas or generators."""
import numpy as np
from framework.contracts import Predictions

class Model:
    def __init__(self):
        try:import xgboost
        except ImportError as error:raise RuntimeError('Install the native model dependencies: pip install -r requirements-models.txt') from error
        self.library=xgboost;self.library_version=xgboost.__version__;self.booster=None

    def fit(self,x,y,feature_names,parameters):
        params={'objective':'binary:logistic','eval_metric':'logloss','tree_method':'hist','max_depth':4,'eta':.08,'seed':42,'nthread':2,**parameters}
        rounds=params.pop('num_boost_round',100)
        if not isinstance(rounds,int) or rounds<1:raise ValueError('num_boost_round must be a positive integer.')
        if params['objective']!='binary:logistic' or params.get('booster','gbtree')!='gbtree':raise ValueError('This plugin supports binary logistic tree boosting.')
        train=self.library.DMatrix(x,label=y,feature_names=list(feature_names))
        self.booster=self.library.train(params,train,num_boost_round=rounds)

    def predict(self,x,feature_names,explain=False):
        matrix=self.library.DMatrix(x,feature_names=list(feature_names))
        probabilities=self.booster.predict(matrix,validate_features=True)
        margins=self.booster.predict(matrix,output_margin=True,validate_features=True)
        contributions=self.booster.predict(matrix,pred_contribs=True,approx_contribs=False,validate_features=True) if explain else None
        if contributions is not None and not np.allclose(contributions.sum(axis=1),margins,rtol=1e-5,atol=1e-5):raise ValueError('Native feature contributions do not sum to the margin.')
        return Predictions(probabilities,margins,contributions)

    def save(self,path):self.booster.save_model(path)
    def load(self,path,feature_names):
        self.booster=self.library.Booster();self.booster.load_model(path)
        if self.booster.feature_names!=list(feature_names):raise ValueError('Native checkpoint feature order differs from artifact metadata.')

def create():return Model()
