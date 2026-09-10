/* Boosted trees · NumPy implementation. See matching Python module for training. */
(function(global){
'use strict';
const registry=typeof module!=='undefined'?require('../../shared/runtime/model-registry'):global.FraudAdapters;
const h=typeof module!=='undefined'?require('./_math'):global.FraudModelMath;
const {zero,matrix,dot,add,linear,sigmoid,softmax,bin,architecture}=h;
const xgb=typeof module!=='undefined'?require('./xgboost_numpy_core'):global.FraudXGBoost;const xgbFeatures=xgb.features,xgbRaw=xgb.raw;
  const xgboost={
    initialize(model,n){return matrix(n,0);},
    update(model,s){return s.memory;},
    readout(model,s){return {embedding:s.memory,slots:[],attention:[]};},
    predict(model,s,e){
      const row=xgbFeatures(model,s,e);
      const gap=s.seen[e.u]?Math.max(0,e.t-s.last[e.u]):null;
      if(s.trainingMode==='supervised'){
        const probability=sigmoid(xgbRaw(model,row)),score=-Math.log2(Math.max(1e-12,1-probability));
        return {score,parts:[score,0,0],probabilities:[[1-probability,probability],[1],[1]],buckets:[probability],gap,embedding:[[],[]],memory:[[],[]],features:row,probability,trainingMode:'supervised',previousPairCount:s.pairs[e.u][e.v]};
      }
      const rarity=Math.max(0,.35*row[0]+.4*row[1]+1.2*(1-row[16])+.7*row[12]+.7*row[13]+.6*row[17]+.8*row[18]+.6*row[19]+.8*row[20]+.8*row[25]+.8*row[26]);
      return {score:rarity,parts:[rarity,0,0],probabilities:[[1],[1],[1]],buckets:[],gap,embedding:[[],[]],memory:[[],[]],features:row,probability:null,trainingMode:'unsupervised',previousPairCount:s.pairs[e.u][e.v]};
    }
  };

const adapter=xgboost;
registry.register('xgboost',adapter);
if(typeof module!=='undefined')module.exports=adapter;
})(globalThis);
