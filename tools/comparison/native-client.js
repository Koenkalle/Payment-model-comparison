/* Python-backed models participate in the same comparison request as browser models. */
(function(global){
  'use strict';
  function modeCapabilities(model,trainingMode){
    const capabilities=model?.capabilities||{},specific=capabilities.training_mode_capabilities?.[trainingMode];
    if(specific)return specific;
    const available=!!model?.available&&!!capabilities.training_modes?.includes(trainingMode);
    return {available,error:available?null:model?.error||'No compatible '+trainingMode+' checkpoint is available.',checkpoint_id:model?.checkpoint_id,
      prediction_heads:available?(capabilities.prediction_heads||[]):[],decision_policies:available?(capabilities.decision_policies||[]):[],default_head:trainingMode==='supervised'?'fraud_linear':'empirical_tail'};
  }
  class Client{
    constructor(){this.models=[];this.error=null;this.cache=new Map();this.identities=new WeakMap();this.nextIdentity=0;this.session=global.crypto?.randomUUID?.()||String(Date.now())+'-'+Math.random().toString(36).slice(2);this.revision=0;}
    async discover(){
      if(!/^https?:$/.test(global.location?.protocol||'')||typeof global.fetch!=='function'){
        this.error='Start the local app with python serve.py to use Python models.';return [];
      }
      try{
        const response=await global.fetch('/api/models');
        if(!response.ok)throw Error('The local model service is unavailable.');
        const payload=await response.json();
        if(!Array.isArray(payload.models))throw Error('The local model service returned an invalid catalog.');
        this.models=payload.models;
        this.error=this.models.length&&!this.models.some(m=>m.available)?this.models.map(m=>m.error||m.label+' is unavailable.').join(' '):null;
        return this.models;
      }catch(error){this.error=error.message+' Start the local app with python serve.py.';return [];}
    }
    async predict(model,data,options,signal){
      if(!this.identities.has(data))this.identities.set(data,++this.nextIdentity);
      const capability=modeCapabilities(model,options.trainingMode||'unsupervised');
      const key=JSON.stringify([model.id,capability.checkpoint_id||model.checkpoint_id||null,capability.feature_contract||null,this.identities.get(data),options]);
      if(this.cache.has(key)){const run=this.cache.get(key);this.cache.delete(key);this.cache.set(key,run);return run;}
      const response=await global.fetch('/api/compare',{method:'POST',headers:{'Content-Type':'application/json','X-Comparison-Session':this.session,'X-Comparison-Revision':String(++this.revision)},
        body:JSON.stringify({model_id:model.id,dataset:data,options}),signal});
      const payload=await response.json();
      if(!response.ok)throw Error(payload.error||'The model could not score this dataset.');
      const checkpoint=global.FraudNativeRun.checkpoint(payload);
      this.cache.set(key,checkpoint);if(this.cache.size>6)this.cache.delete(this.cache.keys().next().value);
      return checkpoint;
    }
  }
  global.FraudNativeClient={Client,modeCapabilities};
  if(typeof module!=='undefined')module.exports=global.FraudNativeClient;
})(globalThis);
