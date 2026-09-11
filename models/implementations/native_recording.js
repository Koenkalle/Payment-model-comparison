/* Recorded native inference: exact observed-request replay with a swappable head. */
(function(global){
  'use strict';
  const registry=typeof module!=='undefined'?require('../../shared/runtime/model-registry'):global.FraudAdapters;
  const heads=typeof module!=='undefined'?require('../../shared/runtime/prediction-heads'):global.FraudPredictionHeads;
  const loadedHeads=new WeakMap();
  const own=(value,key)=>Object.prototype.hasOwnProperty.call(value,key);
  const eventIdentity=event=>[event.id,event.kind,event.t,event.u,event.v,event.amount,own(event,'settled')?event.settled:null,own(event,'reference')?event.reference:null];
  function signature(data){
    const units=data.units||{time:'minutes',currency:'EUR'};
    return JSON.stringify({units:[units.time,units.currency],accounts:data.accounts.map(account=>[account.id,String(account.external_id??account.id)]),events:data.events.map(eventIdentity)});
  }
  function optionIdentity(options={}){
    const policy=options.decisionPolicy||'shared';
    return JSON.stringify([options.mode||'enforce',options.trainingMode||'unsupervised',options.predictionHead||(options.trainingMode==='supervised'?'fraud_linear':'empirical_tail'),policy,
      ...(policy==='shared'?[options.alpha??.02,options.eta??.025,Math.max(1,Math.floor(options.warmup??128))]:policy==='manual'?[options.manualTau]:policy==='tuned'?[options.falseBlockCost??1,options.missedFraudCost??20]:[options.objective||'f1'])]);
  }
  function policyOptions(model,options={}){
    const run=model.native_run;
    if(!run)throw Error('Native recording requires an exported comparison bundle.');
    if(run.version===2){
      if(optionIdentity(options)!==optionIdentity(run.options))throw Error('Native prediction settings changed; compute this dataset again with the current settings.');
      return {...options,...run.options};
    }
    if((options.mode||'enforce')!=='shadow')throw Error('Native recorded predictions require shadow mode with observed history.');
    if((options.trainingMode||model.training_mode||'unsupervised')!=='unsupervised')throw Error('Native recorded predictions do not support a supervised-mode switch.');
    const head=loadedHeads.get(model)||heads.load(run.head);
    return {...options,mode:'shadow',trainingMode:'unsupervised',decisionPolicy:'manual',manualTau:head.threshold(run.alpha),alpha:run.alpha,predictionHead:head.id};
  }
  function validateDataset(model,data,options={}){
    const run=model.native_run;
    policyOptions(model,options);
    if(signature(data)!==run.dataset_signature)throw Error('Dataset differs from the native recording; export predictions for this exact event history.');
    const payments=data.events.filter(event=>event.kind==='payment');
    if(Object.keys(run.predictions).length!==payments.length||payments.some(event=>!own(run.predictions,event.id)||!Number.isFinite(run.version===2?run.predictions[event.id].logit:run.predictions[event.id])))throw Error('Native recording must contain one finite logit per payment.');
    const paymentIds=new Set(payments.map(event=>event.id));
    if(!Array.isArray(run.evaluation_ids)||(run.version!==2&&!run.evaluation_ids.length)||new Set(run.evaluation_ids).size!==run.evaluation_ids.length||run.evaluation_ids.some(id=>!paymentIds.has(id)))throw Error('Invalid native evaluation payment IDs.');
    if(run.version!==2||run.options.trainingMode!=='supervised'){
      const head=heads.load(run.head);head.threshold(run.alpha);loadedHeads.set(model,head);
    }
    return true;
  }
  function current(model,state,event){
    const expected=model.native_run.events[state.events.length];
    if(!expected||JSON.stringify(eventIdentity(event))!==JSON.stringify(eventIdentity(expected)))throw Error('Event or replay position differs from the native recording.');
  }
  const adapter={
    validateDataset,policyOptions,
    initializePolicy(model,trainingMode,options){
      if(model.native_run.version===2)return JSON.parse(JSON.stringify(model.native_run.policy_state));
      const policy=typeof module!=='undefined'?require('../../shared/runtime/policy'):global.FraudPolicy;
      return policy.initialize(model,trainingMode,options);
    },
    initialize(model,n){return Array.from({length:n},()=>[]);},
    update(model,state,event,settled){
      current(model,state,event);
      if(model.native_run.version===2&&event.kind==='payment'&&settled!==model.native_run.predictions[event.id].settled)throw Error('Native settlement differs from the computed model history.');
      return state.memory;
    },
    readout(model,state){return [state.memory];},
    predict(model,state,event){
      current(model,state,event);
      if(model.native_run.version===2){
        const run=model.native_run,row=run.predictions[event.id];
        if(state.mode!==run.options.mode||state.trainingMode!==run.options.trainingMode||state.predictionHead!==run.options.predictionHead)throw Error('Native recording differs from the active execution settings.');
        if(!row)throw Error('No native prediction for this payment.');
        return {score:row.score,parts:[row.score],memory:[[],[]],embedding:[[],[]],evidence:run.options.trainingMode==='supervised'?{...row.evidence}:{...row.evidence,logit:row.logit},evaluationEligible:row.evaluationEligible,
          decisionThreshold:row.tauBefore,nextDecisionThreshold:row.tauAfter,policyDecision:row.decision,policySettlement:row.settled};
      }
      if(state.mode!=='shadow'||state.trainingMode!=='unsupervised')throw Error('Native recording supports only unsupervised scoring in shadow mode.');
      const run=model.native_run,head=loadedHeads.get(model)||heads.load(run.head),logit=run.predictions[event.id];
      if(!own(run.predictions,event.id)||!Number.isFinite(logit))throw Error('No native prediction for this payment.');
      const prediction=head.score(logit);
      return {...prediction,parts:[prediction.score],memory:[[],[]],embedding:[[],[]],
        evidence:{kind:'native-likelihood',logit,likelihood_probability:heads.sigmoid(logit),tail_probability:prediction.tail_probability,head:head.id,reference_count:head.reference_count,alpha:run.alpha},
        evaluationEligible:run.evaluation_ids.includes(event.id)};
    }
  };
  registry.register('native_recording',adapter);
  global.FraudNativeRecording={signature,eventIdentity,optionIdentity,validateDataset};
  if(typeof module!=='undefined')module.exports=global.FraudNativeRecording;
})(globalThis);
