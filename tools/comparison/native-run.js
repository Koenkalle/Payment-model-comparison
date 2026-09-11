/* Import data-bound native predictions without evaluating executable file contents. */
(function(global){
  'use strict';
  const heads=typeof module!=='undefined'?require('../../shared/runtime/prediction-heads'):global.FraudPredictionHeads;
  const recording=typeof module!=='undefined'?require('../../models/implementations/native_recording'):global.FraudNativeRecording;
  const paymentLoader=typeof module!=='undefined'?require('../../datasets/implementations/payment_json'):null;
  const object=value=>!!value&&typeof value==='object'&&!Array.isArray(value);
  const clone=value=>JSON.parse(JSON.stringify(value));
  function validAlpha(alpha){if(!Number.isFinite(alpha)||alpha<=0||alpha>=1)throw Error('Native alpha must be a finite number strictly between zero and one.');}
  const validThreshold=value=>value===null||Number.isFinite(value);
  function parseLive(run){
    if(run.schema!=='native-fraud-comparison/v2')throw Error('Expected a native-fraud-comparison/v2 response.');
    if(!object(run.model)||typeof run.model.id!=='string'||!/^[a-z][a-z0-9_]*$/.test(run.model.id)||typeof run.model.checkpoint_id!=='string'||!run.model.checkpoint_id||typeof run.model.label!=='string'||!run.model.label)throw Error('Native response needs a valid model identity and checkpoint.');
    if(!object(run.options)||!['shadow','enforce'].includes(run.options.mode)||!['unsupervised','supervised'].includes(run.options.trainingMode)||!['shared','manual','tuned','auto'].includes(run.options.decisionPolicy))throw Error('Native response needs supported execution and policy settings.');
    if(!object(run.history)||run.history.mode!==run.options.mode||run.history.timestamps!=='strictly-before')throw Error('Native response history must match the selected execution mode.');
    if(!object(run.heads)||!Object.prototype.hasOwnProperty.call(run.heads,run.head)||run.options.predictionHead!==run.head)throw Error('Native response needs the selected prediction head.');
    const supervised=run.options.trainingMode==='supervised';
    for(const [id,state]of Object.entries(run.heads)){
      if(supervised){
        if(!object(state)||state.version!==1||state.id!==id||state.kind!=='learned-fraud'||state.input!=='transaction-representation'||state.output!=='fraud-logit'||state.score_input!=='fraud-logit'||state.score_output!=='fraud-probability'||state.score_transform!=='softplus(logit)/ln(2)')throw Error('Native supervised head needs the learned-fraud output contract.');
      }else if(heads.load(state).id!==id)throw Error('Native prediction head identity differs from its registry key.');
    }
    validAlpha(run.alpha);validAlpha(run.options.alpha);
    if(run.alpha!==run.options.alpha)throw Error('Native response alpha differs from its settings.');
    if(!object(run.policy_state)||!validThreshold(run.policy_state.tau)||run.policy_state.decisionPolicy!==run.options.decisionPolicy||run.policy_state.mode!==run.options.mode||run.policy_state.predictionHead!==run.head)throw Error('Native response needs a matching initial policy state.');
    run.dataset=paymentLoader?paymentLoader.load(run.dataset):global.FraudDatasets.load('payment_json',run.dataset);
    const payments=run.dataset.events.filter(event=>event.kind==='payment');
    if(!Array.isArray(run.predictions)||run.predictions.length!==payments.length)throw Error('Native response must score every payment in chronological order.');
    let tau=run.policy_state.tau;
    run.predictions.forEach((row,index)=>{
      if(!object(row)||row.id!==payments[index].id||!Number.isFinite(row.logit)||!Number.isFinite(row.score)||row.score<0||!validThreshold(row.tauBefore)||!validThreshold(row.tauAfter)||row.tauBefore!==tau||typeof row.evaluationEligible!=='boolean'||typeof row.settled!=='boolean'||!['CONTEXT','LEARNING','BLOCK','ALLOW'].includes(row.decision))throw Error('Invalid native payment decision or chronological policy threshold.');
      const context=!row.evaluationEligible,learning=row.tauBefore===null;
      if(row.decision==='CONTEXT'&&!context||row.decision==='LEARNING'&&!learning||row.decision==='BLOCK'&&(context||learning||row.score<=row.tauBefore)||row.decision==='ALLOW'&&(context||learning||row.score>row.tauBefore))throw Error('Native payment decision differs from its score and threshold.');
      if(row.settled!==(run.options.mode==='shadow'||row.decision!=='BLOCK'))throw Error('Native settlement differs from its execution mode.');
      if(supervised){
        const evidence=row.evidence,logit=evidence?.fraud_logit;
        if(!object(evidence)||evidence.kind!=='native-fraud'||evidence.version!==1||evidence.head!==run.head||!Number.isFinite(logit)||logit!==row.logit||!Number.isFinite(evidence.fraud_probability)||evidence.fraud_probability<0||evidence.fraud_probability>1||Object.prototype.hasOwnProperty.call(evidence,'tail_probability')||Object.prototype.hasOwnProperty.call(evidence,'likelihood_probability'))throw Error('Native supervised evidence must contain fraud logit and probability, without link-likelihood fields.');
        const score=(Math.max(logit,0)+Math.log1p(Math.exp(-Math.abs(logit))))/Math.LN2,probability=heads.sigmoid(logit);
        if(Math.abs(score-row.score)>1e-10*Math.max(1,score)||Math.abs(probability-evidence.fraud_probability)>1e-12)throw Error('Native fraud probability or score differs from its fraud logit.');
      }
      tau=row.tauAfter;
    });
    const eligible=run.predictions.filter(row=>row.evaluationEligible).map(row=>row.id);
    if(!Array.isArray(run.evaluation_ids)||JSON.stringify(eligible)!==JSON.stringify(run.evaluation_ids))throw Error('Native evaluation IDs must match eligible payment decisions.');
    return run;
  }
  function parse(input){
    const run=typeof input==='string'?JSON.parse(input):clone(input);
    if(object(run)&&run.version===2)return parseLive(run);
    if(!object(run)||run.version!==1||run.schema!=='native-fraud-comparison/v1')throw Error('Expected a native-fraud-comparison/v1 bundle.');
    if(!object(run.model)||typeof run.model.id!=='string'||!/^[a-z][a-z0-9_]*$/.test(run.model.id)||typeof run.model.checkpoint_id!=='string'||!run.model.checkpoint_id||typeof run.model.label!=='string'||!run.model.label)throw Error('Native bundle needs a valid model identity and checkpoint.');
    if(!object(run.history)||run.history.mode!=='shadow'||run.history.payments!=='observed-attempts'||run.history.timestamps!=='strictly-before'||run.history.deposits!==false||run.history.reports!==false)throw Error('Native bundle must declare strictly prior observed-payment history in shadow mode.');
    run.dataset=paymentLoader?paymentLoader.load(run.dataset):global.FraudDatasets.load('payment_json',run.dataset);
    const payments=run.dataset.events.filter(event=>event.kind==='payment'),byId=new Map(payments.map(event=>[event.id,event]));
    if(!Array.isArray(run.predictions)||run.predictions.length!==payments.length||run.predictions.some((row,index)=>!object(row)||row.id!==payments[index].id||!Number.isFinite(row.logit)))throw Error('Native predictions must contain exactly one finite logit per payment, in chronological order.');
    if(!Array.isArray(run.evaluation_ids)||!run.evaluation_ids.length||new Set(run.evaluation_ids).size!==run.evaluation_ids.length||run.evaluation_ids.some(id=>typeof id!=='string'||!byId.has(id)))throw Error('Evaluation IDs must be distinct payments in the bundled dataset.');
    const wanted=new Set(run.evaluation_ids);
    if(JSON.stringify(payments.filter(event=>wanted.has(event.id)).map(event=>event.id))!==JSON.stringify(run.evaluation_ids))throw Error('Evaluation IDs must follow payment chronology.');
    if(!object(run.calibration)||!Array.isArray(run.calibration.ids)||!Number.isSafeInteger(run.calibration.count)||run.calibration.count<1||run.calibration.ids.length!==run.calibration.count||new Set(run.calibration.ids).size!==run.calibration.count||run.calibration.ids.some(id=>typeof id!=='string'||!id))throw Error('Calibration IDs and count must identify a nonempty reference sample.');
    if(!object(run.heads)||!Object.keys(run.heads).length)throw Error('Native bundle needs serialized prediction heads.');
    for(const [id,state] of Object.entries(run.heads)){
      const head=heads.load(state);
      if(head.id!==id||head.reference_count!==run.calibration.count)throw Error('Prediction head reference count or identity differs from calibration.');
    }
    if(typeof run.head!=='string'||!Object.prototype.hasOwnProperty.call(run.heads,run.head))throw Error('Selected prediction head is unavailable in this bundle.');
    validAlpha(run.alpha);heads.load(run.heads[run.head]).threshold(run.alpha);
    if(!object(run.provenance)||typeof run.provenance.same_dataset!=='boolean')throw Error('Native bundle must identify whether calibration and evaluation share a dataset.');
    if(run.provenance.same_dataset){
      if(run.calibration.ids.some(id=>!byId.has(id)))throw Error('Calibration refers to a missing payment.');
      const end=Math.max(...run.calibration.ids.map(id=>byId.get(id).t)),start=Math.min(...run.evaluation_ids.map(id=>byId.get(id).t));
      if(end>=start)throw Error('Calibration must precede all evaluation payments, including timestamp ties.');
      const logitById=new Map(run.predictions.map(row=>[row.id,row.logit]));
      if(run.heads.empirical_tail){
        const reference=run.calibration.ids.map(id=>logitById.get(id)).sort((a,b)=>a-b);
        if(JSON.stringify(reference)!==JSON.stringify(run.heads.empirical_tail.reference_logits))throw Error('Head reference logits differ from the recorded calibration payments.');
      }
    }
    return run;
  }
  // A small content identifier for replay-cache ownership, not a security checksum.
  function fingerprint(text){let hash=2166136261;for(let i=0;i<text.length;i++)hash=Math.imul(hash^text.charCodeAt(i),16777619);return (hash>>>0).toString(16).padStart(8,'0');}
  function checkpoint(input,{head=input.head,alpha=input.alpha}={}){
    const run=parse(input);
    if(run.version===2){
      if(head!==run.head||alpha!==run.alpha)throw Error('Native prediction settings changed; request a new inference run.');
      const signature=recording.signature(run.dataset),native_run={version:2,dataset_signature:signature,events:clone(run.dataset.events),predictions:Object.fromEntries(run.predictions.map(row=>[row.id,clone(row)])),head:clone(run.heads[run.head]),alpha:run.alpha,evaluation_ids:run.evaluation_ids.slice(),calibration:clone(run.calibration||{}),provenance:clone(run.provenance||{}),history:clone(run.history),options:clone(run.options),policy_state:clone(run.policy_state)};
      return {...clone(run.model),implementation_id:'native_recording',family:'native_recording',hidden:0,parameters:run.model.parameters||0,training_mode:run.options.trainingMode,
        checkpoint_id:run.model.checkpoint_id+':'+fingerprint(JSON.stringify([signature,run.options,run.heads,run.model.feature_schema||null,run.predictions])),
        architecture:clone(run.model.architecture||{description:'Native DyGFormer + TAMI inference with the selected fraud prediction head.'}),native_run};
    }
    if(!Object.prototype.hasOwnProperty.call(run.heads,head))throw Error('Selected prediction head is unavailable in this bundle.');
    const state=clone(run.heads[head]);validAlpha(alpha);heads.load(state).threshold(alpha);
    const signature=recording.signature(run.dataset),native_run={dataset_signature:signature,events:clone(run.dataset.events),predictions:Object.fromEntries(run.predictions.map(row=>[row.id,row.logit])),head:state,alpha,evaluation_ids:run.evaluation_ids.slice(),calibration:clone(run.calibration),provenance:clone(run.provenance),history:clone(run.history)};
    const content=fingerprint(JSON.stringify([signature,run.predictions,state,run.evaluation_ids]));
    return {id:run.model.id,implementation_id:'native_recording',label:run.model.label,family:'native_recording',hidden:0,parameters:run.model.parameters||0,training_mode:'unsupervised',
      checkpoint_id:run.model.checkpoint_id+':'+head+':'+alpha+':'+content,
      architecture:{description:'Native DyGFormer + TAMI observed-link scores with '+heads.labels[head].toLowerCase()+'.'},implementation:clone(run.model.implementation||{}),native_run};
  }
  global.FraudNativeRun={parse,checkpoint};
  if(typeof module!=='undefined')module.exports=global.FraudNativeRun;
})(globalThis);
