/* Threshold selection is separate from model training and request scoring. */
(function(global){
  'use strict';
  function costs(options={}){
    const falseBlock=options.falseBlockCost??1,missedFraud=options.missedFraudCost??20;
    if(!Number.isFinite(falseBlock)||falseBlock<=0||!Number.isFinite(missedFraud)||missedFraud<=0)throw Error('Error costs must be finite and positive.');
    return {falseBlock,missedFraud};
  }
  const objectives={f1:'F1',f2:'F2',balanced_accuracy:'Balanced accuracy'};
  function classificationMetrics(tp,fp,fn,tn){
    return {precision:tp+fp?tp/(tp+fp):null,recall:tp+fn?tp/(tp+fn):null,
      f1:2*tp+fp+fn?2*tp/(2*tp+fp+fn):0,f2:5*tp+fp+4*fn?5*tp/(5*tp+fp+4*fn):0,
      balanced_accuracy:tp+fn&&tn+fp?(tp/(tp+fn)+tn/(tn+fp))/2:null};
  }
  function requireValidation(validation){if(!validation?.candidates?.length)throw Error('No historical threshold fit for this model and training mode. Rebuild the demo after training.');}
  function fitDetails(validation,best){
    return {...best,metrics:classificationMetrics(validation.positives-best.fn,best.fp,best.fn,validation.negatives-best.fp),impliedAlpha:best.blocks/validation.requests,requests:validation.requests,positives:validation.positives,negatives:validation.negatives,unknown:validation.unknown};
  }
  function frontier(rows){
    if(!rows.length)throw Error('Threshold tuning needs historical validation requests.');
    if(rows.some(r=>!Number.isFinite(r.score)||r.score<0||![-1,0,1].includes(r.label)))throw Error('Invalid validation score or outcome.');
    const sorted=rows.slice().sort((a,b)=>b.score-a.score),positives=rows.filter(r=>r.label===1).length,negatives=rows.filter(r=>r.label===0).length;
    if(!positives||!negatives)throw Error('Threshold tuning needs both confirmed fraud and legitimate validation outcomes.');
    const candidates=[];let fp=0,fn=positives,blocks=0;
    // Strict score > tau: tied scores are always decided together. Retain
    // only nondominated choices for strictly positive error costs.
    function add(tau){
      const previous=candidates.at(-1);
      if(previous&&previous[2]===fn)return;
      if(previous&&previous[1]===fp)candidates.pop();
      candidates.push([tau,fp,fn,blocks]);
    }
    add(sorted[0].score);
    for(let i=0;i<sorted.length;){
      const score=sorted[i].score;
      do{const r=sorted[i++];blocks++;if(r.label===0)fp++;if(r.label===1)fn--;}while(i<sorted.length&&sorted[i].score===score);
      add(i<sorted.length?sorted[i].score:-Number.EPSILON);
    }
    return {requests:rows.length,positives,negatives,unknown:rows.length-positives-negatives,candidates};
  }
  function select(validation,options={}){
    requireValidation(validation);
    const cost=costs(options);let best=null;
    for(const [tau,fp,fn,blocks]of validation.candidates){
      const loss=cost.falseBlock*fp+cost.missedFraud*fn;
      if(!best||loss<best.loss||(loss===best.loss&&blocks<best.blocks))best={tau,fp,fn,blocks,loss};
    }
    return {...fitDetails(validation,best),costs:cost};
  }
  function autoTune(validation,objective='f1'){
    requireValidation(validation);if(!Object.prototype.hasOwnProperty.call(objectives,objective))throw Error('Unknown automatic tuning objective.');
    let best=null;
    for(const [tau,fp,fn,blocks]of validation.candidates){
      const value=classificationMetrics(validation.positives-fn,fp,fn,validation.negatives-fp)[objective];
      if(!best||value>best.value||(value===best.value&&blocks<best.blocks))best={tau,fp,fn,blocks,value};
    }
    return {...fitDetails(validation,best),objective,candidatesTested:validation.candidates.length};
  }
  function initialize(model,trainingMode,options={}){
    const decisionPolicy=options.decisionPolicy||'shared',alpha=options.alpha??.02,errorCosts=costs(options);
    if(!['shared','tuned','auto','manual'].includes(decisionPolicy))throw Error('Unknown decision policy.');
    if(!Number.isFinite(alpha)||alpha<=0||alpha>=1)throw Error('The comparison budget must be between zero and one.');
    if(decisionPolicy==='manual'&&!Number.isFinite(options.manualTau))throw Error('Manual τ must be finite.');
    const policyFit=decisionPolicy==='tuned'?select(model.policy_validation?.[trainingMode],options):decisionPolicy==='auto'?autoTune(model.policy_validation?.[trainingMode],options.objective||'f1'):null;
    return {decisionPolicy,alpha,errorCosts,policyFit,tau:decisionPolicy==='manual'?options.manualTau:policyFit?.tau??null};
  }
  global.FraudPolicy={costs,frontier,select,autoTune,objectives,classificationMetrics,initialize};
  if(typeof module!=='undefined')module.exports=global.FraudPolicy;
})(typeof globalThis!=='undefined'?globalThis:window);
