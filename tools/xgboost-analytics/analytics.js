/* Completed-scenario analysis. Predictions use only the history before each payment. */
(function(global){
  'use strict';
  const core=typeof module!=='undefined'?require('../../shared/runtime/model'):global.FraudCore;
  const xgb=typeof module!=='undefined'?require('../../shared/runtime/xgboost'):global.FraudXGBoost;
  const evaluation=typeof module!=='undefined'?require('../../shared/runtime/metrics'):global.FraudMetrics;
  const policy=typeof module!=='undefined'?require('../../shared/runtime/policy'):global.FraudPolicy;
  const now=()=>global.performance?.now?global.performance.now():Date.now();
  const yieldTask=()=>new Promise(resolve=>setTimeout(resolve,0));
  const histories=new Map();
  function freeze(value){
    if(value&&typeof value==='object'&&!Object.isFrozen(value)){
      for(const item of Object.values(value))freeze(item);
      Object.freeze(value);
    }
    return value;
  }
  const copy=value=>JSON.parse(JSON.stringify(value));
  function configuration(options){
    if(options.trainingMode!==undefined&&options.trainingMode!=='supervised')throw Error('XGBoost analytics requires supervised tree scoring.');
    if(options.mode!==undefined&&options.mode!=='shadow')throw Error('XGBoost analytics uses observed history.');
    const result={decisionPolicy:'shared',alpha:.02,warmup:128,eta:.025,manualTau:1,falseBlockCost:1,missedFraudCost:20,objective:'f1',...options,trainingMode:'supervised',mode:'shadow'};
    if(!Number.isInteger(result.warmup)||result.warmup<1)throw Error('Warm-up must be a positive whole number.');
    if(!Number.isFinite(result.eta)||result.eta<0)throw Error('The cutoff learning rate must be finite and nonnegative.');
    return result;
  }
  function truthLabel(truth,id){
    if(!Object.prototype.hasOwnProperty.call(truth,id))return null;
    const value=truth[id];
    return value===true||value===1?1:value===false||value===0?0:null;
  }
  function metrics(records,config){
    const truth={};for(const r of records)if(r.label!==null)truth[r.event.id]=r.label===1;
    const result=evaluation.metrics(records,truth,config.alpha,policy.costs(config));
    return {...result,warmup:records.length-result.eligible,unknown:records.filter(r=>r.label===null).length,
      assessedUnknown:result.eligible-result.labeled,f1:result.labeled?result.f1:null,f2:result.labeled?result.f2:null,
      errorCost:result.labeled?result.errorCost:null};
  }
  function cleanEvent(raw){
    const event={id:raw.id,t:raw.t,u:raw.u,v:raw.v,amount:raw.amount,kind:raw.kind};
    if(raw.reference!==undefined)event.reference=raw.reference;
    if(raw.settled!==undefined)event.settled=raw.settled;
    return event;
  }
  async function run(model,data,reference,options={},control={}){
    const config=configuration(options),started=now();
    const cancelled=control.cancelled||(()=>false),onProgress=control.onProgress||(()=>{});
    const yieldControl=control.yieldControl||yieldTask,budgetMs=Math.max(1,control.budgetMs??8);
    if(model.family!=='xgboost')throw Error('Select an XGBoost tree checkpoint.');
    if(!Array.isArray(data.accounts)||!Array.isArray(data.events))throw Error('A scenario needs accounts and chronological events.');
    // Validate both the requested policy and explanation reference even on a cache hit.
    policy.initialize(model,'supervised',config);
    let cached,rows=[],finalTau,policyFit,inferenceCalls=0,explanationCalls=0,yields=0,maxSliceMs=0;
    let sliceStart=now();
    const pause=async(done)=>{
      maxSliceMs=Math.max(maxSliceMs,now()-sliceStart);
      onProgress({done,total:data.events.length,phase:'scoring',percent:data.events.length?100*done/data.events.length:100});
      yields++;await yieldControl();sliceStart=now();
    };
    await pause(0);if(cancelled())return null;
    const explainer=xgb.createExplainer(model,reference);
    const key=JSON.stringify([model,reference,data]);cached=histories.get(key);
    if(now()-sliceStart>=budgetMs)await pause(0);
    if(cancelled())return null;
    if(cached){
      histories.delete(key);histories.set(key,cached);
      const decisions=core.evaluatePolicy(model,cached,config);
      rows=cached.map((r,i)=>freeze({...r,...decisions.records[i]}));
      finalTau=decisions.tau;policyFit=decisions.policyFit;
    }else{
      const state=core.initial(model,data.accounts.length,config);
      for(let i=0;i<data.events.length;i++){
        if(cancelled())return null;
        const event=cleanEvent(data.events[i]);
        const prediction=core.score(model,state,event);
        let details=null;
        if(prediction){
          inferenceCalls++;
          const described=xgb.describe(model,state,event);
          const explanation=explainer.explain(described.values);explanationCalls++;
          if(!Number.isFinite(explanation.rawMargin))throw Error('Explanation returned a non-finite margin.');
          const sum=explanation.baseline+explanation.contributions.reduce((a,b)=>a+b,0);
          if(Math.abs(sum-explanation.rawMargin)>1e-8*(1+Math.abs(explanation.rawMargin)))throw Error('Feature contributions do not reproduce this prediction.');
          details={features:described.values.slice(),readableValues:described.readableValues.slice(),rawMargin:explanation.rawMargin,
            probability:prediction.probability,explanation:copy(explanation),explanationReference:reference.reference_id};
        }
        // The shared core makes the decision, settles in shadow mode, then updates tau.
        const decision=core.step(model,state,event,prediction);
        if(details)rows.push(freeze({...decision,...details,label:truthLabel(data.truth||{},event.id)}));
        if(now()-sliceStart>=budgetMs)await pause(i+1);
      }
      finalTau=state.tau;policyFit=state.policyFit;
      if(cancelled())return null;
      cached=freeze(rows.slice());histories.set(key,cached);
      if(histories.size>2)histories.delete(histories.keys().next().value);
    }
    if(cancelled())return null;
    maxSliceMs=Math.max(maxSliceMs,now()-sliceStart);
    const report={version:1,model:{id:model.id,label:model.label,checkpointId:model.checkpoint_id,checkpointHash:reference.checkpoint_sha256},
      dataset:{schema:data.schema||'payment-events/v1',provenance:copy(data.provenance||{origin:'synthetic'})},
      featureSchemaVersion:reference.feature_schema_version,reference:{id:reference.reference_id,...copy(reference.reference)},
      configuration:{...copy(config),scenario:data.name,size:data.size,seed:data.seed},accounts:copy(data.accounts),
      finalPosition:data.events.length,finalTime:data.events.length?data.events[data.events.length-1].t:null,finalTau,
      policyFit:policyFit?copy(policyFit):null,records:rows,metrics:metrics(rows,config),baseline:explainer.baseline,
      timing:{durationMs:now()-started,inferenceCalls,explanationCalls,cacheHit:inferenceCalls===0&&rows.length>0,yields,maxSliceMs}};
    onProgress({done:data.events.length,total:data.events.length,phase:'complete',percent:100});
    return freeze(report);
  }
  function outcome(record){
    if(record.decision==='LEARNING')return 'warmup';
    if(record.label===null)return 'unknown';
    return record.label===1?(record.decision==='BLOCK'?'tp':'fn'):(record.decision==='BLOCK'?'fp':'tn');
  }
  function filter(report,options={}){
    const query=String(options.query||'').trim().toLowerCase(),decision=options.decision||'all',wanted=options.outcome||'all';
    const rows=report.records.filter(r=>{
      if(decision!=='all'&&r.decision!==decision)return false;
      if(wanted==='known'&&r.label===null)return false;
      if(wanted==='unknown'&&r.label!==null)return false;
      if(!['all','known','unknown'].includes(wanted)&&outcome(r)!==wanted)return false;
      if(!query)return true;
      return [r.event.id,r.event.amount,r.event.u,r.event.v,report.accounts[r.event.u]?.name,report.accounts[r.event.v]?.name].join(' ').toLowerCase().includes(query);
    });
    const value=r=>options.sort==='amount'?r.event.amount:options.sort==='score'?r.score:options.sort==='tau'?r.tauBefore:options.sort==='id'?r.event.id:r.event.t;
    const direction=options.direction==='desc'?-1:1;
    return rows.sort((a,b)=>{
      const x=value(a),y=value(b);
      if(x===null||y===null)return x===y?0:x===null?1:-1;
      return direction*(x<y?-1:x>y?1:0)||a.event.t-b.event.t||String(a.event.id).localeCompare(String(b.event.id));
    });
  }
  function summarize(report,records=report.records){
    const featureStats=xgb.featureDefinitions.map((definition,index)=>{
      let sum=0,absolute=0,values=0,min=Infinity,max=-Infinity,minValue=Infinity,maxValue=-Infinity;
      for(const r of records){const contribution=r.explanation.contributions[index],value=r.readableValues[index];sum+=contribution;absolute+=Math.abs(contribution);values+=value;min=Math.min(min,contribution);max=Math.max(max,contribution);minValue=Math.min(minValue,value);maxValue=Math.max(maxValue,value);}
      return {index,id:definition.id,label:definition.label,meanAbs:records.length?absolute/records.length:0,
        mean:records.length?sum/records.length:null,min:records.length?min:null,max:records.length?max:null,
        meanValue:records.length?values/records.length:null,minValue:records.length?minValue:null,maxValue:records.length?maxValue:null};
    });
    return {count:records.length,metrics:metrics(records,report.configuration),featureStats};
  }
  function csvCell(value){
    if(value===null||value===undefined)return '';
    // A text field beginning with a spreadsheet formula must remain text on export.
    let text=String(value);if(typeof value==='string'&&/^[=+\-@\t\r]/.test(text))text="'"+text;
    return /[",\n\r]/.test(text)?'"'+text.replace(/"/g,'""')+'"':text;
  }
  function toCSV(report,records=report.records){
    const names=xgb.featureDefinitions.map(f=>f.id);
    const header=['event_id','time_minutes','sender','recipient','amount_eur','probability','score_bits','threshold_bits','decision','settled','outcome','raw_margin','baseline',
      ...names.map(n=>n+'_value'),...names.map(n=>n+'_input'),...names.map(n=>n+'_shap')];
    const rows=records.map(r=>[r.event.id,r.event.t,report.accounts[r.event.u]?.name,report.accounts[r.event.v]?.name,r.event.amount,r.probability,r.score,r.tauBefore,r.decision,r.settled,r.label,r.rawMargin,r.explanation.baseline,...r.readableValues,...r.features,...r.explanation.contributions]);
    return [header,...rows].map(row=>row.map(csvCell).join(',')).join('\r\n')+'\r\n';
  }
  function toJSON(report,records=report.records){
    return JSON.stringify({...report,records,metrics:metrics(records,report.configuration),exportedPopulation:{count:records.length,total:report.records.length},featureDefinitions:xgb.featureDefinitions},null,2);
  }
  function clearCache(){histories.clear();}
  global.FraudAnalytics={run,filter,summarize,outcome,toCSV,toJSON,clearCache};
  if(typeof module!=='undefined')module.exports=global.FraudAnalytics;
})(typeof globalThis!=='undefined'?globalThis:window);
