/* Completed-run integration checks against the existing chronological runner. */
'use strict';
const assert=require('assert'),fs=require('fs'),path=require('path');
const analytics=require('../../tools/xgboost-analytics/analytics'),core=require('../../shared/runtime/model');
const xgb=require('../../shared/runtime/xgboost'),scenarios=require('../../shared/runtime/scenarios');
const metrics=require('../../shared/runtime/metrics'),policy=require('../../shared/runtime/policy');
const root=path.resolve(__dirname,'../..'),read=file=>JSON.parse(fs.readFileSync(path.join(root,file),'utf8'));
const model={...read('models/xgboost.json'),policy_validation:read('policy-validation.json').models.xgboost};
const reference=read('models/xgboost-explanations.json'),copy=value=>JSON.parse(JSON.stringify(value));
const immediate={yieldControl:()=>Promise.resolve(),budgetMs:3};
const sameDecision=(actual,expected)=>{
  for(const key of ['score','tauBefore','tauAfter','decision','settled','gap','previousPairCount'])assert.deepStrictEqual(actual[key],expected[key],key+' differs for '+actual.event.id);
  for(const key of ['id','kind','u','v','amount','t'])assert.deepStrictEqual(actual.event[key],expected.event[key],key+' differs');
};
const assertMetrics=(report,data)=>{
  const expected=metrics.metrics(report.records,data.truth||{},report.configuration.alpha,policy.costs(report.configuration));
  if(!expected.labeled){expected.f1=null;expected.f2=null;expected.errorCost=null;}
  for(const key of Object.keys(expected))assert.deepStrictEqual(report.metrics[key],expected[key],'metric '+key);
  assert.strictEqual(report.metrics.warmup,report.records.filter(r=>r.decision==='LEARNING').length);
  assert.strictEqual(report.metrics.assessedUnknown,report.records.filter(r=>r.decision!=='LEARNING'&&r.label===null).length);
};
function parseCSV(input){
  const rows=[];let row=[],cell='',quoted=false;
  for(let i=0;i<input.length;i++){
    const c=input[i];
    if(c==='"'){if(quoted&&input[i+1]==='"'){cell+='"';i++;}else quoted=!quoted;}
    else if(!quoted&&c===','){row.push(cell);cell='';}
    else if(!quoted&&c==='\r'&&input[i+1]==='\n'){row.push(cell);rows.push(row);row=[];cell='';i++;}
    else cell+=c;
  }
  assert(!quoted,'CSV quotes must close');return rows;
}

(async()=>{
  const data=scenarios.build('mixed','small',314);
  const configurations=[
    {decisionPolicy:'shared',alpha:.02,warmup:32,eta:.025},
    {decisionPolicy:'shared',alpha:.08,warmup:7,eta:.08},
    {decisionPolicy:'manual',manualTau:.3},
    {decisionPolicy:'tuned',falseBlockCost:2,missedFraudCost:30},
    {decisionPolicy:'auto',objective:'f1'},
    {decisionPolicy:'auto',objective:'f2'},
    {decisionPolicy:'auto',objective:'balanced_accuracy'},
  ];
  analytics.clearCache();let first=null;
  for(const configuration of configurations){
    const report=await analytics.run(model,data,reference,configuration,immediate);
    const expected=core.replay(model,data,data.events.length,report.configuration);
    assert.strictEqual(report.finalPosition,expected.position);assert.strictEqual(report.finalTime,data.events.at(-1).t);
    assert.strictEqual(report.finalTau,expected.state.tau);assert.deepStrictEqual(report.policyFit,expected.state.policyFit);
    assert.strictEqual(report.records.length,expected.state.decisions.length);
    report.records.forEach((record,i)=>sameDecision(record,expected.state.decisions[i]));
    assertMetrics(report,data);
    const reapplied=core.evaluatePolicy(model,report.records,report.configuration);
    reapplied.records.forEach((record,i)=>sameDecision(record,expected.state.decisions[i]));
    assert.strictEqual(reapplied.tau,expected.state.tau);
    if(first){
      assert.strictEqual(report.timing.inferenceCalls,0);assert.strictEqual(report.timing.explanationCalls,0);assert(report.timing.cacheHit);
      report.records.forEach((record,i)=>{
        assert.deepStrictEqual(record.features,first.records[i].features);assert.deepStrictEqual(record.explanation,first.records[i].explanation);
        assert.strictEqual(record.score,first.records[i].score);assert.strictEqual(record.probability,first.records[i].probability);
      });
    }else first=report;
  }
  let state=core.initial(model,data.accounts.length,first.configuration),index=0;
  for(const event of data.events){const prediction=core.score(model,state,event);if(prediction){const record=first.records[index++];assert.deepStrictEqual(record.features,prediction.features);assert.strictEqual(record.probability,prediction.probability);assert.strictEqual(record.rawMargin,xgb.raw(model,prediction.features));}core.step(model,state,event,prediction);}
  assert(Object.isFrozen(first)&&Object.isFrozen(first.records)&&Object.isFrozen(first.records[0].features)&&Object.isFrozen(first.records[0].event));
  assert.throws(()=>first.records[0].features[0]=99,TypeError);
  console.log('PASS final runner decisions, predictions, original cutoffs and metrics across all policy choices; policy changes reuse immutable scores');

  const accounts=[{id:0,name:'=SUM(1,2)'},{id:1,name:'Recipient, "quoted"\nline'},{id:2,name:'Third'}];
  const event=(id,t,u,v,amount)=>({id,t,u,v,amount,kind:'payment'});
  const prefix={name:'prefix',size:'test',seed:1,accounts,truth:{a:false,b:true,c:false},events:[event('a',1,0,1,100),event('b',2,1,0,600),event('c',3,0,2,10)]};
  const before=await analytics.run(model,prefix,reference,{warmup:1},immediate),saved=JSON.stringify(before);
  const extended=copy(prefix);extended.events.push(event('later',4,0,1,30000),{id:'deposit',t:8,u:-1,v:0,amount:900,kind:'deposit'},{id:'report',t:12,u:-1,v:1,amount:0,kind:'report',reference:'a'});
  extended.truth.later=true;
  const after=await analytics.run(model,extended,reference,{warmup:1},immediate);
  assert.deepStrictEqual(after.records.slice(0,before.records.length),before.records);assert.strictEqual(JSON.stringify(before),saved);
  assert.strictEqual(after.finalTime,12);assert.strictEqual(after.finalPosition,6);assert.strictEqual(after.records.length,4);
  const endDeposit=copy(prefix);endDeposit.events.push({id:'final-deposit',t:40,u:-1,v:0,amount:40,kind:'deposit'});
  const deposited=await analytics.run(model,endDeposit,reference,{warmup:1},immediate);assert.strictEqual(deposited.finalTime,40);assert.strictEqual(deposited.records.length,3);
  console.log('PASS later history and ending reports/deposits preserve original captured features, decisions and explanations');

  const equality=await analytics.run(model,prefix,reference,{decisionPolicy:'manual',manualTau:before.records[0].score},immediate);
  assert.strictEqual(equality.records[0].decision,'ALLOW');assert.strictEqual(equality.records[0].score,equality.records[0].tauBefore);
  const unknown=copy(prefix);unknown.truth={};
  const unlabelled=await analytics.run(model,unknown,reference,{decisionPolicy:'manual',manualTau:0},immediate);
  assert.strictEqual(unlabelled.metrics.labeled,0);assert.strictEqual(unlabelled.metrics.eligible,3);assert.strictEqual(unlabelled.metrics.blocks,3);assert.strictEqual(unlabelled.metrics.blockRate,1);
  for(const name of ['precision','recall','f1','f2','balanced_accuracy','errorCost'])assert.strictEqual(unlabelled.metrics[name],null,name);
  assert.strictEqual(analytics.filter(unlabelled,{outcome:'unknown'}).length,3);assert.strictEqual(analytics.filter(unlabelled,{outcome:'known'}).length,0);
  const warmup=await analytics.run(model,unknown,reference,{warmup:20},immediate);
  assert.strictEqual(warmup.metrics.warmup,3);assert.strictEqual(warmup.metrics.eligible,0);assert.strictEqual(warmup.metrics.blockRate,null);assert.strictEqual(warmup.finalTau,null);
  assert.strictEqual(analytics.filter(warmup,{decision:'LEARNING'}).length,3);
  assert.strictEqual(analytics.filter(warmup,{outcome:'warmup'}).length,3,'unknown warm-up rows remain warm-up rows');
  const empty=await analytics.run(model,{name:'empty',size:'test',seed:0,accounts,truth:{},events:[]},reference,{},immediate);
  assert.strictEqual(empty.finalPosition,0);assert.strictEqual(empty.finalTime,null);assert.strictEqual(empty.records.length,0);assert.strictEqual(empty.metrics.f1,null);
  const benignData=scenarios.build('benign','small',42),benign=await analytics.run(model,benignData,reference,{},immediate);
  assert.strictEqual(benign.metrics.tp+benign.metrics.fn,0);assert.strictEqual(benign.metrics.recall,null);assertMetrics(benign,benignData);
  console.log('PASS strict threshold equality, unknown outcomes, all-warm-up, empty scenarios, all-benign metrics and nonpayment completion');

  const subset=analytics.filter(after,{query:'quoted',sort:'amount',direction:'desc'});
  assert(subset.every(row=>row.event.u===1||row.event.v===1));assert(subset.every((row,i)=>!i||subset[i-1].event.amount>=row.event.amount));
  assert.strictEqual(analytics.filter(after,{decision:'BLOCK'}).length,after.records.filter(r=>r.decision==='BLOCK').length);
  assert.strictEqual(analytics.filter(after,{outcome:'fp'}).length,after.records.filter(r=>r.label===0&&r.decision==='BLOCK').length);
  assert.strictEqual(analytics.filter(after,{outcome:'fn'}).length,after.records.filter(r=>r.label===1&&r.decision==='ALLOW').length);
  const summary=analytics.summarize(after,subset);assert.strictEqual(summary.count,subset.length);assert.strictEqual(summary.featureStats.length,27);
  summary.featureStats.forEach((feature,i)=>assert.strictEqual(feature.meanAbs,subset.reduce((sum,row)=>sum+Math.abs(row.explanation.contributions[i]),0)/subset.length));
  const nothing=analytics.summarize(after,analytics.filter(after,{query:'no-such-payment'}));
  assert.strictEqual(nothing.count,0);assert.strictEqual(nothing.metrics.f1,null);assert(nothing.featureStats.every(f=>f.mean===null&&f.meanAbs===0&&f.minValue===null));
  const exported=JSON.parse(analytics.toJSON(after,subset));
  assert.strictEqual(exported.version,1);assert.deepStrictEqual(exported.records,subset);assert.strictEqual(exported.featureDefinitions.length,27);
  assert.deepStrictEqual(exported.exportedPopulation,{count:subset.length,total:after.records.length});assert.strictEqual(exported.metrics.requests,subset.length);
  const csv=parseCSV(analytics.toCSV(after,subset));
  assert.strictEqual(csv.length,subset.length+1);assert.strictEqual(csv[0].length,94);assert(csv.every(row=>row.length===94));
  assert(csv.slice(1).some(row=>row[2]==="'=SUM(1,2)"||row[3]==="'=SUM(1,2)"));
  assert(csv.slice(1).some(row=>row[2]===accounts[1].name||row[3]===accounts[1].name));
  assert.strictEqual(parseCSV(analytics.toCSV(after,[])).length,1);
  console.log('PASS filtered population statistics, decision/outcome filters and complete JSON/CSV exports including quoted text');

  analytics.clearCache();
  const cancelledData=scenarios.build('mixed','large',918),progress=[];let cancelled=false;
  const stopped=await analytics.run(model,cancelledData,reference,{},
    {budgetMs:1,cancelled:()=>cancelled,onProgress:p=>progress.push(p),yieldControl:async()=>{if(progress.at(-1).done>0)cancelled=true;}});
  assert.strictEqual(stopped,null);assert(progress.some(p=>p.done>0));assert(!progress.some(p=>p.phase==='complete'));
  const finished=await analytics.run(model,cancelledData,reference,{},immediate);
  assert.strictEqual(finished.timing.inferenceCalls,finished.records.length);assert.strictEqual(finished.timing.cacheHit,false);
  assert.strictEqual(await analytics.run(model,cancelledData,reference,{}, {...immediate,cancelled:()=>true}),null);
  const cached=await analytics.run(model,cancelledData,reference,{decisionPolicy:'manual',manualTau:1},immediate);assert(cached.timing.cacheHit);
  const changed=copy(cancelledData);changed.events.find(e=>e.kind==='payment').amount+=1000;
  assert((await analytics.run(model,changed,reference,{},immediate)).timing.inferenceCalls>0);
  const relabeled=copy(changed),firstId=relabeled.events.find(e=>e.kind==='payment').id;relabeled.truth[firstId]=!relabeled.truth[firstId];
  const relabeledResult=await analytics.run(model,relabeled,reference,{},immediate);assert(relabeledResult.timing.inferenceCalls>0);assert.strictEqual(relabeledResult.records[0].label,relabeled.truth[firstId]?1:0);
  const otherReference=copy(reference);otherReference.reference.description+=' Distinct reference identity.';delete otherReference.reference_id;otherReference.reference_id=xgb.fingerprint(otherReference);
  assert((await analytics.run(model,relabeled,otherReference,{},immediate)).timing.inferenceCalls>0);
  const mismatched=copy(model);mismatched.base_score+=1;
  await assert.rejects(()=>analytics.run(mismatched,relabeled,reference,{},immediate),/checkpoint differs/);
  const stale=copy(reference);stale.reference.row_count++;
  await assert.rejects(()=>analytics.run(model,relabeled,stale,{},immediate),/Incompatible/);
  console.log('PASS cancellation never publishes or caches partial reports; completed cache invalidates on data, outcomes and reference identity');

  const noFit={...model};delete noFit.policy_validation;
  for(const decisionPolicy of ['tuned','auto'])await assert.rejects(()=>analytics.run(noFit,prefix,reference,{decisionPolicy},immediate),/No historical threshold fit/);
  for(const options of [{trainingMode:'unsupervised'},{trainingMode:null},{mode:'enforce'},{mode:null},{warmup:0},{eta:-1},{alpha:0}])await assert.rejects(()=>analytics.run(model,prefix,reference,options,immediate));
  const undefinedModes=await analytics.run(model,prefix,reference,{trainingMode:undefined,mode:undefined},immediate);
  assert.strictEqual(undefinedModes.configuration.trainingMode,'supervised');assert.strictEqual(undefinedModes.configuration.mode,'shadow');
  assert(undefinedModes.records.every(record=>Number.isFinite(record.probability)&&record.settled));
  assert.throws(()=>core.evaluatePolicy(model,before.records,{mode:'enforce'}),/observed-history/);
  console.log('PASS missing historical fits, invalid controls and incompatible scoring/history modes rejected');

  const mutableData=copy(prefix),mutableReference=copy(reference),mutableModel=copy(model),mutableOptions={warmup:1};
  const isolated=await analytics.run(mutableModel,mutableData,mutableReference,mutableOptions,immediate),snapshot=JSON.stringify(isolated);
  assert(!Object.isFrozen(mutableData.accounts[0])&&!Object.isFrozen(mutableReference.reference)&&!Object.isFrozen(mutableOptions)&&!Object.isFrozen(mutableModel));
  mutableData.accounts[0].name='Changed after completion';mutableData.events[0].amount=999;mutableData.truth.a=true;
  mutableReference.reference.description='Changed after completion';mutableOptions.warmup=200;mutableModel.label='Changed after completion';
  assert.strictEqual(JSON.stringify(isolated),snapshot);
  console.log('PASS report metadata and captured rows are isolated from later changes to caller-owned inputs');
  console.log('Analytics integration checks passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
