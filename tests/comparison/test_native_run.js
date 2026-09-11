/* Native bundles: Python/head parity, data binding, context and replay isolation. */
'use strict';
const assert=require('node:assert/strict');
const {execFile}=require('node:child_process');
const {promisify}=require('node:util');
const path=require('node:path');
const heads=require('../../shared/runtime/prediction-heads');
const native=require('../../tools/comparison/native-run');
const recording=require('../../models/implementations/native_recording');
const core=require('../../shared/runtime/model');
const ROOT=path.resolve(__dirname,'../..');
const copy=value=>JSON.parse(JSON.stringify(value));

async function main(){
function fixture(){
  const logits=[-2,-1,0,1,-3,0,2],events=[{id:'deposit',kind:'deposit',t:0,u:-1,v:0,amount:100}];
  for(let i=0;i<logits.length;i++)events.push({id:'p'+i,kind:'payment',t:i+1,u:i%2,v:1-i%2,amount:10+i,settled:i!==4});
  events.push({id:'report',kind:'report',t:8,u:-1,v:0,amount:0,reference:'p4'});
  return {schema:'native-fraud-comparison/v1',version:1,
    dataset:{schema:'payment-events/v1',name:'Native fixture',accounts:[{id:0,external_id:'a'},{id:1,external_id:'b'}],events,truth:{p4:true,p5:false}},
    model:{id:'dyg_tami_native',label:'DyGFormer + TAMI · native',checkpoint_id:'fixture-checkpoint',implementation:{status:'upstream-architecture'}},
    predictions:logits.map((logit,i)=>({id:'p'+i,logit})),
    heads:{empirical_tail:{version:1,id:'empirical_tail',reference_logits:logits.slice(0,4)},fixed_likelihood:{version:1,id:'fixed_likelihood',reference_count:4}},
    head:'empirical_tail',alpha:.25,evaluation_ids:['p4','p5','p6'],calibration:{ids:['p0','p1','p2','p3'],count:4},
    provenance:{same_dataset:true},history:{mode:'shadow',payments:'observed-attempts',timestamps:'strictly-before',deposits:false,reports:false}};
}
const bundle=fixture(),parsed=native.parse(bundle),model=native.checkpoint(parsed);
const options={mode:'shadow',trainingMode:'unsupervised',decisionPolicy:'manual',manualTau:heads.load(model.native_run.head).threshold(model.native_run.alpha)};

// The independent Python implementations are the cross-language scoring oracle.
const cases=[{state:bundle.heads.empirical_tail,logits:[-1000,-2,-1,0,.5,1,1000]},
  {state:{version:1,id:'empirical_tail',reference_logits:[0,0,0]},logits:[-1,0,1]},
  {state:bundle.heads.fixed_likelihood,logits:[-1000,-700,-2,0,2,700,1000]}];
const script='import json,sys\nfrom prediction_heads.registry import load_head\npayload=json.loads(sys.argv[1])\nprint(json.dumps([{key:value.tolist() for key,value in load_head(case["state"]).score(case["logits"]).items()} for case in payload]))';
const oracle=await promisify(execFile)(process.env.PYTHON_BINARY||'python',['-c',script,JSON.stringify(cases)],{cwd:ROOT,encoding:'utf8',timeout:30000});
const expected=JSON.parse(oracle.stdout);
cases.forEach((test,index)=>test.logits.forEach((logit,i)=>{
  const actual=heads.load(test.state).score(logit);
  for(const key of ['score','tail_probability'])assert.ok(Math.abs(actual[key]-expected[index][key][i])<=1e-12*Math.max(1,Math.abs(expected[index][key][i])),key+' parity at '+logit);
}));
const empirical=heads.load(bundle.heads.empirical_tail);
assert.equal(empirical.score(-100).tail_probability,.2);
assert.equal(empirical.score(-2).tail_probability,.4);
assert.equal(empirical.score(1).score,0);
assert.equal(empirical.score(-100).score>empirical.threshold(.2),false,'Exact tail boundary is allowed.');
assert.equal(empirical.score(-100).score>empirical.threshold(.2001),true);
for(const state of [{version:1,id:'empirical_tail',reference_logits:[]},{version:1,id:'empirical_tail',reference_logits:[1,0]},{version:1,id:'empirical_tail',reference_logits:[NaN]},{version:true,id:'fixed_likelihood',reference_count:1},{version:1,id:'fixed_likelihood',reference_count:0},{version:1,id:'fixed_likelihood',reference_count:1,extra:true}])assert.throws(()=>heads.load(state));
for(const alpha of [0,-1,1.001,NaN,Infinity])assert.throws(()=>empirical.threshold(alpha));
for(const logit of [NaN,Infinity,'0',null])assert.throws(()=>empirical.score(logit));
console.log('PASS Python/JavaScript head parity, ties, strict cutoff and numeric validation');

assert.equal(recording.validateDataset(model,parsed.dataset,options),true);
for(const mutate of [
  run=>run.predictions.pop(),run=>run.predictions.push({id:'extra',logit:1}),run=>run.predictions[1].id='p0',run=>run.predictions[0].logit=Infinity,
  run=>run.evaluation_ids.push('p4'),run=>run.evaluation_ids.push('absent'),run=>run.evaluation_ids.reverse(),
  run=>run.calibration.ids[0]='p5',run=>run.calibration.count=5,run=>run.heads.empirical_tail.reference_logits[0]=-9,
  run=>run.head='missing',run=>run.history.mode='enforce',run=>run.history.timestamps='including-current',
  run=>run.alpha=1,run=>run.alpha=0,run=>run.alpha='0.25',run=>run.model.id='__proto__'
]){const bad=copy(bundle);mutate(bad);assert.throws(()=>native.parse(bad));}
for(const mutate of [
  data=>data.accounts[0].external_id='different',data=>data.events[1].amount+=1,data=>data.events[1].t+=.1,
  data=>data.events[1].settled=false,data=>data.events[1].id='different',data=>data.events.at(-1).reference='p5',
  data=>data.events.splice(0,1),data=>{data.events[1].u=1;data.events[1].v=0;}
]){const bad=copy(parsed.dataset);mutate(bad);assert.throws(()=>new core.Runner(model,bad,options),/Dataset differs/);}
const relabeled=copy(parsed.dataset);relabeled.truth={p4:false,p5:true,p6:true};
assert.equal(recording.signature(relabeled),recording.signature(parsed.dataset));
assert.doesNotThrow(()=>new core.Runner(model,relabeled,options));
assert.throws(()=>new core.Runner(model,parsed.dataset,{...options,mode:'enforce'}),/shadow/);
assert.throws(()=>new core.Runner(model,parsed.dataset,{...options,trainingMode:'supervised'}),/supervised/);
// External calibration identities need not occur in this target dataset.
const external=copy(bundle);external.provenance.same_dataset=false;external.calibration.ids=['old0','old1','old2','old3'];assert.doesNotThrow(()=>native.parse(external));
console.log('PASS exact event-history binding, outcome isolation and malformed bundle rejection');

const runner=new core.Runner(model,parsed.dataset,options),final=copy(runner.seek(parsed.dataset.events.length));
assert.deepEqual(final.state.decisions.map(row=>row.decision),['CONTEXT','CONTEXT','CONTEXT','CONTEXT','BLOCK','ALLOW','ALLOW']);
assert.equal(final.state.payments.length,7,'Shadow replay retains every observed request.');
assert.equal(final.state.decisions[4].evidence.logit,-3);
assert.equal(final.state.decisions[4].evidence.tail_probability,.2);
assert.equal(final.state.tau,options.manualTau);
assert.equal(final.state.memory.every(vector=>vector.length===0),true,'Recording does not fabricate native embeddings.');
runner.seek(2);assert.equal(runner.preview().evaluationEligible,false);runner.seek(parsed.dataset.events.length);
assert.deepEqual(copy(runner.result()),final,'Rewinding preserves context, evidence, scores and decisions.');
const again=new core.Runner(model,relabeled,options);assert.deepEqual(again.seek(parsed.dataset.events.length).state.decisions.map(row=>row.score),final.state.decisions.map(row=>row.score));
const alteredAlpha=native.checkpoint(parsed,{alpha:.1}),otherHead=native.checkpoint(parsed,{head:'fixed_likelihood'});
assert.notEqual(alteredAlpha.checkpoint_id,model.checkpoint_id);assert.notEqual(otherHead.checkpoint_id,model.checkpoint_id);
const alternate=new core.Runner(alteredAlpha,parsed.dataset,{...options,manualTau:heads.threshold(.1)}).seek(parsed.dataset.events.length);
assert.equal(alternate.state.decisions[4].decision,'ALLOW');
assert.deepEqual(alternate.state.decisions.map(row=>row.evidence.logit),final.state.decisions.map(row=>row.evidence.logit));
assert.equal(parsed.head,'empirical_tail');assert.equal(parsed.alpha,.25,'Checkpoint selection leaves imported calibration unchanged.');
for(const alpha of [0,1,-1,Infinity,NaN,'0.25'])assert.throws(()=>native.checkpoint(parsed,{alpha}),/alpha/);
assert.equal(native.checkpoint(parsed,{alpha:.75}).native_run.alpha,.75);
assert.equal(native.checkpoint(parsed,{alpha:.0001}).native_run.alpha,.0001);
assert.equal(native.checkpoint(parsed).native_run.alpha,.25,'Rejected cutoff changes preserve a usable original bundle.');
// The low-level APIs must preserve head semantics without Comparison/UI help.
const direct=new core.Runner(model,parsed.dataset,{mode:'shadow'});
assert.equal(direct.state.decisionPolicy,'manual');
assert.equal(direct.state.predictionHead,'empirical_tail');
assert.equal(direct.state.alpha,.25);assert.equal(direct.state.tau,2);
direct.seek(parsed.dataset.events.length);
assert.deepEqual(copy(direct.result()).state.decisions,final.state.decisions);
for(const conflicting of [
  {mode:'shadow',decisionPolicy:'shared',alpha:.49,warmup:1},
  {mode:'shadow',decisionPolicy:'manual',manualTau:0,alpha:.49,predictionHead:'fixed_likelihood'},
  {mode:'shadow',decisionPolicy:'auto',objective:'f1'}
]){
  direct.reconfigure(conflicting);
  assert.equal(direct.state.tau,2);assert.equal(direct.state.alpha,.25);assert.equal(direct.state.predictionHead,'empirical_tail');
  assert.deepEqual(copy(direct.result()).state.decisions,final.state.decisions);
  direct.seek(2);direct.seek(parsed.dataset.events.length);
  assert.deepEqual(copy(direct.result()).state.decisions,final.state.decisions);
}
const rescored=core.evaluatePolicy(model,final.state.decisions,{decisionPolicy:'shared',alpha:.49,warmup:1});
assert.equal(rescored.tau,2);assert.deepEqual(copy(rescored.records),final.state.decisions);
const bare=core.initial(model,parsed.dataset.accounts.length,{mode:'shadow'});
assert.equal(bare.tau,2);assert.equal(bare.predictionHead,'empirical_tail');
assert.throws(()=>core.initial(model,2,{mode:'enforce'}),/shadow/);
assert.throws(()=>core.initial(model,2,{mode:'shadow',trainingMode:'supervised'}),/supervised/);
assert.throws(()=>direct.reconfigure({mode:'enforce'}),/same-history/);
assert.throws(()=>direct.reconfigure({mode:'shadow',trainingMode:'supervised'}),/same-history/);
assert.throws(()=>core.evaluatePolicy(model,final.state.decisions,{trainingMode:'supervised'}),/supervised/);
const legacy=require('../../model-bundle.json').models[0];
const contextOnly=core.evaluatePolicy(legacy,[{event:parsed.dataset.events[1],score:10,evaluationEligible:false}],{decisionPolicy:'shared',warmup:1});
assert.equal(contextOnly.tau,null,'Context records cannot start or adjust adaptive calibration.');
assert.equal(contextOnly.records[0].decision,'CONTEXT');
const atStart=new core.Runner(model,parsed.dataset,options);atStart.seek(1);
assert.throws(()=>core.score(model,atStart.state,{...parsed.dataset.events[1],amount:9999}),/differs/);
console.log('PASS context exclusion, native evidence, rewind parity and independent alpha/head checkpoints');

// Service runs use the same per-model policies as every browser model. Recorded
// thresholds are authoritative because enforcement changes later native logits.
const policy=require('../../shared/runtime/policy');
const {Comparison}=require('../../tools/comparison/comparison');
function liveFixture(mode,decisionPolicy,contextCount=0){
  const run=fixture(),head=heads.load(run.heads.fixed_likelihood);
  run.version=2;run.schema='native-fraud-comparison/v2';run.head='fixed_likelihood';
  run.options={mode,trainingMode:'unsupervised',decisionPolicy,predictionHead:run.head,alpha:.25,eta:.1,warmup:2,manualTau:1.5,falseBlockCost:2,missedFraudCost:5,objective:'f1'};
  run.history={...run.history,mode,payments:mode==='shadow'?'observed-attempts':'accepted-payments'};
  const validation=policy.frontier(run.predictions.map((row,index)=>({score:head.score(row.logit).score,label:index===4?1:0})));
  run.model.policy_validation={unsupervised:validation};
  run.policy_state={...policy.initialize(run.model,'unsupervised',run.options),...run.options};
  let tau=run.policy_state.tau;const calibration=[];
  run.predictions=run.predictions.map((row,index)=>{
    const prediction=head.score(row.logit),tauBefore=tau,context=index<contextCount,decision=context?'CONTEXT':tau===null?'LEARNING':prediction.score>tau?'BLOCK':'ALLOW';
    if(decisionPolicy==='shared'&&!context){
      if(decision==='LEARNING'){calibration.push(prediction.score);if(calibration.length===2)tau=Math.max(...calibration);}
      else tau+=.1*((decision==='BLOCK'?1:0)-.25);
    }
    return {...row,...prediction,tauBefore,tauAfter:tau,decision,settled:mode==='shadow'||decision!=='BLOCK',evaluationEligible:!context,evidence:{logit:row.logit,head:run.head,...prediction}};
  });
  run.evaluation_ids=run.predictions.filter(row=>row.evaluationEligible).map(row=>row.id);return run;
}
for(const mode of ['shadow','enforce'])for(const decisionPolicy of ['shared','manual','tuned','auto']){
  const run=liveFixture(mode,decisionPolicy),checkpoint=native.checkpoint(run),data=native.parse(run).dataset;
  const runner=new core.Runner(checkpoint,data,run.options),completed=copy(runner.seek(data.events.length));
  const select=row=>({id:row.event?.id??row.id,score:row.score,tauBefore:row.tauBefore,tauAfter:row.tauAfter,decision:row.decision,settled:row.settled});
  assert.deepEqual(completed.state.decisions.map(select),run.predictions.map(select));
  assert.equal(completed.state.payments.length,run.predictions.filter(row=>row.settled).length);
  assert.equal(completed.state.decisionPolicy,decisionPolicy);
  runner.seek(3);runner.seek(data.events.length);assert.deepEqual(copy(runner.result()),completed);
  const {mode:unusedMode,trainingMode:unusedTraining,...perModel}=run.options;
  const comparison=new Comparison([checkpoint],data,{mode,trainingMode:'unsupervised',modelPolicies:{[checkpoint.id]:perModel}});
  comparison.seek(data.events.length);
  assert.deepEqual(comparison.runners.get(checkpoint.id).state.decisions.map(select),run.predictions.map(select));
  assert.equal(comparison.evaluationRecords(checkpoint.id).length,decisionPolicy==='shared'?5:7);
  const changed={...run.options,decisionPolicy:'manual',manualTau:123};
  assert.throws(()=>new core.Runner(checkpoint,data,changed),/settings changed/);
  assert.throws(()=>native.checkpoint(run,{head:'empirical_tail'}),/settings changed/);
  if(mode==='shadow'){
    assert.throws(()=>runner.reconfigure(changed),/settings changed/);
    assert.deepEqual(copy(runner.result()),completed,'Rejected policy changes leave the existing run intact.');
  }
  for(const mutate of [value=>value.predictions[0].tauAfter=999,value=>value.predictions[4].settled=!value.predictions[4].settled,value=>value.options.mode=value.options.mode==='enforce'?'shadow':'enforce',value=>value.evaluation_ids.pop(),value=>value.predictions[0].decision='INVALID']){
    const bad=copy(run);mutate(bad);assert.throws(()=>native.parse(bad));
  }
}
const contextual=liveFixture('shadow','shared',2),contextualModel=native.checkpoint(contextual);
const contextualRunner=new core.Runner(contextualModel,native.parse(contextual).dataset,contextual.options);
contextualRunner.seek(3);contextualRunner.reconfigure(contextual.options);
assert.equal(contextualRunner.state.calibration.length,0,'Restored context cannot count as threshold warm-up.');
contextualRunner.seek(5);contextualRunner.reconfigure(contextual.options);
assert.equal(contextualRunner.state.calibration.length,2,'Restored native warm-up counts actual learning requests.');
assert.notEqual(contextualRunner.state.tau,null);
console.log('PASS live service policies, native enforcement settlement, rewind and stale-settings rejection');
function supervisedFixture(mode,decisionPolicy){
  const run=liveFixture(mode,decisionPolicy),score=logit=>(Math.max(logit,0)+Math.log1p(Math.exp(-Math.abs(logit))))/Math.LN2;
  run.options={...run.options,trainingMode:'supervised',predictionHead:'fraud_linear',manualTau:1};
  run.head='fraud_linear';run.heads={fraud_linear:{version:1,id:'fraud_linear',kind:'learned-fraud',input:'transaction-representation',output:'fraud-logit',score_input:'fraud-logit',score_output:'fraud-probability',score_transform:'softplus(logit)/ln(2)'}};
  const logits=[-1000,-2,0,2,1000,-1,1];
  run.model.policy_validation={supervised:policy.frontier(logits.map((logit,index)=>({score:score(logit),label:index===4?1:0})))};
  run.policy_state={...policy.initialize(run.model,'supervised',run.options),...run.options};
  let tau=run.policy_state.tau;const calibration=[];
  run.predictions=logits.map((logit,index)=>{
    const value=score(logit),tauBefore=tau,decision=tau===null?'LEARNING':value>tau?'BLOCK':'ALLOW';
    if(decisionPolicy==='shared'){
      if(decision==='LEARNING'){calibration.push(value);if(calibration.length===2)tau=Math.max(...calibration);}
      else tau+=.1*((decision==='BLOCK'?1:0)-.25);
    }
    return {id:'p'+index,logit,fraud_logit:logit,score:value,tauBefore,tauAfter:tau,decision,settled:mode==='shadow'||decision!=='BLOCK',evaluationEligible:true,
      evidence:{kind:'native-fraud',version:1,fraud_logit:logit,fraud_probability:heads.sigmoid(logit),head:'fraud_linear',alpha:.25}};
  });
  return run;
}
for(const mode of ['shadow','enforce'])for(const strategy of ['shared','manual','tuned','auto']){
  const run=supervisedFixture(mode,strategy),model=native.checkpoint(run),data=native.parse(run).dataset;
  const runner=new core.Runner(model,data,run.options),final=copy(runner.seek(data.events.length));
  assert.equal(final.state.trainingMode,'supervised');assert.equal(final.state.predictionHead,'fraud_linear');
  assert.deepEqual(final.state.decisions.map(row=>row.evidence),run.predictions.map(row=>row.evidence));
  assert.deepEqual(final.state.decisions.map(row=>row.score),run.predictions.map(row=>row.score));
  assert.deepEqual(final.state.decisions.map(row=>row.decision),run.predictions.map(row=>row.decision));
  assert.ok(final.state.decisions.every(row=>!Object.hasOwn(row.evidence,'logit')&&!Object.hasOwn(row.evidence,'tail_probability')));
  runner.seek(2);runner.seek(data.events.length);assert.deepEqual(copy(runner.result()),final);
  assert.throws(()=>new core.Runner(model,data,{...run.options,trainingMode:'unsupervised'}),/settings changed/);
  const relabeled=copy(data);relabeled.truth={p0:true,p1:true,p2:true,p3:true,p4:false,p5:true,p6:true};
  assert.deepEqual(new core.Runner(model,relabeled,run.options).seek(data.events.length).state.decisions.map(row=>row.score),run.predictions.map(row=>row.score));
  for(const mutate of [value=>value.predictions[0].evidence.kind='native-likelihood',value=>value.predictions[1].evidence.fraud_probability=.9,value=>value.predictions[0].evidence.tail_probability=.1,value=>value.predictions[3].score+=.5,value=>value.heads.fraud_linear.kind='link',value=>value.options.trainingMode='unsupervised']){
    const bad=copy(run);mutate(bad);assert.throws(()=>native.parse(bad));
  }
}
const {Client,modeCapabilities}=require('../../tools/comparison/native-client');
const capability={available:true,checkpoint_id:'fallback',capabilities:{training_modes:['unsupervised','supervised'],training_mode_capabilities:{unsupervised:{available:true,checkpoint_id:'link-weights',prediction_heads:['empirical_tail'],default_head:'empirical_tail',decision_policies:['shared']},supervised:{available:false,error:'Missing supervised checkpoint.',prediction_heads:[],decision_policies:[]}}}};
assert.equal(modeCapabilities(capability,'unsupervised').checkpoint_id,'link-weights');
assert.equal(modeCapabilities(capability,'supervised').available,false);
assert.match(modeCapabilities(capability,'supervised').error,/Missing supervised/);
assert.equal(modeCapabilities({available:true,capabilities:{training_modes:['unsupervised'],prediction_heads:['empirical_tail'],decision_policies:['shared']}},'supervised').available,false);
const previousFetch=global.fetch;
try{
  let requests=0,served=supervisedFixture('shadow','manual');
  const catalog={id:'dyg_tami_native',available:true,capabilities:{training_mode_capabilities:{supervised:{available:true,checkpoint_id:served.model.checkpoint_id,prediction_heads:['fraud_linear'],decision_policies:['manual'],default_head:'fraud_linear'}}}};
  global.fetch=async()=>{requests++;return {ok:true,json:async()=>copy(served)};};
  const client=new Client(),data=native.parse(served).dataset;
  await client.predict(catalog,data,served.options);await client.predict(catalog,data,served.options);assert.equal(requests,1);
  served.model.checkpoint_id='different-trained-weights';catalog.capabilities.training_mode_capabilities.supervised.checkpoint_id=served.model.checkpoint_id;
  const changed=await client.predict(catalog,data,served.options);assert.equal(requests,2);assert.match(changed.checkpoint_id,/different-trained-weights/);
  catalog.capabilities.training_mode_capabilities.supervised.feature_contract='different-representation-schema';
  await client.predict(catalog,data,served.options);assert.equal(requests,3);
}finally{global.fetch=previousFetch;}
console.log('PASS supervised fraud evidence, extreme logits, all policies, outcome isolation and per-mode capabilities');
}
main().catch(error=>{console.error(error);process.exitCode=1;});
