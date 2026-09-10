/* Policy guarantees: cost optimum, label boundaries, frozen replay and rewind. */
'use strict';
const assert=require('assert'),fs=require('fs'),policy=require('./policy'),core=require('./model'),cmp=require('./comparison'),sc=require('./scenarios'),offline=require('./tune_policy');
const read=f=>JSON.parse(fs.readFileSync(__dirname+'/'+f,'utf8')),copy=x=>JSON.parse(JSON.stringify(x));
const bundle=read('model-bundle.json'),artifact=read('policy-validation.json'),dataset=read('xgb-training.json'),models=bundle.models,checks=[];
function pass(message){checks.push(message);console.log('PASS',message);}
function brute(rows,cost){
  const candidates=[...new Set(rows.map(r=>r.score)),-Number.EPSILON].map(tau=>{
    const fp=rows.filter(r=>r.label===0&&r.score>tau).length,fn=rows.filter(r=>r.label===1&&r.score<=tau).length,blocks=rows.filter(r=>r.score>tau).length;
    return {tau,fp,fn,blocks,loss:fp+cost*fn};
  });
  return candidates.sort((a,b)=>a.loss-b.loss||a.blocks-b.blocks)[0];
}
function bruteObjective(rows,objective){
  const validation=policy.frontier(rows);
  return validation.candidates.map(([tau,fp,fn,blocks])=>({tau,fp,fn,blocks,value:policy.classificationMetrics(validation.positives-fn,fp,fn,validation.negatives-fp)[objective]}))
    .sort((a,b)=>b.value-a.value||a.blocks-b.blocks)[0];
}
// Enumerate many small, tie-heavy cases, including 0 bits and unknown labels.
for(let seed=1;seed<=100;seed++){
  const rows=Array.from({length:24},(_,i)=>({score:(i*i+seed*i+seed)%7,label:(i+seed)%3-1}));
  for(const cost of [.01,.1,1,3.5,20,100,10000]){
    const fit=policy.select(policy.frontier(rows),{missedFraudCost:cost}),expected=brute(rows,cost);
    for(const key of['fp','fn','blocks','loss'])assert.strictEqual(fit[key],expected[key]);
    assert.deepStrictEqual(rows.map(r=>r.score>fit.tau),rows.map(r=>r.score>expected.tau));
  }
}
for(const rows of[[{score:4,label:0},{score:2,label:1}],[{score:0,label:0},{score:0,label:1}],[{score:7,label:1},{score:3,label:0},{score:9,label:-1}]]){
  for(const cost of [.01,1,100]){
    const fit=policy.select(policy.frontier(rows),{missedFraudCost:cost}),expected=brute(rows,cost);
    assert.strictEqual(fit.loss,expected.loss);assert.strictEqual(fit.blocks,expected.blocks);
  }
}
assert.throws(()=>policy.frontier([{score:1,label:-1}]),/both confirmed/);
assert.throws(()=>policy.select(null),/No historical/);
assert.throws(()=>policy.costs({missedFraudCost:0}),/positive/);
pass('Exact minimum cost agrees with exhaustive thresholds, including ties, unknowns, all-allow and all-block');

for(let seed=1;seed<=100;seed++){
  const rows=Array.from({length:31},(_,i)=>({score:(i*17+seed*3+i%5)%11,label:(i+seed)%4===0?1:(i+seed)%4===1?0:-1}));
  for(const objective of Object.keys(policy.objectives)){
    const fit=policy.autoTune(policy.frontier(rows),objective),expected=bruteObjective(rows,objective);
    assert.strictEqual(fit.tau,expected.tau);assert.strictEqual(fit.fp,expected.fp);assert.strictEqual(fit.fn,expected.fn);assert.strictEqual(fit.blocks,expected.blocks);assert.strictEqual(fit.objective,objective);assert.strictEqual(fit.candidatesTested,policy.frontier(rows).candidates.length);
  }
}
assert.throws(()=>policy.autoTune(null),/No historical/);assert.throws(()=>policy.autoTune(policy.frontier([{score:1,label:1},{score:0,label:0}]),'bogus'),/Unknown automatic/);
pass('Automatic F1, F2 and balanced-accuracy cutoffs agree with exhaustive validation search and deterministic ties');

assert.deepStrictEqual(artifact.sources,offline.signatures());
const split=Math.floor(dataset.episodes.length*.8),indices=artifact.provenance.validationEpisodeIndices;
assert.deepStrictEqual(indices,Array.from({length:dataset.episodes.length-split},(_,i)=>split+i));
assert(indices.every(i=>!artifact.provenance.trainingEpisodeIndices.includes(i)));
const heldout=dataset.episodes.slice(split),raw=offline.validationRows(models[0],heldout,'unsupervised');
assert.deepStrictEqual(policy.frontier(raw),artifact.models[models[0].id].unsupervised);
const relabeled=copy(heldout);relabeled.forEach(ep=>ep.events.forEach(e=>{e.label=e.label===1?0:1;e.fraud=true;}));
assert.deepStrictEqual(offline.validationRows(models[0],relabeled,'unsupervised').map(r=>r.score),raw.map(r=>r.score));
pass('Calibration matches held-out episodes and source hashes; labels cannot change historical model scores');

const data=sc.build('relay','small',7341),short={...data,events:data.events.slice(0,190)},report={id:'extra-report',kind:'report',u:-1,v:1,amount:0,t:short.events.at(-1).t+1,reference:'historical-fraud'};
const fits=[];
for(const trainingMode of['unsupervised','supervised']){
  const group=new cmp.Comparison(models,short,{mode:'shadow',trainingMode,decisionPolicy:'tuned'}),rows=group.seek(short.events.length);
  let commonBudget;
  rows.forEach(({model,result})=>{
    const state=result.state,fit=policy.select(model.policy_validation[trainingMode]);
    assert.deepStrictEqual(state.policyFit,fit);assert.strictEqual(state.tau,fit.tau);assert.strictEqual(state.calibration.length,0);
    assert(state.decisions.length>0&&state.decisions.every(r=>r.decision!=='LEARNING'&&r.tauBefore===fit.tau&&r.tauAfter===fit.tau));
    assert.deepStrictEqual(state.payments.map(e=>e.id),rows[0].result.state.payments.map(e=>e.id));
    const evaluation=cmp.metrics(state.decisions,short.truth,.02,state.errorCosts);
    commonBudget??=evaluation.budget;assert.strictEqual(evaluation.budget,commonBudget);
    let lastBlocks=-1;
    for(const cost of [.01,1,5,20,100,10000]){const v=policy.select(model.policy_validation[trainingMode],{missedFraudCost:cost});assert(v.blocks>=lastBlocks);lastBlocks=v.blocks;}
    const before=JSON.stringify(state.memory);core.step(model,state,report);assert.strictEqual(state.tau,fit.tau);assert.strictEqual(JSON.stringify(state.memory),before);
    fits.push({model:model.id,trainingMode,tau:fit.tau,impliedAlpha:fit.impliedAlpha,validationCost:fit.loss});
  });
}
assert(new Set(fits.map(f=>f.tau)).size>2);assert(new Set(fits.map(f=>f.impliedAlpha)).size>2);
pass('All nine models in both modes fit independent thresholds; increasing missed-fraud cost never reduces fitted blocks');
pass('Tuned replay freezes thresholds, skips warm-up, shares observed history and preserves the common ranking budget');

const perModel={statistics:{decisionPolicy:'manual',manualTau:5},gru:{decisionPolicy:'tuned',falseBlockCost:1,missedFraudCost:5},gru_mean:{decisionPolicy:'auto',objective:'f2'}};
const individual=new cmp.Comparison(models,short,{mode:'shadow',trainingMode:'unsupervised',alpha:.02,modelPolicies:perModel}),individualRows=individual.seek(short.events.length);
const byId=Object.fromEntries(individualRows.map(x=>[x.model.id,x.result.state]));
assert.strictEqual(byId.statistics.decisionPolicy,'manual');assert.strictEqual(byId.statistics.tau,5);assert.strictEqual(byId.statistics.policyFit,null);
assert.strictEqual(byId.gru.decisionPolicy,'tuned');assert.strictEqual(byId.gru.policyFit.costs.missedFraud,5);assert(Number.isFinite(byId.gru.tau));
assert.strictEqual(byId.gru_mean.decisionPolicy,'auto');assert.strictEqual(byId.gru_mean.policyFit.objective,'f2');assert(Number.isFinite(byId.gru_mean.tau));
const beforePolicies={statistics:byId.statistics.tau,gru:byId.gru.tau,gru_mean:byId.gru_mean.tau};
const changed={...perModel,gru:{...perModel.gru,missedFraudCost:100}};individual.reconfigure({mode:'shadow',trainingMode:'unsupervised',alpha:.02,modelPolicies:changed});const changedRows=individual.seek(short.events.length),changedById=Object.fromEntries(changedRows.map(x=>[x.model.id,x.result.state]));
assert.strictEqual(changedById.statistics.tau,beforePolicies.statistics);assert.strictEqual(changedById.gru_mean.tau,beforePolicies.gru_mean);assert.notStrictEqual(changedById.gru.tau,beforePolicies.gru);
pass('Individual model policies keep fixed, cost-tuned and auto-tuned settings isolated; changing one model cost changes only that model');

const model=models.find(m=>m.id==='gru_attention'),options={decisionPolicy:'tuned',trainingMode:'unsupervised',mode:'enforce'},runner=new core.Runner(model,short,options);
runner.seek(190);assert.deepStrictEqual(runner.seek(171).state,core.replay(model,short,171,options).state);
const different=copy(short);different.truth=Object.fromEntries(Object.keys(different.truth).map(k=>[k,!different.truth[k]]));different.events.slice(150).forEach(e=>e.amount+=1000);
assert.deepStrictEqual(core.replay(model,different,150,options).state,core.replay(model,short,150,options).state);
const a=core.replay(model,short,190,options).state,b=core.replay(model,short,190,{...options,alpha:.05,warmup:512}).state;
assert.deepStrictEqual(a.decisions,b.decisions);assert.strictEqual(a.tau,b.tau);
assert.throws(()=>core.initial({...model,policy_validation:{}},32,options),/No historical/);
pass('Rewind is exact; replay outcomes, future events, warm-up and ranking budget cannot refit tuned τ');
fs.writeFileSync(__dirname+'/policy-test-results.json',JSON.stringify({checks,fits,validation:artifact.provenance,notice:'Historical fit costs are calibration results, not independent test performance.'},null,2)+'\n');
