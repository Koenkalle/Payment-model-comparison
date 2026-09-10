'use strict';
const assert=require('assert'),fs=require('fs'),path=require('path'),crypto=require('crypto');
const xgb=require('../../shared/runtime/xgboost'),core=require('../../shared/runtime/model');
const root=path.resolve(__dirname,'../..'),read=file=>JSON.parse(fs.readFileSync(path.join(root,file),'utf8'));
const close=(actual,expected,message)=>assert(Math.abs(actual-expected)<1e-11,(message||'values differ')+': '+actual+' vs '+expected);
const copy=value=>JSON.parse(JSON.stringify(value));
const model=read('models/xgboost.json'),sidecar=read('models/xgboost-explanations.json'),fixture=read('tests/xgboost-analytics/explanation-oracle.json');

assert.deepStrictEqual(xgb.featureDefinitions.map(f=>f.id),model.features);
assert.strictEqual(xgb.featureDefinitions.length,27);
assert(xgb.featureDefinitions.every(f=>f.label&&f.unit&&f.description&&f.transform&&f.window&&f.source));
assert.strictEqual(sidecar.checkpoint_sha256,crypto.createHash('sha256').update(fs.readFileSync(path.join(root,'models/xgboost.json'))).digest('hex'));
assert.strictEqual(xgb.modelFingerprint(model),sidecar.model_fingerprint);
assert.strictEqual(xgb.fingerprint('abc'),crypto.createHash('sha256').update('"abc"').digest('hex'));
console.log('PASS ordered feature dictionary and independent SHA-256 / Python fingerprint parity');

const toy=xgb.createExplainer(fixture.model,fixture.sidecar);
for(const {values,expected} of fixture.cases){
  const actual=toy.explain(values);
  close(actual.baseline,expected.baseline,'oracle baseline');close(actual.rawMargin,expected.rawMargin,'oracle raw margin');
  actual.contributions.forEach((value,i)=>close(value,expected.contributions[i],'oracle feature '+i));
  close(actual.baseline+actual.contributions.reduce((sum,value)=>sum+value,0),actual.rawMargin,'additivity');
  assert(actual.contributions.slice(3).every(value=>value===0));
}
const a=toy.explain(fixture.cases[0].values),b=toy.explain(fixture.cases[1].values);
assert.strictEqual(a.rawMargin,b.rawMargin);assert.notDeepStrictEqual(a.contributions,b.contributions);
a.contributions[0]=999;assert.notStrictEqual(toy.explain(fixture.cases[0].values).contributions[0],999);
console.log('PASS independent permutation oracle: repeated splits, constant tree, unused features, boundary equality, off-path cache and immutable results');

const changed=copy(model);changed.trees[0].threshold+=1;
assert.throws(()=>xgb.createExplainer(changed,sidecar),/checkpoint differs/);
for(const mutate of [s=>s.feature_schema_version++,s=>s.features.reverse(),s=>s.reference.row_count++,s=>s.node_counts[0][0]++,s=>s.reference.convention='hessian_cover',s=>s.checkpoint_sha256='missing']){
  const broken=copy(sidecar);mutate(broken);assert.throws(()=>xgb.createExplainer(model,broken),/Incompatible/);
}
const broken=copy(sidecar);broken.node_counts[0][1]++;delete broken.reference_id;broken.reference_id=xgb.fingerprint(broken);
assert.throws(()=>xgb.createExplainer(model,broken),/child counts/);
assert.throws(()=>toy.explain([1,2]),/finite value/);
assert.throws(()=>toy.explain(Array(27).fill(NaN)),/finite value/);
assert.throws(()=>xgb.treePath(model,Array(27).fill(0),-1),/Tree index/);
console.log('PASS incompatible schemas, checkpoint edits, corrupt reference counts and invalid rows rejected');

let state=core.initial(model,3,{trainingMode:'supervised',mode:'shadow'});
const payment=(id,t,u,v,amount,extra={})=>({id,t,u,v,amount,kind:'payment',...extra});
let description=xgb.describe(model,state,payment('first',20,0,1,150));
assert.strictEqual(description.readableValues[12],0);assert.strictEqual(description.readableValues[13],0);
assert.strictEqual(description.readableValues[25],150);assert.strictEqual(description.readableValues[26],150);
assert.strictEqual(description.readableValues[1],4);close(description.values[1],4/9);
const before=JSON.stringify(state);xgb.features(model,state,payment('probe',20,0,1,150));assert.strictEqual(JSON.stringify(state),before);
core.apply(model,state,payment('settled',1,0,1,200));
core.apply(model,state,payment('unsettled',2,0,1,900,{settled:false}));
core.apply(model,state,{id:'deposit',t:3,u:-1,v:0,amount:50,kind:'deposit'});
core.apply(model,state,{id:'report',t:4,u:-1,v:0,amount:0,kind:'report'});
description=xgb.describe(model,state,payment('boundary',61,0,1,100));
assert.strictEqual(description.readableValues[2],1);assert.strictEqual(description.readableValues[6],200);
assert.strictEqual(description.readableValues[10],3);assert.strictEqual(description.readableValues[11],2);
assert.strictEqual(description.readableValues[12],58);assert.strictEqual(description.readableValues[13],59);
assert.strictEqual(description.readableValues[18],1);assert.strictEqual(description.readableValues[19],1);
assert.strictEqual(description.readableValues[25],.5);assert.strictEqual(description.readableValues[26],.5);
assert.strictEqual(xgb.describe(model,state,payment('after-boundary',61.001,0,1,100)).readableValues[18],0);
const frozen=copy(description);core.apply(model,state,payment('later',62,1,0,800));assert.deepStrictEqual(description,frozen);
console.log('PASS causal readable inputs, amount-bin equality, first-activity gaps, denominator floors, settlement and inclusive 60-minute history');

const explainer=xgb.createExplainer(model,sidecar),parity=read('parity-supervised/xgboost.json');
state=core.initial(model,parity.episode.accounts,{trainingMode:'supervised'});let checked=0,maxError=0;
for(let i=0;i<parity.episode.events.length;i++){
  const event=parity.episode.events[i];
  if(event.kind==='payment'){
    const prediction=core.score(model,state,event),values=xgb.features(model,state,event),explanation=explainer.explain(values);
    assert.deepStrictEqual(values,prediction.features);
    close(xgb.probability(explanation.rawMargin),prediction.probability,'adapter probability');
    close(prediction.score,parity.expected[i].scores[0],'Python inference parity');
    const total=explanation.baseline+explanation.contributions.reduce((sum,value)=>sum+value,0);maxError=Math.max(maxError,Math.abs(total-explanation.rawMargin));
    close(total,explanation.rawMargin,'checkpoint attribution additivity');
    const tree=checked%model.trees.length,trace=xgb.treePath(model,values,tree);close(trace.weightedLeaf,model.learning_rate*xgb.treeValue(model.trees[tree],values),'weighted path');
    trace.steps.forEach(step=>assert.strictEqual(step.direction,values[step.feature]<=step.threshold?'left':'right'));
    checked++;
  }core.apply(model,state,event);
}
console.log('PASS '+checked+' real checkpoint explanations, Python inference parity and weighted tree paths; maximum additive error '+maxError);
