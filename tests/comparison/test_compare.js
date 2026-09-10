/* Model registry, parity, and comparison-isolation checks. */
'use strict';
const fs=require('fs'),assert=require('assert'),path=require('path');
const core=require('../../shared/runtime/model'),adapters=require('../../shared/runtime/model-adapters'),sc=require('../../shared/runtime/scenarios'),cmp=require('../../tools/comparison/comparison');
const read=f=>JSON.parse(fs.readFileSync(path.join(__dirname,'../..',f),'utf8'));
const bundle=read('model-bundle.json'),models=bundle.models;
assert.strictEqual(models.length,9);
assert.strictEqual(bundle.default,'gru_attention');
assert.deepStrictEqual(models.map(m=>m.id),['statistics','gru','gru_mean','gru_attention','xgboost','dygformer','tami','dyg_tami','dyg_tami_gnn']);
assert(models.every(m=>adapters.get(m)));
console.log('PASS all nine model checkpoints resolve through the adapter registry');

for(const flags of [0,7,96]){
  const labeled=sc.trainingLabeled(flags,811),split=Math.floor(labeled.episodes.length*.8),observed=labeled.episodes.flatMap((e,i)=>e.events.filter(x=>x.label===1&&i<split)).length;
  assert.strictEqual(observed,flags);assert.strictEqual(labeled.validation_flags_used,flags?Math.max(1,Math.floor(flags*.25)):0);
}
console.log('PASS offline generator produces the requested training and validation flag counts');

for(const model of models){
  const payload=read('parity/'+model.id+'.json'),episode=payload.episode,expected=payload.expected;
  let state=core.initial(model,episode.accounts,{trainingMode:'unsupervised'}),max=0;
  episode.events.forEach((e,i)=>{
    if(e.kind==='payment'){
      const p=core.score(model,state,e),x=expected[i];
      max=Math.max(max,Math.abs(p.score-x.scores[0]));
      if(model.family==='xgboost'){
        // No-label XGBoost mode deliberately does not expose its supervised
        // probability; its parity fixture therefore contains a single
        // certainty bucket instead.
        assert.strictEqual(p.trainingMode,'unsupervised');
        assert.strictEqual(p.probability,null);
        assert.strictEqual(x.probabilities[0][0].length,1);
      }else if(model.family==='temporal_family'){
        p.probabilities.forEach((a,k)=>a.forEach((v,j)=>max=Math.max(max,Math.abs(v-x.probabilities[k][0][j]))));
      }else p.probabilities.forEach((a,k)=>a.forEach((v,j)=>max=Math.max(max,Math.abs(v-x.probabilities[k][0][j]))));
    }
    core.apply(model,state,e);
    state.memory.forEach((a,n)=>a.forEach((v,j)=>max=Math.max(max,Math.abs(v-expected[i].memory_after[0][n][j]))));
  });
  assert(max<1e-10,model.id+' parity '+max);
  console.log('PASS parity',model.id,'max error',max);
}

for(const model of models){
  const payload=read('parity-supervised/'+model.id+'.json'),episode=payload.episode,expected=payload.expected;
  let state=core.initial(model,episode.accounts,{trainingMode:'supervised'}),max=0;
  episode.events.forEach((e,i)=>{
    if(e.kind==='payment'){
      const p=core.score(model,state,e),x=expected[i];
      assert.strictEqual(p.trainingMode,'supervised');
      max=Math.max(max,Math.abs(p.score-x.scores[0]));
      max=Math.max(max,Math.abs(p.probability-x.probabilities[0][0][1]));
    }
    core.apply(model,state,e);
    state.memory.forEach((a,n)=>a.forEach((v,j)=>max=Math.max(max,Math.abs(v-expected[i].memory_after[0][n][j]))));
  });
  assert(max<1e-10,model.id+' supervised parity '+max);
  console.log('PASS supervised parity',model.id,'max error',max);
}

const data=sc.build('mixed','medium',314),shadow=new cmp.Comparison(models,data,{mode:'shadow',alpha:.02});
const rows=shadow.seek(data.events.length),paymentIds=rows.map(x=>x.result.state.payments.map(e=>e.id));
assert(paymentIds.every(x=>JSON.stringify(x)===JSON.stringify(paymentIds[0])));
assert(rows.every(x=>x.result.state.mode==='shadow'));
assert(rows.every(x=>x.result.state.modelOwner===core.initial(x.model,data.accounts.length).modelOwner));
const metrics=rows.map(x=>cmp.metrics(x.result.state.decisions,data.truth,.02));
assert(metrics.every(x=>x.requests>0&&x.labeled>0));
console.log('PASS shadow comparison gives every model the same completed-payment history');

const probe=data.events.findIndex(e=>e.kind==='payment')+1;
const flagged=new cmp.Comparison(models,data,{mode:'shadow',alpha:.02,trainingMode:'supervised'}),frows=flagged.seek(probe);
assert(frows.every(x=>x.result.state.trainingMode==='supervised'));
assert(frows.every(x=>x.prediction&&Number.isFinite(x.prediction.score)));
assert(frows.every(x=>Number.isFinite(x.prediction.probability)));
console.log('PASS flagged-history comparison activates a supervised head for every model');

const enforce=new cmp.Comparison(models,data,{mode:'enforce',alpha:.02}),erows=enforce.seek(data.events.length);
assert(erows.some((x,i)=>x.result.state.payments.length!==erows[0].result.state.payments.length));
console.log('PASS enforce comparison keeps independent histories when models block differently');

const state=core.initial(models[0],data.accounts.length),request=data.events.find(e=>e.kind==='payment');
assert.throws(()=>core.score(models[1],state,request),/Model\/state mismatch/);
assert.throws(()=>core.apply(models[1],state,{id:'bad-owner',t:1,u:-1,v:0,amount:1,kind:'deposit'}),/Model\/state mismatch/);
const supervisedState=core.initial(models[0],data.accounts.length,{trainingMode:'supervised'});
assert.notStrictEqual(state.modelOwner,supervisedState.modelOwner);
assert.throws(()=>core.score(models[1],supervisedState,request),/Model\/state mismatch/);
console.log('PASS model/state ownership prevents accidental checkpoint mixing');

const tami=models.find(m=>m.id==='tami'),tamiData=sc.build('relay','small',42),tamiState=core.initial(tami,tamiData.accounts.length);
const tamiEvent=tamiData.events.find(e=>e.kind==='payment');core.apply(tami,tamiState,tamiEvent);
assert.strictEqual(tamiState.adapterState.pairHistory[tamiEvent.u][tamiEvent.v].length,1);
console.log('PASS TAMI adapter maintains bounded directed pair history in replay state');

console.log('Comparison checks passed. Metrics:',JSON.stringify(metrics.map((m,i)=>({id:models[i].id,precision:m.precision,recall:m.recall,recallAtBudget:m.recallAtBudget}))));
