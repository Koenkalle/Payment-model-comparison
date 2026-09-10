'use strict';
const fs=require('fs'),assert=require('assert'),path=require('path');
const core=require('../../shared/runtime/model'),sc=require('../../shared/runtime/scenarios');
const read=f=>JSON.parse(fs.readFileSync(path.join(__dirname,'../..',f),'utf8'));
const model=read('model.json'),ep=read('parity-input.json'),expected=read('parity-expected.json');
const copy=x=>JSON.parse(JSON.stringify(x));
let s=core.initial(model,ep.accounts),maxError=0,checks=[];
function pass(msg){checks.push(msg);console.log('PASS',msg);}
ep.events.forEach((e,i)=>{
  if(e.kind==='payment'){
    const p=core.score(model,s,e);maxError=Math.max(maxError,Math.abs(p.score-expected[i].scores[0]));
    p.probabilities.forEach((a,k)=>a.forEach((v,j)=>maxError=Math.max(maxError,Math.abs(v-expected[i].probabilities[k][0][j]))));
  }
  core.apply(model,s,e);s.memory.forEach((a,n)=>a.forEach((v,j)=>maxError=Math.max(maxError,Math.abs(v-expected[i].memory_after[0][n][j]))));
});
assert(maxError<1e-10);pass('Python/JavaScript parity for every probability, request score, and recurrent state');

// Exercise the learned-cutoff branch independently of a checkpoint's trained default.
const replayOptions={warmup:128};
const data=sc.build('relay','small'),runner=new core.Runner(model,data,replayOptions),prefix=runner.seek(data.startIndex),state=copy(prefix.state),event=data.events[data.startIndex],before=copy(state);
const p=core.score(model,state,event),changed=core.score(model,state,{...event,amount:20000,t:event.t+2000});
assert.deepStrictEqual(state,before);assert.deepStrictEqual(p.probabilities,changed.probabilities);assert.notStrictEqual(p.score,changed.score);
pass('Request scoring does not mutate state; its amount and gap cannot leak into their predictive distributions');
const one=core.step(model,state,event);assert.strictEqual(one.tauBefore,before.tau);assert.strictEqual(one.score,p.score);
assert(Math.abs(one.tauAfter-one.tauBefore-state.eta*((one.decision==='BLOCK'?1:0)-state.alpha))<1e-12);
pass('Decision uses the pre-request cutoff; quantile update occurs afterward');

s=core.initial(model,32);s.tau=0;
const attempt={id:'blocked',t:2,u:0,v:1,amount:900,kind:'payment'};
const blocked=core.step(model,s,attempt);assert.strictEqual(blocked.decision,'BLOCK');assert.strictEqual(s.payments.length,0);assert.strictEqual(s.outValue[0],0);assert.strictEqual(s.inValue[1],0);assert.strictEqual(s.pairs[0][1],0);assert(s.incidents.every(x=>!x.length));assert(s.memory[0].some(x=>x!==0)&&s.memory[1].some(x=>x!==0));
pass('Blocked attempt updates both behavioral memories without completed edges, pair counts, or transferred value');
s=core.initial(model,32);core.apply(model,s,{id:'outside',t:1,u:-1,v:3,amount:800,kind:'deposit'});assert.strictEqual(s.memory.filter(x=>x.some(v=>v!==0)).length,1);assert.strictEqual(s.payments.length,0);assert(s.incidents.every(x=>!x.length));
pass('External funding updates only the observed account and invents no external path');

const full=runner.seek(data.events.length),first=full.state.decisions.slice(0,full.state.warmup),rest=full.state.decisions.slice(full.state.warmup);
assert(first.every(r=>r.decision==='LEARNING'&&r.settled));assert.strictEqual(first.at(-1).tauAfter,core.quantile(first.map(r=>r.score),.98));
let tau=first.at(-1).tauAfter;for(const r of rest){assert.strictEqual(r.tauBefore,tau);tau+=.025*((r.decision==='BLOCK'?1:0)-.02);assert(Math.abs(r.tauAfter-tau)<1e-12);}
pass('Threshold initialization and every subsequent update use unlabeled historical scores');
const rewind=runner.seek(240),fresh=core.replay(model,data,240,replayOptions);assert.deepStrictEqual(rewind.state,fresh.state);
pass('Checkpoint rewind exactly matches fresh chronological replay');

const late=sc.build('relay','small',42,4320),later=core.replay(model,late),earlier=core.replay(model,data);
assert.deepStrictEqual(earlier.state.decisions,later.state.decisions);
pass('Changing delayed-report arrival times cannot change any unsupervised decision');
const renamed=copy(data);renamed.truth=Object.fromEntries(Object.keys(data.truth).map(k=>[k,!data.truth[k]]));renamed.accounts.forEach(a=>{a.name='Changed';a.role='fraud';});
assert.deepStrictEqual(core.replay(model,renamed,270).state.decisions,core.replay(model,data,270).state.decisions);
pass('Generated outcome labels and account display roles do not enter scoring or calibration');
const future=copy(data);future.events.slice(280).forEach(e=>e.amount+=15000);
assert.deepStrictEqual(core.replay(model,future,280).state.decisions,core.replay(model,data,280).state.decisions);
pass('Future payment changes leave all preceding decisions unchanged');

const summaries=[];fs.mkdirSync(path.join(__dirname,'../../datasets'),{recursive:true});
for(const size of Object.keys(sc.sizes))for(const item of sc.catalog)for(const seed of [42,314]){
  const d=sc.build(item.id,size,seed),r=core.replay(model,d),all=r.state.decisions,eligible=all.filter(x=>x.decision!=='LEARNING');
  assert(d.events.every((e,i)=>i===0||e.t>=d.events[i-1].t));assert.strictEqual(new Set(d.events.map(e=>e.id)).size,d.events.length);
  assert(all.every(x=>Number.isFinite(x.score)&&x.score>=0));assert(eligible.every(x=>Number.isFinite(x.tauBefore)));
  assert.strictEqual(r.state.payments.length,all.filter(x=>x.settled).length);
  const tp=eligible.filter(x=>x.decision==='BLOCK'&&d.truth[x.event.id]).length,fn=eligible.filter(x=>x.decision==='ALLOW'&&d.truth[x.event.id]).length,fp=eligible.filter(x=>x.decision==='BLOCK'&&!d.truth[x.event.id]).length,tn=eligible.length-tp-fn-fp;
  summaries.push({scenario:item.id,size,seed,accounts:d.accounts.length,events:d.events.length,requests:all.length,calibration:all.length-eligible.length,tp,fn,fp,tn,final_tau:r.state.tau});
  if((size==='medium'||size==='large')&&seed===42||size==='large'&&item.id==='mixed'&&seed===314)fs.writeFileSync(path.join(__dirname,'../../datasets',item.id+'-'+size+'-'+seed+'.json'),JSON.stringify(d,null,2));
  console.log('DATA',item.id,size,seed,'requests',all.length,'TP/FN/FP',tp,fn,fp);
}
assert(summaries.some(x=>x.tp>0));assert(summaries.some(x=>x.fn>0));assert(summaries.some(x=>x.fp>0));
pass('All 30 combinations of scenario, scale, and seed produce valid causal decisions, including detections and mistakes');
const result={parity_max_absolute_error:maxError,checks,datasets:summaries,notice:'Synthetic generated outcomes used only after inference. No tuning was done to maximize these fraud metrics. Block counts exclude calibration.'};
fs.writeFileSync(path.join(__dirname,'../../test-results.json'),JSON.stringify(result,null,2));
const keys=Object.keys(summaries[0]);fs.writeFileSync(path.join(__dirname,'../../scenario-results.csv'),keys.join(',')+'\n'+summaries.map(r=>keys.map(k=>r[k]).join(',')).join('\n')+'\n');
console.log('All logic checks passed. Parity max error:',maxError);
