'use strict';
const assert=require('assert'),fs=require('fs'),core=require('../../shared/runtime/model'),{Comparison,ComparisonCache}=require('../../tools/comparison/comparison'),sc=require('../../shared/runtime/scenarios');
const models=require('../../model-bundle.json').models,copy=x=>JSON.parse(JSON.stringify(x)),calls=g=>[...g.runners.values()].reduce((s,r)=>s+r.inferenceCalls,0);
async function run(){
  const data=sc.build('mixed','small',2718),pool=new ComparisonCache(models),target=190;
  for(const trainingMode of['unsupervised','supervised']){
    const original={mode:'shadow',trainingMode,alpha:.02,warmup:32},group=pool.acquire(data,original);
    await group.seekAsync(target);const count=calls(group),memory=[...group.runners.values()].map(r=>copy(r.state.memory));
    for(const extra of[{alpha:.05,warmup:64},{decisionPolicy:'tuned',missedFraudCost:100},{decisionPolicy:'tuned',missedFraudCost:5,alpha:.01},{alpha:.01,warmup:32}]){
      const options={...original,...extra};assert.strictEqual(pool.acquire(data,options),group);
      const rows=await group.seekAsync(target);assert.strictEqual(calls(group),count,'Policy-only change reran inference');
      for(const [i,row]of rows.entries()){
        assert.deepStrictEqual(row.result.state,core.replay(row.model,data,target,options).state);
        assert.deepStrictEqual(row.result.state.memory,memory[i]);
      }
    }
    for(const position of[171,29,190]){
      const rows=await group.seekAsync(position);
      rows.forEach(row=>assert.deepStrictEqual(row.result.state,core.replay(row.model,data,position,group.options).state));
    }
  }
  console.log('PASS All 18 model/mode combinations: policy changes use zero new inferences and preserve exact states, decisions and rewind.');

  const a={mode:'enforce',trainingMode:'unsupervised',alpha:.01,warmup:32},b={...a,alpha:.05};
  const first=pool.acquire(data,a);await first.seekAsync(target);
  const second=pool.acquire(data,b);assert.strictEqual(first,second);
  const changed=await second.seekAsync(target);
  changed.forEach(row=>assert.deepStrictEqual(row.result.state,core.replay(row.model,data,target,b).state));
  assert.strictEqual(pool.acquire(data,a),first);
  const fits={...a,decisionPolicy:'tuned',missedFraudCost:20},tuned=pool.acquire(data,fits);await tuned.seekAsync(target);const before=calls(tuned);
  assert.strictEqual(pool.acquire(data,{...fits,alpha:.05,warmup:512}),tuned);await tuned.seekAsync(target);assert.strictEqual(calls(tuned),before);
  assert(pool.entries.size<=3);
  console.log('PASS Changed blocking policies retain separate histories; returning to a cached setup and changing ranking-only settings avoids inference.');

  const cancellable=new Comparison(models,data),old=cancellable.seekAsync(400),latest=cancellable.seekAsync(21);
  assert.strictEqual(await old,null);const fresh=await latest;assert(fresh.every(r=>r.result.position===21));
  let cancelled=false,heartbeat=0;const task=cancellable.seekAsync(400,{cancelled:()=>cancelled,budgetMs:6});
  setTimeout(()=>{heartbeat++;cancelled=true;},5);assert.strictEqual(await task,null);assert.strictEqual(heartbeat,1);
  const completed=await cancellable.seekAsync(43);completed.forEach(row=>assert.deepStrictEqual(row.result.state,core.replay(row.model,data,43,{mode:'shadow',trainingMode:'unsupervised'}).state));
  console.log('PASS Timers run during replay, stale requests cancel, and interrupted comparisons resume correctly.');

  const long=sc.build('mixed','medium',314),bounded=new core.Runner(models[0],long,{mode:'shadow'});bounded.seek(long.events.length);
  assert(bounded.snapshots.size<=4&&bounded.predictions.size<=8);
  for(const i of[20,650,long.events.length])assert.deepStrictEqual(bounded.seek(i).state,core.replay(models[0],long,i,{mode:'shadow'}).state);
  console.log('PASS Snapshot, prediction and comparison caches stay bounded and eviction preserves replay correctness.');
}
run().catch(error=>{console.error(error);process.exitCode=1;});
