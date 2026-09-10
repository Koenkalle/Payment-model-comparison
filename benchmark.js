/* Run after correctness checks, without other benchmark jobs competing for CPU. */
'use strict';
const fs=require('fs'),{performance}=require('perf_hooks'),{ComparisonCache}=require('./comparison'),sc=require('./scenarios'),models=require('./model-bundle.json').models;
const median=a=>a.slice().sort((a,b)=>a-b)[Math.floor(a.length/2)];
async function main(){
  const data=sc.build('mixed','medium',42),pool=new ComparisonCache(models),base={mode:'shadow',trainingMode:'unsupervised'},start=performance.now(),group=pool.acquire(data,base);
  let pulses=0;const timer=setInterval(()=>pulses++,10);await group.seekAsync(data.startIndex);clearInterval(timer);
  const cold={milliseconds:performance.now()-start,heartbeatCallbacks:pulses,...group.lastTiming},runs=[];
  const calls=()=>[...group.runners.values()].reduce((s,r)=>s+r.inferenceCalls,0),before=calls();
  for(const cost of[5,20,100,1,20]){const t=performance.now();pool.acquire(data,{...base,decisionPolicy:'tuned',missedFraudCost:cost});await group.seekAsync(data.startIndex);runs.push(performance.now()-t);}
  const nowCalls=calls(),t=performance.now();for(let i=0;i<20;i++)await group.seekAsync(data.startIndex);
  const result={scenario:'mixed',size:'medium',seed:42,accounts:data.accounts.length,events:data.events.length,position:data.startIndex,coldReplay:cold,policyChanges:{milliseconds:runs,medianMilliseconds:median(runs),additionalInferences:nowCalls-before},cachedInspectionMillisecondsPerCall:(performance.now()-t)/20,notice:'Local Node.js replay timings, excluding DOM rendering; device and browser timings vary. Slices target 8 ms between yields, with an event or checkpoint as the indivisible unit.'};
  fs.writeFileSync(__dirname+'/performance-results.json',JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
}
main().catch(error=>{console.error(error);process.exitCode=1;});
