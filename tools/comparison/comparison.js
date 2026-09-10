/* Shared request stream; independent model state and threshold policies. */
(function(global){
  'use strict';
  const core=typeof module!=='undefined'?require('../../shared/runtime/model'):global.FraudCore;
  const {metrics}=typeof module!=='undefined'?require('../../shared/runtime/metrics'):global.FraudMetrics;
  const normalized=options=>({...options,modelPolicies:Object.fromEntries(Object.entries(options.modelPolicies||{}).map(([id,value])=>[id,{...value}])),mode:options.mode||'shadow',trainingMode:options.trainingMode||'unsupervised'});
  function modelOptions(model,options){const {modelPolicies,...rest}=options;return {...rest,...modelPolicies?.[model.id],mode:options.mode,trainingMode:options.trainingMode};}
  const policyKey=o=>JSON.stringify(o.decisionPolicy==='tuned'?['tuned',o.falseBlockCost??1,o.missedFraudCost??20]:o.decisionPolicy==='auto'?['auto',o.objective||'f1']:o.decisionPolicy==='manual'?['manual',o.manualTau]:['shared',o.alpha??.02,o.warmup??null,o.eta??.025]);
  const now=()=>global.performance?.now?global.performance.now():Date.now();
  const yieldTask=()=>new Promise(resolve=>setTimeout(resolve,0));
  class Comparison{
    constructor(models,data,options={}){
      if(!models.length||new Set(models.map(m=>m.id)).size!==models.length)throw Error('Comparison requires distinct model IDs.');
      const target=m=>JSON.stringify([m.amount_bins,m.gap_bins]);
      if(models.some(m=>target(m)!==target(models[0])))throw Error('Models must use the same prediction targets and bins.');
      this.models=models;this.data=data;this.options=normalized(options);
      this.runners=new Map(models.map(m=>[m.id,new core.Runner(m,data,modelOptions(m,this.options))]));
      this.runnerVariants=new Map(models.map(m=>[m.id,new Map([[policyKey(modelOptions(m,this.options)),this.runners.get(m.id)]])]));this.position=0;this.revision=0;
    }
    cancel(){this.revision++;}
    reconfigure(options){
      const next=normalized(options);this.cancel();
      if(next.trainingMode!==this.options.trainingMode||next.mode!==this.options.mode)throw Error('A different history requires a separate comparison.');
      for(const model of this.models){
        const before=modelOptions(model,this.options),after=modelOptions(model,next);
        if(JSON.stringify(before)===JSON.stringify(after))continue;
        let runner=this.runners.get(model.id);const samePolicy=policyKey(before)===policyKey(after);
        if(samePolicy){runner.options={...after};runner.restorePolicy();}
        else if(next.mode==='shadow')runner.reconfigure(after);
        else{
          const variants=this.runnerVariants.get(model.id),key=policyKey(after);
          runner=variants.get(key)||new core.Runner(model,this.data,after);
          runner.options={...after};runner.restorePolicy();variants.delete(key);variants.set(key,runner);
          if(variants.size>2)variants.delete(variants.keys().next().value);
          this.runners.set(model.id,runner);
        }
      }
      this.options=next;
      return this;
    }
    entry(model){
      const runner=this.runners.get(model.id),result=runner.result(),prediction=runner.preview(),tau=result.state.tau;
      return {model,result,prediction,decision:prediction?(tau===null?'LEARNING':prediction.score>tau?'BLOCK':'ALLOW'):null};
    }
    seek(count){
      this.cancel();
      this.position=Math.max(0,Math.min(this.data.events.length,Math.floor(count)));
      return this.models.map(model=>{
        this.runners.get(model.id).seek(this.position);return this.entry(model);
      });
    }
    async seekAsync(count,{cancelled=()=>false,onProgress=()=>{},budgetMs=8,priorityModel,yieldControl=yieldTask}={}){
      const revision=++this.revision,target=Math.max(0,Math.min(this.data.events.length,Math.floor(count))),aborted=()=>revision!==this.revision||cancelled();
      const order=this.models.slice().sort((a,b)=>(b.id===priorityModel?1:0)-(a.id===priorityModel?1:0)),results=new Map();
      let sliceStart=now(),lastProgress=0,maxSliceMs=0,yields=0;
      const pause=async(model,done)=>{
        const time=now();maxSliceMs=Math.max(maxSliceMs,time-sliceStart);
        if(time-lastProgress>80){onProgress({model:model?.label,done,total:this.models.length,target});lastProgress=time;}
        yields++;await yieldControl();sliceStart=now();
      };
      if(order.some(m=>this.runners.get(m.id).position!==target))await pause(null,0);
      for(const model of order){
        if(aborted())return null;
        const runner=this.runners.get(model.id);runner.prepare(target);
        while(runner.position<target){
          runner.advance();
          if(now()-sliceStart>=budgetMs){await pause(model,results.size);if(aborted())return null;}
        }
        results.set(model.id,this.entry(model));
        if(now()-sliceStart>=budgetMs){await pause(model,results.size);if(aborted())return null;}
      }
      if(aborted())return null;
      this.position=target;this.lastTiming={yields,maxSliceMs:Math.max(maxSliceMs,now()-sliceStart)};
      return this.models.map(model=>results.get(model.id));
    }
  }
  // Keep a few recent comparisons, bounded independently of slider changes.
  // Only shadow histories may share inference across different policies.
  class ComparisonCache{
    constructor(models,limit=3){this.models=models;this.limit=limit;this.entries=new Map();this.dataIds=new WeakMap();this.nextDataId=0;}
    acquire(data,options={}){
      if(!this.dataIds.has(data))this.dataIds.set(data,++this.nextDataId);
      const o=normalized(options),key=JSON.stringify([this.dataIds.get(data),o.trainingMode,o.mode]);
      let group=this.entries.get(key);
      if(group){this.entries.delete(key);group.reconfigure(o);}else group=new Comparison(this.models,data,o);
      this.entries.set(key,group);
      if(this.entries.size>this.limit){const oldest=this.entries.keys().next().value;this.entries.get(oldest).cancel();this.entries.delete(oldest);}
      return group;
    }
  }
  global.FraudComparison={Comparison,ComparisonCache,metrics};
  if(typeof module!=='undefined')module.exports=global.FraudComparison;
})(typeof globalThis!=='undefined'?globalThis:window);
