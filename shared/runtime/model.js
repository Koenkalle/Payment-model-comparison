/* Shared chronological runner and decision policy; neural code lives in adapters. */
(function(global){
  'use strict';
  const adapters=typeof module!=='undefined'?require('./model-adapters'):global.FraudAdapters;
  const policy=typeof module!=='undefined'?require('./policy'):global.FraudPolicy;
  const zero=n=>Array(n).fill(0),matrix=(n,m)=>Array.from({length:n},()=>zero(m));
  const clone=x=>JSON.parse(JSON.stringify(x));
  const trainingMode=(model,mode)=>mode||model.training_mode||'unsupervised';
  const policyState=(model,mode,options)=>({...policy.initialize(model,mode,options),eta:options.eta??.025,warmup:Math.max(1,Math.floor(options.warmup??model.policy?.default_warmup_requests??128)),mode:options.mode||'enforce'});
  const owner=(model,mode)=> (model.id||'gru_attention')+'@'+(model.checkpoint_id||'v2')+':'+trainingMode(model,mode)+':'+JSON.stringify(model.architecture||null);
  function assertOwner(model,s){if(s.modelOwner!==owner(model,s.trainingMode))throw Error('Model/state mismatch. Replay history with the selected checkpoint and training mode.');}
  function initial(model,n,options={}){
    const selectedTrainingMode=trainingMode(model,options.trainingMode),adapter=adapters.get(model);return {modelOwner:owner(model,selectedTrainingMode),trainingMode:selectedTrainingMode,memory:adapter.initialize(model,n),adapterState:typeof adapter.initializeState==='function'?adapter.initializeState(model,n):null,last:zero(n),seen:zero(n),outCount:zero(n),inCount:zero(n),outValue:zero(n),inValue:zero(n),pairs:matrix(n,n),incidents:Array.from({length:n},()=>[]),payments:[],reports:[],events:[],decisions:[],now:0,time:0,ids:{},calibration:[],...policyState(model,selectedTrainingMode,options)};
  }
  function validate(s,e){
    const n=s.memory.length;
    if(!Number.isFinite(e.t)||e.t<s.time)throw Error('Events must arrive in chronological order.');
    if(!Number.isFinite(e.amount)||e.amount<0)throw Error('Invalid amount.');
    if(!Number.isInteger(e.v)||e.v<0||e.v>=n||!Number.isInteger(e.u)||e.u>=n||e.u< -1||e.u===e.v)throw Error('Invalid account.');
    if(!['payment','deposit','report'].includes(e.kind)||e.kind==='payment'&&e.u<0||e.kind!=='payment'&&e.u!==-1)throw Error('Invalid event type.');
    if(e.id&&s.ids[e.id])throw Error('Duplicate event.');
  }
  function apply(model,s,e,settled=e.settled!==false){
    assertOwner(model,s);validate(s,e);s.time=e.t;if(e.id)s.ids[e.id]=true;
    if(e.kind==='report'){s.reports.push({...e});s.events.push({...e});return s;}
    const adapter=adapters.get(model);s.memory=adapter.update(model,s,e,settled);
    if(typeof adapter.updateState==='function')s.adapterState=adapter.updateState(model,s,e,settled)||s.adapterState;
    for(const n of(e.u>=0?[e.v,e.u]:[e.v])){s.last[n]=e.t;s.seen[n]++;}s.now=e.t;
    if(e.kind==='payment'&&settled){
      s.incidents[e.u].push({other:e.v,t:e.t,amount:e.amount,role:-1,id:e.id});s.incidents[e.v].push({other:e.u,t:e.t,amount:e.amount,role:1,id:e.id});
      s.payments.push({...e});s.outCount[e.u]++;s.inCount[e.v]++;s.outValue[e.u]+=e.amount;s.inValue[e.v]+=e.amount;s.pairs[e.u][e.v]++;
    }
    s.events.push(e.kind==='payment'?{...e,settled}:{...e});return s;
  }
  function readout(model,s){
    assertOwner(model,s);return adapters.get(model).readout(model,s);
  }
  function score(model,s,e){
    assertOwner(model,s);if(e.kind!=='payment')return null;validate(s,e);
    const prediction=adapters.get(model).predict(model,s,e);
    if(!prediction||!Number.isFinite(prediction.score)||prediction.score<0)throw Error('Adapter must return a finite, nonnegative surprise score in bits.');
    return prediction;
  }
  function quantile(values,q){const a=values.slice().sort((a,b)=>a-b);return a[Math.min(a.length-1,Math.max(0,Math.ceil(q*a.length)-1))];}
  function decision(s,e,result){
    if(!result)return {event:e,score:null,tauBefore:s.tau,tauAfter:s.tau,decision:null};
    const learning=s.tau===null,block=!learning&&result.score>s.tau;
    return {event:e,score:result.score,parts:result.parts,buckets:result.buckets,gap:result.gap,memory:result.memory,embedding:result.embedding,previousPairCount:result.previousPairCount,tauBefore:s.tau,decision:learning?'LEARNING':block?'BLOCK':'ALLOW',settled:s.mode==='shadow'||!block};
  }
  function finishDecision(s,record){
    if(record.score!==null){
      if(s.decisionPolicy==='shared'){
        if(record.decision==='LEARNING'){s.calibration.push(record.score);if(s.calibration.length>=s.warmup)s.tau=quantile(s.calibration,1-s.alpha);}
        else s.tau+=s.eta*((record.decision==='BLOCK'?1:0)-s.alpha);
      }
      record.tauAfter=s.tau;s.decisions.push(record);
    }
    return record;
  }
  function step(model,s,e,prediction){
    const result=prediction===undefined?score(model,s,e):prediction,record=decision(s,e,result);
    // Scoring precedes both settlement and threshold updates.
    apply(model,s,e,result?record.settled:e.settled!==false);
    return finishDecision(s,record);
  }
  class Runner{
    constructor(model,data,options={}){this.model=model;this.data=data;this.options={...options};this.state=initial(model,data.accounts.length,options);this.position=0;this.trace=[];this.predictions=new Map();this.inferenceCalls=0;this.snapshots=new Map([[0,this.snapshot()]]);}
    snapshot(){return clone({...this.state,decisions:[]});}
    restorePolicy(){
      Object.assign(this.state,policyState(this.model,this.state.trainingMode,this.options));
      this.state.decisions=this.trace.slice(0,this.position).filter(r=>r.score!==null);
      this.state.calibration=this.state.decisionPolicy==='shared'?this.state.decisions.slice(0,this.state.warmup).map(r=>r.score):[];
      if(this.position)this.state.tau=this.trace[this.position-1].tauAfter;
    }
    reconfigure(options){
      if(this.state.mode!=='shadow'||(options.mode||'shadow')!=='shadow'||trainingMode(this.model,options.trainingMode)!==this.state.trainingMode)throw Error('Only same-history policy changes can reuse model state.');
      this.options={...options};const s={...policyState(this.model,this.state.trainingMode,options),calibration:[],decisions:[]};
      this.trace=this.trace.map(r=>finishDecision(s,decision(s,r.event,r.score===null?null:r)));
      this.restorePolicy();return this;
    }
    prepare(count){
      count=Math.max(0,Math.min(this.data.events.length,Math.floor(count)));
      if(count<this.position){const at=Math.max(...Array.from(this.snapshots.keys()).filter(x=>x<=count));this.state=clone(this.snapshots.get(at));this.position=at;this.restorePolicy();}
      return count;
    }
    preview(){
      const e=this.data.events[this.position];if(e?.kind!=='payment')return null;
      if(!this.predictions.has(this.position)){
        this.inferenceCalls++;this.predictions.set(this.position,score(this.model,this.state,e));
        if(this.predictions.size>8)this.predictions.delete(this.predictions.keys().next().value);
      }
      return this.predictions.get(this.position);
    }
    advance(){
      const i=this.position,e=this.data.events[i],cached=this.trace[i];
      const prediction=e.kind!=='payment'?null:this.predictions.get(i)||(cached&&cached.score!==null?cached:this.preview());
      this.trace[i]=step(this.model,this.state,e,prediction);this.position++;
      if(this.position%160===0&&!this.snapshots.has(this.position)){
        this.snapshots.set(this.position,this.snapshot());
        if(this.snapshots.size>4)this.snapshots.delete(Array.from(this.snapshots.keys()).filter(k=>k!==0).sort((a,b)=>a-b)[0]);
      }
    }
    result(){const count=this.position;return {state:this.state,trace:this.trace.slice(0,count),position:count,last:count?this.trace[count-1]:null};}
    seek(count){
      count=this.prepare(count);while(this.position<count)this.advance();return this.result();
    }
  }
  function replay(model,data,count=data.events.length,options={}){return new Runner(model,data,options).seek(count);}
  // Reapply a policy to completed, observed-history predictions without inference.
  // This deliberately uses the same decision and cutoff update as live scoring.
  function evaluatePolicy(model,records,options={}){
    if(options.mode&&options.mode!=='shadow')throw Error('Cached predictions require observed-history execution.');
    const s={...policyState(model,trainingMode(model,options.trainingMode),{...options,mode:'shadow'}),calibration:[],decisions:[]};
    for(const record of records)finishDecision(s,decision(s,record.event,record));
    return {records:s.decisions,tau:s.tau,policyFit:s.policyFit};
  }
  global.FraudCore={initial,apply,readout,score,step,quantile,Runner,replay,evaluatePolicy};
  if(typeof module!=='undefined')module.exports=global.FraudCore;
})(typeof globalThis!=='undefined'?globalThis:window);
