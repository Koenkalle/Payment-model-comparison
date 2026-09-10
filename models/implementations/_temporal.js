/* Fixed-projection history prototypes. These are not trained DyGFormer/TAMI encoders. */
(function(global){
'use strict';
const h=typeof module!=='undefined'?require('./_math'):global.FraudModelMath;
const {zero,matrix,dot,add,linear,sigmoid,softmax,bin,architecture}=h;
const xgb=typeof module!=='undefined'?require('./xgboost_numpy_core'):global.FraudXGBoost;
const xgbFeatures=xgb.features;const GAP=[1,5,20,60,180,720,2880];
  const AMOUNT_PLACEHOLDER=[15,35,75,150,300,650,1500,4000,10000];
  const sequenceConfig=model=>model.sequence||{},sequenceEvents=(s,n,limit)=>s.incidents[n].slice(-limit).reverse();
  function sequenceHistoryVector(model,s,n,t,cache){
    const key='history:'+n;if(cache&&cache[key])return cache[key];
    const cfg=sequenceConfig(model),dim=cfg.dim||8,W=cfg.history_weights||[],out=zero(dim),events=sequenceEvents(s,n,cfg.history_limit||8);let total=0;
    for(const e of events){const age=Math.max(0,t-e.t),weight=Math.exp(-age/(cfg.history_decay||720)),basis=[1,e.role,Math.log1p(e.amount)/8,Math.min(age/720,4),Math.sin(age/60),Math.cos(age/60),Math.sin(age/1440),Math.cos(age/1440)];for(let j=0;j<dim;j++)out[j]+=weight*dot(W[j]||zero(8),basis);total+=weight;}
    const result=total?out.map(x=>Math.tanh(x/total)):out;if(cache)cache[key]=result;return result;
  }
  function sequencePairEvents(model,s,u,v){
    const cfg=sequenceConfig(model),stored=s.adapterState?.pairHistory?.[u]?.[v];
    if(stored&&stored.length)return stored.slice(-(cfg.pair_limit||6)).reverse();
    return s.incidents[u].filter(e=>e.other===v&&e.role===-1).slice(-(cfg.pair_limit||6)).reverse();
  }
  function sequencePairVector(model,s,u,v,t){
    const cfg=sequenceConfig(model),dim=cfg.pair_dim||4,W=cfg.pair_weights||[],out=zero(dim),events=sequencePairEvents(model,s,u,v);let total=0;
    for(const e of events){const age=Math.max(0,t-e.t),weight=Math.exp(-age/(cfg.pair_decay||1440)),basis=[1,Math.log1p(e.amount)/8,Math.min(age/720,4),Math.sin(age/60),Math.cos(age/60)];for(let j=0;j<dim;j++)out[j]+=weight*dot(W[j]||zero(5),basis);total+=weight;}
    return total?out.map(x=>Math.tanh(x/total)):out;
  }
  function sequenceNeighbors(model,s,n,t){
    const cfg=sequenceConfig(model),seen=new Set(),out=[];
    for(const e of sequenceEvents(s,n,cfg.neighbor_limit||4))if(!seen.has(e.other)){seen.add(e.other);out.push(e);}
    return out;
  }
  function sequenceGraphVector(model,s,n,t,depth,cache){
    const cfg=sequenceConfig(model),key=n+':'+depth;if(cache[key])return cache[key];const base=sequenceHistoryVector(model,s,n,t,cache);if(!cfg.use_gnn||depth<=0)return cache[key]=base;
    const out=base.slice(),W=cfg.gnn_weights||[],dim=cfg.dim||8;let total=0;
    for(const e of sequenceNeighbors(model,s,n,t)){const child=sequenceGraphVector(model,s,e.other,t,depth-1,cache),age=Math.max(0,t-e.t),edge=[e.role,Math.log1p(e.amount)/8,Math.min(age/720,4),1],value=(W||[]).map(row=>Math.tanh(dot(row,child.concat(edge))));const weight=Math.exp(-age/(cfg.history_decay||720));for(let j=0;j<dim;j++)out[j]+=weight*(value[j]||0);total+=weight;}
    if(!total)return cache[key]=base;
    return cache[key]=out.map((x,j)=>Math.tanh(base[j]+(x-base[j])/total));
  }
  function sequenceAccountEmbedding(model,s,n,t,cache){const cfg=sequenceConfig(model);return cfg.use_gnn?sequenceGraphVector(model,s,n,t,cfg.gnn_layers||2,cache):sequenceHistoryVector(model,s,n,t,cache);}
  function sequenceAmountLogits(model,s,u,v,t){
    const cfg=sequenceConfig(model),edges=model.amount_bins||AMOUNT_PLACEHOLDER,hist=zero(edges.length+1);
    for(const e of s.incidents[u])if(e.role===-1)hist[bin(e.amount,edges)]+=Math.exp(-Math.max(0,t-e.t)/(cfg.history_decay||720));
    if(cfg.use_pair)for(const e of sequencePairEvents(model,s,u,v))hist[bin(e.amount,edges)]+=(cfg.pair_amount_weight||1.0)*Math.exp(-Math.max(0,t-e.t)/(cfg.pair_decay||1440));
    return hist.map(x=>Math.log(x+(cfg.histogram_smoothing||.5)));
  }
  function sequenceGapLogits(model,s,u,v,t){
    const cfg=sequenceConfig(model),edges=model.gap_bins||[1,5,20,60,180,720,2880],hist=zero(edges.length+2),out=s.incidents[u].filter(e=>e.role===-1).sort((a,b)=>a.t-b.t);let last=null;
    for(const e of out){if(last!==null)hist[bin(Math.max(0,e.t-last),edges)]+=Math.exp(-Math.max(0,t-e.t)/(cfg.history_decay||720));last=e.t;}
    if(cfg.use_pair){const pair=sequencePairEvents(model,s,u,v).slice().sort((a,b)=>a.t-b.t);last=null;for(const e of pair){if(last!==null)hist[bin(Math.max(0,e.t-last),edges)]+=(cfg.pair_gap_weight||1.0)*Math.exp(-Math.max(0,t-e.t)/(cfg.pair_decay||1440));last=e.t;}}
    return hist.map(x=>Math.log(x+(cfg.histogram_smoothing||.5)));
  }
  function sequencePrediction(model,s,e){
    const cfg=sequenceConfig(model),n=s.memory.length,u=e.u,v=e.v,t=e.t,cache={},sender=sequenceAccountEmbedding(model,s,u,t,cache),historySender=sequenceHistoryVector(model,s,u,t,cache),recipientLogits=Array(n).fill(-1e9),D=Math.max(1,sender.length);
    for(let candidate=0;candidate<n;candidate++)if(candidate!==u){const candidateEmbedding=sequenceAccountEmbedding(model,s,candidate,t,cache),historyCandidate=sequenceHistoryVector(model,s,candidate,t,cache),pair=cfg.use_pair?sequencePairVector(model,s,u,candidate,t):[],temporalSimilarity=dot(historySender,historyCandidate)/D,graphSimilarity=cfg.use_gnn?dot(sender,candidateEmbedding)/D:0;recipientLogits[candidate]=(cfg.recipient_pair_weight||0)*Math.log1p(s.pairs[u][candidate])+(cfg.recipient_activity_weight||0)*Math.log1p(s.inCount[candidate])+(cfg.recipient_temporal_weight||0)*temporalSimilarity+(cfg.recipient_pair_state_weight||0)*(pair[0]||0)+(cfg.recipient_graph_weight||0)*graphSimilarity;}
    const amountLogits=sequenceAmountLogits(model,s,u,v,t),gapLogits=sequenceGapLogits(model,s,u,v,t),probabilities=[softmax(recipientLogits),softmax(amountLogits),softmax(gapLogits)],gap=s.seen[u]?Math.max(0,t-s.last[u]):null,buckets=[v,bin(e.amount,model.amount_bins||AMOUNT_PLACEHOLDER),gap===null?(model.gap_bins||GAP).length+1:bin(gap,model.gap_bins||GAP)],parts=probabilities.map((p,i)=>-Math.log2(Math.max(1e-30,p[buckets[i]]))),baseScore=parts.reduce((a,b)=>a+b,0),common={gap,embedding:[sender,sequenceAccountEmbedding(model,s,v,t,cache)],memory:[[],[]],previousPairCount:s.pairs[u][v]};
    if(s.trainingMode==='supervised'&&model.supervised){const row=xgbFeatures(model,s,e).concat([baseScore]),head=model.supervised,scaled=row.map((value,i)=>(value-head.mean[i])/Math.max(1e-12,head.scale[i])),probability=sigmoid(head.bias+scaled.reduce((sum,value,i)=>sum+value*head.weights[i],0)),score=-Math.log2(Math.max(1e-12,1-probability));return {...common,score,parts:[score],probabilities:[[1-probability,probability],[1],[1]],buckets:[probability],probability,features:row,trainingMode:'supervised'};}
    return {...common,score:baseScore,parts,probabilities,buckets,features:xgbFeatures(model,s,e),trainingMode:'unsupervised'};
  }
  const temporalFamily={
    initialize(model,n){return matrix(n,0);},
    initializeState(model,n){const cfg=sequenceConfig(model);return {pairHistory:cfg.use_pair?Array.from({length:n},()=>Array.from({length:n},()=>[])):null};},
    update(model,s){return s.memory;},
    updateState(model,s,e,settled){const cfg=sequenceConfig(model),history=s.adapterState?.pairHistory;if(!cfg.use_pair||!history||e.kind!=='payment'||!settled||e.u<0)return s.adapterState;if(!history[e.u][e.v])history[e.u][e.v]=[];history[e.u][e.v].push({t:e.t,amount:e.amount,id:e.id});const limit=cfg.pair_limit||6;if(history[e.u][e.v].length>limit)history[e.u][e.v].splice(0,history[e.u][e.v].length-limit);return s.adapterState;},
    readout(model,s){const cfg=sequenceConfig(model),cache={},embedding=Array.from({length:s.memory.length},(_,n)=>sequenceAccountEmbedding(model,s,n,s.now,cache));return {embedding,slots:[],attention:[]};},
    predict(model,s,e){return sequencePrediction(model,s,e);}
  };

 global.FraudTemporalPrototype=temporalFamily;
 if(typeof module!=='undefined')module.exports=temporalFamily;
})(globalThis);
