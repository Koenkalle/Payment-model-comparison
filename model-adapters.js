/* Model boundary: initialize, update, readout, predict. No decision policy here. */
(function(global){
  'use strict';
  const adapters=new Map(),zero=n=>Array(n).fill(0),matrix=(n,m)=>Array.from({length:n},()=>zero(m));
  const dot=(a,b)=>{let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s;},add=(a,b)=>a.map((x,i)=>x+b[i]);
  function linear(a,w,b){const out=new Array(w[0].length);for(let j=0;j<out.length;j++){let sum=b?b[j]:0;for(let i=0;i<a.length;i++)sum+=a[i]*w[i][j];out[j]=sum;}return out;}
  const sigmoid=x=>1/(1+Math.exp(-Math.max(-50,Math.min(50,x))));
  function softmax(a){const m=Math.max(...a),e=a.map(x=>Math.exp(x-m)),s=e.reduce((a,b)=>a+b,0);return e.map(x=>x/s);}
  const bin=(x,edges)=>edges.reduce((s,v)=>s+(x>=v),0);
  const architecture=m=>m.architecture||{memory:'gru',readout:'attention',layers:2};
  function register(id,adapter){
    if(adapters.has(id))throw Error('Adapter already registered: '+id);
    for(const k of['initialize','update','readout','predict'])if(typeof adapter[k]!=='function')throw Error('Adapter '+id+' requires '+k+'().');
    adapters.set(id,Object.freeze(adapter));
  }
  function get(model){const key=model.adapter||'categorical_graph';if(!adapters.has(key))throw Error('Unknown model adapter: '+key);return adapters.get(key);}
  const featureLog=(x,scale=1)=>Math.log1p(Math.max(0,Number(x)||0))/scale;
  function xgbRecent(s,n,role,t){const events=s.incidents[n].filter(x=>t-x.t<=60&&x.role===role);return [events.length,events.reduce((sum,x)=>sum+x.amount,0)];}
  function xgbFeatures(model,s,e){
    const u=e.u,v=e.v,t=e.t,amount=e.amount,si=xgbRecent(s,u,1,t),so=xgbRecent(s,u,-1,t),ri=xgbRecent(s,v,1,t),ro=xgbRecent(s,v,-1,t);
    const senderMean=s.outValue[u]/Math.max(1,s.outCount[u]),recipientMean=s.inValue[v]/Math.max(1,s.inCount[v]);
    const gapU=s.seen[u]?Math.max(0,t-s.last[u]):0,gapV=s.seen[v]?Math.max(0,t-s.last[v]):0;
    const amountBins=model.amount_bins||AMOUNT_PLACEHOLDER;
    return [featureLog(amount,8),bin(amount,amountBins)/amountBins.length,
      featureLog(s.outCount[u],6),featureLog(s.inCount[u],6),featureLog(s.outCount[v],6),featureLog(s.inCount[v],6),
      featureLog(s.outValue[u],10),featureLog(s.inValue[u],10),featureLog(s.outValue[v],10),featureLog(s.inValue[v],10),
      featureLog(s.seen[u],6),featureLog(s.seen[v],6),featureLog(gapU,6),featureLog(gapV,6),
      featureLog(s.pairs[u][v],3),featureLog(s.pairs[u][v]+s.pairs[v][u],3),s.pairs[u][v]+s.pairs[v][u]>0?1:0,
      featureLog(si[0],4),featureLog(so[0],4),featureLog(ri[0],4),featureLog(ro[0],4),
      featureLog(si[1],10),featureLog(so[1],10),featureLog(ri[1],10),featureLog(ro[1],10),
      featureLog(amount/Math.max(1,senderMean),8),featureLog(amount/Math.max(1,recipientMean),8)];
  }
  function treeValue(tree,row){let node=tree;while(node.leaf===undefined)node=row[node.feature]<=node.threshold?node.left:node.right;return node.leaf;}
  function xgbRaw(model,row){return model.base_score+model.learning_rate*model.trees.reduce((sum,tree)=>sum+treeValue(tree,row),0);}
  // Filled from the model checkpoint at call time; keeping this indirection
  // avoids making feature extraction depend on one checkpoint's bin edges.
  let AMOUNT_PLACEHOLDER=[15,35,75,150,300,650,1500,4000,10000];
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
  const categorical={
    initialize(model,n){return matrix(n,model.hidden);},
    update(model,s,e,settled){
      if(architecture(model).memory==='none')return s.memory;
      const p=model.weights,h=s.memory,next=h.slice(),actors=[[e.v,e.u,e.u<0?2:1]];if(e.u>=0)actors.push([e.u,e.v,0]);
      for(const[n,other,role]of actors){
        const f=[0,0,0,settled?0:1,Math.log1p(e.amount)/8,Math.log1p(Math.max(0,e.t-s.last[n]))/6];f[role]=1;
        const cp=other<0?zero(model.hidden):h[other],x=cp.concat(f),joined=x.concat(h[n]);
        const z=linear(joined,p.Wz,p.bz).map(sigmoid),r=linear(joined,p.Wr,p.br).map(sigmoid);
        const cand=linear(x.concat(h[n].map((v,j)=>v*r[j])),p.Wn,p.bn).map(Math.tanh);
        next[n]=h[n].map((v,j)=>(1-z[j])*v+z[j]*cand[j]);
      }
      return next;
    },
    readout(model,s){
      const cfg=architecture(model),p=model.weights,H=model.hidden;
      if(!cfg.layers)return {embedding:s.memory,slots:[],attention:[]};
      const slots=s.incidents.map((events,n)=>[{other:n,attrs:[0,0,0,0],id:null}].concat(events.slice(-(model.neighbors-1)).reverse().map(e=>({other:e.other,attrs:[Math.log1p(e.amount)/8,Math.log1p(Math.max(0,s.now-e.t))/6,e.role,1],id:e.id}))));
      let h=s.memory;const attention=[];
      for(let l=0;l<cfg.layers;l++){
        const isAttention=cfg.readout==='attention',queries=isAttention?h.map(x=>linear(x,p['q'+l])):null,keys=isAttention?h.map(x=>linear(x,p['k'+l])):null,values=h.map(x=>linear(x,p['v'+l])),weights=[];
        h=h.map((own,n)=>{
          const edges=slots[n].map(x=>linear(x.attrs,p['edge'+l]));
          const w=isAttention?softmax(slots[n].map((x,k)=>dot(queries[n],add(keys[x.other],edges[k]))/Math.sqrt(H))):slots[n].map(()=>1/slots[n].length);weights.push(w);
          const agg=zero(H);slots[n].forEach((x,k)=>values[x.other].forEach((v,j)=>agg[j]+=w[k]*(v+edges[k][j])));
          return add(linear(own,p['self'+l],p['out'+l]),agg).map(Math.tanh);
        });attention.push(weights);
      }
      return {embedding:h,slots,attention};
    },
    predict(model,s,e){
      const p=model.weights,z=this.readout(model,s).embedding,u=e.u,v=e.v,H=model.hidden;
      const q=H?linear(z[u],p.rq):null;
      const logits=z.map((h,n)=>n===u?-1e9:(H?dot(q,linear(h,p.rk))/Math.sqrt(H):0)+linear([Math.log1p(s.pairs[u][n])/3,Math.log1p(s.inCount[n])/6,s.pairs[u][n]+s.pairs[n][u]>0?1:0],p.pair)[0]);
      const ctx=[Math.log1p(s.outValue[u]/Math.max(1,s.outCount[u]))/8,Math.log1p(s.inValue[v]/Math.max(1,s.inCount[v]))/8,Math.log1p(s.seen[u])/6,Math.log1p(s.seen[v])/6,Math.log1p(s.pairs[u][v])/3];
      const x=z[u].concat(z[v],ctx),probabilities=[softmax(logits),softmax(linear(x,p.amount,p.amountb)),softmax(linear(x,p.gap,p.gapb))];
      const gap=s.seen[u]?Math.max(0,e.t-s.last[u]):null,buckets=[v,bin(e.amount,model.amount_bins),gap===null?model.gap_bins.length+1:bin(gap,model.gap_bins)];
      const parts=probabilities.map((prob,i)=>-Math.log2(Math.max(1e-30,prob[buckets[i]])));
      const baseScore=parts.reduce((a,b)=>a+b,0),common={gap,embedding:[z[u],z[v]],memory:[s.memory[u].slice(),s.memory[v].slice()],previousPairCount:s.pairs[u][v]};
      if(s.trainingMode==='supervised'&&model.supervised){
        const head=model.supervised,row=z[u].concat(z[v],ctx,baseScore),scaled=row.map((value,i)=>(value-head.mean[i])/Math.max(1e-12,head.scale[i])),probability=sigmoid(head.bias+scaled.reduce((sum,value,i)=>sum+value*head.weights[i],0)),score=-Math.log2(Math.max(1e-12,1-probability));
        return {...common,score,parts:[score],probabilities:[[1-probability,probability],[1],[1]],buckets:[probability],probability,features:row,trainingMode:'supervised'};
      }
      return {...common,score:baseScore,parts,probabilities,buckets,trainingMode:'unsupervised'};
    }
  };
  const xgboost={
    initialize(model,n){return matrix(n,0);},
    update(model,s){return s.memory;},
    readout(model,s){return {embedding:s.memory,slots:[],attention:[]};},
    predict(model,s,e){
      const row=xgbFeatures(model,s,e);
      const gap=s.seen[e.u]?Math.max(0,e.t-s.last[e.u]):null;
      if(s.trainingMode==='supervised'){
        const probability=sigmoid(xgbRaw(model,row)),score=-Math.log2(Math.max(1e-12,1-probability));
        return {score,parts:[score,0,0],probabilities:[[1-probability,probability],[1],[1]],buckets:[probability],gap,embedding:[[],[]],memory:[[],[]],features:row,probability,trainingMode:'supervised',previousPairCount:s.pairs[e.u][e.v]};
      }
      const rarity=Math.max(0,.35*row[0]+.4*row[1]+1.2*(1-row[16])+.7*row[12]+.7*row[13]+.6*row[17]+.8*row[18]+.6*row[19]+.8*row[20]+.8*row[25]+.8*row[26]);
      return {score:rarity,parts:[rarity,0,0],probabilities:[[1],[1],[1]],buckets:[],gap,embedding:[[],[]],memory:[[],[]],features:row,probability:null,trainingMode:'unsupervised',previousPairCount:s.pairs[e.u][e.v]};
    }
  };
  register('categorical_graph',categorical);
  register('temporal_family',temporalFamily);
  register('xgboost',xgboost);
  global.FraudAdapters={register,get,architecture};
  if(typeof module!=='undefined')module.exports=global.FraudAdapters;
})(typeof globalThis!=='undefined'?globalThis:window);
