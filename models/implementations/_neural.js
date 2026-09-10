/* Shared GRU gates, graph aggregation and likelihood heads; per-model adapters select their readout. */
(function(global){
'use strict';
const h=typeof module!=='undefined'?require('./_math'):global.FraudModelMath;
const {zero,matrix,dot,add,linear,sigmoid,softmax,bin,architecture}=h;
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

 global.FraudNeuralPrimitives=categorical;
 if(typeof module!=='undefined')module.exports=categorical;
})(globalThis);
