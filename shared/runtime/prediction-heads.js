/* Serialized fraud heads shared with prediction_heads/implementations in Python. */
(function(global){
  'use strict';
  const labels={empirical_tail:'Historical likelihood rank',fixed_likelihood:'Fixed link likelihood'};
  function object(value){return !!value&&typeof value==='object'&&!Array.isArray(value);}
  function keys(state,expected){
    if(!object(state)||Object.keys(state).sort().join(',')!==expected.slice().sort().join(','))throw Error('Invalid prediction head state fields.');
  }
  function threshold(alpha){
    if(!Number.isFinite(alpha)||alpha<=0||alpha>1)throw Error('Head alpha must be finite and between zero and one, excluding zero.');
    return -Math.log2(alpha);
  }
  function sigmoid(logit){
    if(!Number.isFinite(logit))throw Error('Link logit must be finite.');
    if(logit>=0)return 1/(1+Math.exp(-logit));
    const value=Math.exp(logit);return Math.max(Number.MIN_VALUE,value/(1+value));
  }
  function load(state){
    if(!object(state)||state.version!==1)throw Error('Unsupported prediction head state version.');
    let probability,count;
    if(state.id==='empirical_tail'){
      keys(state,['version','id','reference_logits']);
      if(!Array.isArray(state.reference_logits)||!state.reference_logits.length||state.reference_logits.some((value,index,all)=>!Number.isFinite(value)||(index>0&&value<all[index-1])))throw Error('Head reference logits must be nonempty, finite and sorted.');
      const reference=state.reference_logits.slice();count=reference.length;
      probability=logit=>{
        let lo=0,hi=reference.length;
        while(lo<hi){const mid=lo+Math.floor((hi-lo)/2);if(reference[mid]<=logit)lo=mid+1;else hi=mid;}
        return (1+lo)/(reference.length+1);
      };
    }else if(state.id==='fixed_likelihood'){
      keys(state,['version','id','reference_count']);
      if(!Number.isSafeInteger(state.reference_count)||state.reference_count<1)throw Error('Head reference count must be a positive integer.');
      count=state.reference_count;probability=sigmoid;
    }else throw Error('Unknown prediction head: '+state.id);
    return Object.freeze({id:state.id,label:labels[state.id],reference_count:count,threshold,
      score(logit){
        if(!Number.isFinite(logit))throw Error('Link logit must be finite.');
        const tail_probability=probability(logit);
        return {tail_probability,score:Math.max(0,-Math.log2(tail_probability))};
      }});
  }
  global.FraudPredictionHeads={load,threshold,sigmoid,labels:Object.freeze(labels)};
  if(typeof module!=='undefined')module.exports=global.FraudPredictionHeads;
})(globalThis);
