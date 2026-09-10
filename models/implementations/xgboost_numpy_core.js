/* Shared causal XGBoost inputs, tree inference and exact path-dependent SHAP.
 * Reference branch probabilities use unweighted fitting-row counts, not Hessians.
 * All attributions are in raw margin (log-odds) units. */
(function(global){
  'use strict';
  const SCHEMA_VERSION=1, DEFAULT_AMOUNT_BINS=[15,35,75,150,300,650,1500,4000,10000];
  const featureLog=(x,scale=1)=>Math.log1p(Math.max(0,Number(x)||0))/scale;
  const specs=[
    ['log_amount','Payment amount','currency units',8,'Current requested payment amount.','payment','current payment'],
    ['amount_bin','Amount bucket','bucket',null,'Zero-based bucket: number of checkpoint amount-bin edges less than or equal to the requested amount.','payment','current payment'],
    ['sender_out_count','Sender outgoing payments','payments',6,'Number of earlier settled outgoing payments.','sender','all preceding history'],
    ['sender_in_count','Sender incoming payments','payments',6,'Number of earlier settled incoming payments.','sender','all preceding history'],
    ['recipient_out_count','Recipient outgoing payments','payments',6,'Number of earlier settled outgoing payments.','recipient','all preceding history'],
    ['recipient_in_count','Recipient incoming payments','payments',6,'Number of earlier settled incoming payments.','recipient','all preceding history'],
    ['sender_out_value','Sender outgoing amount','currency units',10,'Sum of earlier settled outgoing payment amounts.','sender','all preceding history'],
    ['sender_in_value','Sender incoming amount','currency units',10,'Sum of earlier settled incoming payment amounts.','sender','all preceding history'],
    ['recipient_out_value','Recipient outgoing amount','currency units',10,'Sum of earlier settled outgoing payment amounts.','recipient','all preceding history'],
    ['recipient_in_value','Recipient incoming amount','currency units',10,'Sum of earlier settled incoming payment amounts.','recipient','all preceding history'],
    ['sender_seen','Sender observed activities','activities',6,'Number of earlier observed payments and deposits. Unsettled payment attempts count as activity; outcome reports do not.','sender','all preceding history'],
    ['recipient_seen','Recipient observed activities','activities',6,'Number of earlier observed payments and deposits. Unsettled payment attempts count as activity; outcome reports do not.','recipient','all preceding history'],
    ['sender_gap','Sender activity gap','minutes',6,'Nonnegative minutes since the last observed payment or deposit; outcome reports do not reset the gap. Zero for first activity.','sender','last observed activity'],
    ['recipient_gap','Recipient activity gap','minutes',6,'Nonnegative minutes since the last observed payment or deposit; outcome reports do not reset the gap. Zero for first activity.','recipient','last observed activity'],
    ['pair_out_count','Prior payments to recipient','payments',3,'Earlier settled payments from this sender to this recipient.','account pair','all preceding history'],
    ['pair_total_count','Prior payments in either direction','payments',3,'Earlier settled payment count summed across both directions of this account pair.','account pair','all preceding history'],
    ['prior_contact','Prior payment contact','boolean',null,'One if this pair has an earlier settled payment in either direction; zero otherwise.','account pair','all preceding history'],
    ['sender_recent_in_count','Sender recent incoming payments','payments',4,'Earlier settled incoming payments during the preceding 60 minutes, including the window boundary.','sender','preceding 60 minutes'],
    ['sender_recent_out_count','Sender recent outgoing payments','payments',4,'Earlier settled outgoing payments during the preceding 60 minutes, including the window boundary.','sender','preceding 60 minutes'],
    ['recipient_recent_in_count','Recipient recent incoming payments','payments',4,'Earlier settled incoming payments during the preceding 60 minutes, including the window boundary.','recipient','preceding 60 minutes'],
    ['recipient_recent_out_count','Recipient recent outgoing payments','payments',4,'Earlier settled outgoing payments during the preceding 60 minutes, including the window boundary.','recipient','preceding 60 minutes'],
    ['sender_recent_in_value','Sender recent incoming amount','currency units',10,'Sum of earlier settled incoming payment amounts during the preceding 60 minutes, including the window boundary.','sender','preceding 60 minutes'],
    ['sender_recent_out_value','Sender recent outgoing amount','currency units',10,'Sum of earlier settled outgoing payment amounts during the preceding 60 minutes, including the window boundary.','sender','preceding 60 minutes'],
    ['recipient_recent_in_value','Recipient recent incoming amount','currency units',10,'Sum of earlier settled incoming payment amounts during the preceding 60 minutes, including the window boundary.','recipient','preceding 60 minutes'],
    ['recipient_recent_out_value','Recipient recent outgoing amount','currency units',10,'Sum of earlier settled outgoing payment amounts during the preceding 60 minutes, including the window boundary.','recipient','preceding 60 minutes'],
    ['amount_vs_sender_out_mean','Amount / sender outgoing mean','ratio',8,'Requested amount divided by the sender historical settled outgoing mean. Count and mean denominators are floored at one; without history this equals the amount.','payment and sender','all preceding history'],
    ['amount_vs_recipient_in_mean','Amount / recipient incoming mean','ratio',8,'Requested amount divided by the recipient historical settled incoming mean. Count and mean denominators are floored at one; without history this equals the amount.','payment and recipient','all preceding history']
  ];
  const featureDefinitions=Object.freeze(specs.map(([id,label,unit,scale,description,source,window],index)=>Object.freeze({id,label,unit,description,source,window,index,transform:scale?'log1p(max(0, value)) / '+scale:id==='amount_bin'?'bucket index / number of amount-bin edges':'identity',scale})));
  function recent(s,n,role,t){const events=s.incidents[n].filter(x=>t-x.t<=60&&x.role===role);return [events.length,events.reduce((sum,x)=>sum+x.amount,0)];}
  function describe(model,s,e){
    const u=e.u,v=e.v,t=e.t,amount=e.amount,si=recent(s,u,1,t),so=recent(s,u,-1,t),ri=recent(s,v,1,t),ro=recent(s,v,-1,t);
    const senderMean=s.outValue[u]/Math.max(1,s.outCount[u]),recipientMean=s.inValue[v]/Math.max(1,s.inCount[v]);
    const gapU=s.seen[u]?Math.max(0,t-s.last[u]):0,gapV=s.seen[v]?Math.max(0,t-s.last[v]):0;
    const amountBins=model.amount_bins||DEFAULT_AMOUNT_BINS;
    const readableValues=[amount,amountBins.reduce((n,edge)=>n+(amount>=edge),0),
      s.outCount[u],s.inCount[u],s.outCount[v],s.inCount[v],s.outValue[u],s.inValue[u],s.outValue[v],s.inValue[v],
      s.seen[u],s.seen[v],gapU,gapV,s.pairs[u][v],s.pairs[u][v]+s.pairs[v][u],s.pairs[u][v]+s.pairs[v][u]>0?1:0,
      si[0],so[0],ri[0],ro[0],si[1],so[1],ri[1],ro[1],amount/Math.max(1,senderMean),amount/Math.max(1,recipientMean)];
    const values=readableValues.map((value,i)=>i===1?value/amountBins.length:specs[i][3]?featureLog(value,specs[i][3]):value);
    return {values,readableValues};
  }
  const features=(model,s,e)=>describe(model,s,e).values;
  function treeValue(tree,row){let node=tree;while(node.leaf===undefined)node=row[node.feature]<=node.threshold?node.left:node.right;return node.leaf;}
  function raw(model,row){return model.base_score+model.learning_rate*model.trees.reduce((sum,tree)=>sum+treeValue(tree,row),0);}
  const probability=margin=>1/(1+Math.exp(-Math.max(-50,Math.min(50,margin))));
  function treePath(model,row,index){
    if(!Number.isInteger(index)||index<0||index>=model.trees.length)throw Error('Tree index is outside this checkpoint.');
    let nextId=0;const ids=new Map();function assign(node){ids.set(node,nextId++);if(node.leaf===undefined){assign(node.left);assign(node.right);}}assign(model.trees[index]);
    let node=model.trees[index],path='';const steps=[];
    while(node.leaf===undefined){const direction=row[node.feature]<=node.threshold?'left':'right';steps.push({feature:node.feature,featureId:model.features[node.feature],value:row[node.feature],threshold:node.threshold,direction,branch:direction,nodeId:ids.get(node),path:path||'root'});path+=direction==='left'?'L':'R';node=node[direction];}
    return {treeIndex:index,steps,leaf:node.leaf,weightedLeaf:model.learning_rate*node.leaf,leafPath:path||'root',leafNodeId:ids.get(node)};
  }

  // Stable across Python and JavaScript: numbers are encoded as IEEE-754 bytes.
  // This hashes semantic checkpoint data independently of JSON whitespace.
  function canonical(value){
    if(typeof value==='number'){if(!Number.isFinite(value))throw Error('Cannot fingerprint a nonfinite number.');const bytes=new DataView(new ArrayBuffer(8));bytes.setFloat64(0,value===0?0:value);let hex='';for(let i=0;i<8;i++)hex+=bytes.getUint8(i).toString(16).padStart(2,'0');return '~'+hex;}
    if(Array.isArray(value))return '['+value.map(canonical).join(',')+']';
    if(value&&typeof value==='object')return '{'+Object.keys(value).sort().map(k=>JSON.stringify(k)+':'+canonical(value[k])).join(',')+'}';
    return JSON.stringify(value);
  }
  function sha256(input){
    const K=[0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2];
    const utf8=unescape(encodeURIComponent(input)),bytes=Array.from(utf8,c=>c.charCodeAt(0)),bits=bytes.length*8;bytes.push(128);while(bytes.length%64!==56)bytes.push(0);for(let i=7;i>=0;i--)bytes.push(Math.floor(bits/Math.pow(256,i))&255);
    const h=[0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19],w=new Int32Array(64),rot=(x,n)=>(x>>>n)|(x<<(32-n));
    for(let offset=0;offset<bytes.length;offset+=64){for(let i=0;i<16;i++){const p=offset+i*4;w[i]=(bytes[p]<<24)|(bytes[p+1]<<16)|(bytes[p+2]<<8)|bytes[p+3];}for(let i=16;i<64;i++){const x=w[i-15],y=w[i-2];w[i]=(w[i-16]+(rot(x,7)^rot(x,18)^(x>>>3))+w[i-7]+(rot(y,17)^rot(y,19)^(y>>>10)))|0;}
      let [a,b,c,d,e,f,g,z]=h;for(let i=0;i<64;i++){const t1=(z+(rot(e,6)^rot(e,11)^rot(e,25))+((e&f)^(~e&g))+K[i]+w[i])|0,t2=((rot(a,2)^rot(a,13)^rot(a,22))+((a&b)^(a&c)^(b&c)))|0;z=g;g=f;f=e;e=(d+t1)|0;d=c;c=b;b=a;a=(t1+t2)|0;}[a,b,c,d,e,f,g,z].forEach((v,i)=>h[i]=(h[i]+v)|0);
    }return h.map(x=>(x>>>0).toString(16).padStart(8,'0')).join('');
  }
  const fingerprint=value=>sha256(canonical(value));
  const modelFingerprint=model=>fingerprint({features:model.features,amount_bins:model.amount_bins||DEFAULT_AMOUNT_BINS,base_score:model.base_score,learning_rate:model.learning_rate,trees:model.trees});
  function fail(message){throw Error('Incompatible XGBoost explanation reference: '+message);}
  function validateReference(model,sidecar){
    if(!sidecar||sidecar.version!==1||sidecar.feature_schema_version!==SCHEMA_VERSION)fail('unsupported schema version.');
    if(JSON.stringify(model.features)!==JSON.stringify(featureDefinitions.map(f=>f.id))||JSON.stringify(sidecar.features)!==JSON.stringify(model.features))fail('feature order differs.');
    if(!Number.isFinite(model.base_score)||!Number.isFinite(model.learning_rate)||!Array.isArray(model.trees)||!model.trees.length)fail('invalid booster.');
    if(sidecar.checkpoint_id!==model.checkpoint_id||sidecar.model_fingerprint!==modelFingerprint(model))fail('checkpoint differs; regenerate the sidecar.');
    if(!/^[a-f0-9]{64}$/.test(sidecar.checkpoint_sha256||''))fail('missing checkpoint provenance.');
    const reference=sidecar.reference;if(!reference||reference.convention!=='unweighted_labeled_fitting_rows'||!Number.isInteger(reference.row_count)||reference.row_count<=0)fail('reference population is invalid.');
    const unsigned={...sidecar};delete unsigned.reference_id;if(sidecar.reference_id!==fingerprint(unsigned))fail('reference metadata or counts changed.');
    if(!Array.isArray(sidecar.node_counts)||sidecar.node_counts.length!==model.trees.length)fail('tree count differs.');
    model.trees.forEach((tree,index)=>{const counts=sidecar.node_counts[index];if(!Array.isArray(counts))fail('node counts missing.');let cursor=0;function visit(node){const count=counts[cursor++];if(!Number.isInteger(count)||count<0)fail('invalid node count.');if(node.leaf!==undefined){if(!Number.isFinite(node.leaf))fail('invalid leaf.');}else{if(!Number.isInteger(node.feature)||node.feature<0||node.feature>=model.features.length||!Number.isFinite(node.threshold)||!node.left||!node.right)fail('invalid split.');if(!count||visit(node.left)+visit(node.right)!==count)fail('child counts do not sum to a positive parent count.');}return count;}if(visit(tree)!==reference.row_count||cursor!==counts.length)fail('node counts do not match tree topology or population.');});
    return true;
  }
  function createExplainer(model,sidecar){
    validateReference(model,sidecar);
    const compiled=model.trees.map((tree,index)=>{
      const nodes=[],splits=[],used=new Set();function walk(node){const out={...node,id:nodes.length,count:sidecar.node_counts[index][nodes.length]};nodes.push(out);if(node.leaf===undefined){used.add(node.feature);out.splitIndex=splits.length;splits.push(out);out.left=walk(node.left);out.right=walk(node.right);}return out;}const root=walk(tree),featureIds=Array.from(used).sort((a,b)=>a-b),m=featureIds.length;
      // Exact enumeration is bounded for this repository's small exported trees.
      if(m>16)fail('exact explanation supports at most 16 distinct features per tree.');
      splits.forEach(node=>node.featureBit=1<<featureIds.indexOf(node.feature));
      function expected(node){return node.leaf!==undefined?node.leaf:(node.left.count*expected(node.left)+node.right.count*expected(node.right))/node.count;}
      const expectedValue=expected(root),weights=new Float64Array(m),factorial=[1];for(let i=1;i<=m;i++)factorial[i]=factorial[i-1]*i;for(let size=0;size<m;size++)weights[size]=factorial[size]*factorial[m-size-1]/factorial[m];
      const sizes=new Uint8Array(1<<m);for(let mask=1;mask<sizes.length;mask++)sizes[mask]=sizes[mask>>1]+(mask&1);
      return {root,splits,featureIds,expectedValue,weights,sizes,cache:new Map()};
    });
    const baseline=model.base_score+model.learning_rate*compiled.reduce((sum,tree)=>sum+tree.expectedValue,0);
    function explain(values){
      if(!Array.isArray(values)||values.length!==model.features.length||values.some(x=>!Number.isFinite(x)))throw Error('Explanation needs one finite value per model feature.');
      const contributions=Array(model.features.length).fill(0);
      for(const tree of compiled){
        // Coalition expectations can visit off-path nodes. Cache ALL split outcomes.
        const directions=tree.splits.map(node=>values[node.feature]<=node.threshold),key=directions.map(x=>x?'L':'R').join('');let attribution=tree.cache.get(key);
        if(!attribution){const m=tree.featureIds.length,coalitions=new Float64Array(1<<m);function value(node,mask){if(node.leaf!==undefined)return node.leaf;if(mask&node.featureBit)return value(directions[node.splitIndex]?node.left:node.right,mask);return (node.left.count*value(node.left,mask)+node.right.count*value(node.right,mask))/node.count;}
          for(let mask=0;mask<coalitions.length;mask++)coalitions[mask]=value(tree.root,mask);
          attribution=new Float64Array(m);for(let feature=0;feature<m;feature++){const bit=1<<feature;let sum=0;for(let mask=0;mask<coalitions.length;mask++)if(!(mask&bit))sum+=tree.weights[tree.sizes[mask]]*(coalitions[mask|bit]-coalitions[mask]);attribution[feature]=sum*model.learning_rate;}
          // Limit memory even for future larger trees with many split patterns.
          if(tree.cache.size>=4096)tree.cache.delete(tree.cache.keys().next().value);tree.cache.set(key,attribution);
        }tree.featureIds.forEach((feature,i)=>contributions[feature]+=attribution[i]);
      }
      return {baseline,contributions,rawMargin:raw(model,values)};
    }
    return Object.freeze({baseline,referenceId:sidecar.reference_id,explain});
  }
  global.FraudXGBoost={SCHEMA_VERSION,featureDefinitions,features,describe,treeValue,raw,probability,treePath,fingerprint,modelFingerprint,validateReference,createExplainer};
  if(typeof module!=='undefined')module.exports=global.FraudXGBoost;
})(typeof globalThis!=='undefined'?globalThis:window);
