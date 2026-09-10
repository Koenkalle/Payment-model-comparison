/* Numerical primitives shared by the explicitly registered model implementations. */
(function(global){
'use strict';
  const zero=n=>Array(n).fill(0),matrix=(n,m)=>Array.from({length:n},()=>zero(m));
  const dot=(a,b)=>{let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s;},add=(a,b)=>a.map((x,i)=>x+b[i]);
  function linear(a,w,b){const out=new Array(w[0].length);for(let j=0;j<out.length;j++){let sum=b?b[j]:0;for(let i=0;i<a.length;i++)sum+=a[i]*w[i][j];out[j]=sum;}return out;}
  const sigmoid=x=>1/(1+Math.exp(-Math.max(-50,Math.min(50,x))));
  function softmax(a){const m=Math.max(...a),e=a.map(x=>Math.exp(x-m)),s=e.reduce((a,b)=>a+b,0);return e.map(x=>x/s);}
  const bin=(x,edges)=>edges.reduce((s,v)=>s+(x>=v),0);
  const architecture=m=>m.architecture||{memory:'gru',readout:'attention',layers:2};

 global.FraudModelMath={zero,matrix,dot,add,linear,sigmoid,softmax,bin,architecture};
 if(typeof module!=='undefined')module.exports=global.FraudModelMath;
})(globalThis);
