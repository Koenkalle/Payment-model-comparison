/* Deterministic synthetic requests. Outcomes are separate from model inputs. */
(function(global){
  'use strict';
  const catalog=[
    {id:'relay',name:'Collection and long relay',detail:'Collections, a six-stage relay, outside deposits, and background commerce.'},
    {id:'split',name:'Split, merge, and cycle',detail:'Collections split across intermediaries, merge again, and partly return through a cycle.'},
    {id:'benign',name:'Busy merchants and payroll',detail:'Legitimate collections, supplier payouts, payroll, and large one-off payments.'},
    {id:'takeover',name:'Account takeover and new recipients',detail:'An established account switches counterparties; another attempt resembles ordinary spending.'},
    {id:'mixed',name:'Mixed week with a traffic surge',detail:'Relay, takeover, and legitimate bursts amid a broad increase in activity.'}
  ];
  const sizes={small:{n:32,base:450,days:4},medium:{n:64,base:1000,days:7},large:{n:112,base:2100,days:10}};
  function rng(seed){let a=seed>>>0;return()=>{a+=0x6D2B79F5;let t=a;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296;};}
  const normal=r=>Math.sqrt(-2*Math.log(Math.max(1e-9,r())))*Math.cos(2*Math.PI*r());
  function population(n,seed){
    const r=rng(seed),perm=Array.from({length:n},(_,i)=>i);
    for(let i=n-1;i>0;i--){const j=Math.floor(r()*(i+1));[perm[i],perm[j]]=[perm[j],perm[i]];}
    const people=['Ava','Ben','Cleo','Dani','Eli','Faye','Gus','Hana','Ida','Jules','Kai','Lena'],accounts=Array(n);
    perm.forEach((id,l)=>{accounts[id]={id,name:l<5?['North market','City shop','Harbor payroll','West supply','Corner cafe'][l]:people[(l-5)%12]+' '+String(l+1),role:l<5?['merchant','merchant','employer','supplier','merchant'][l]:'personal',logical:l};});
    return {accounts,perm};
  }
  function background(n,count,duration,seed,training=false){
    const r=rng(seed),{accounts,perm}=population(n,seed+1),events=[],truth={};let serial=0;
    const add=(t,u,v,amount,kind='payment',extra={})=>{const e={id:'P'+String(++serial).padStart(5,'0'),t:+t.toFixed(4),u:u<0?-1:perm[u],v:perm[v],amount:+Math.max(1,amount).toFixed(2),kind,...extra};events.push(e);return e;};
    // Registered high-ID accounts stay inactive until a scenario begins.
    const active=n-6,personal=()=>5+Math.floor(r()*(active-5));
    for(let i=0;i<count;i++){
      const t=(i+r())*duration/count,mode=r();let u,v,amount;
      if(mode<.10){add(t,-1,personal(),Math.exp(6+normal(r)*.75),'deposit');continue;}
      if(mode<.64){u=personal();v=r()<.7?u%2:(r()<.5?4:personal());amount=Math.exp(3.65+normal(r)*.7);}
      else if(mode<.82){u=personal();v=5+(u-5+1)%(active-5);amount=Math.exp(4+normal(r)*.65);}
      else if(mode<.91){u=2;v=personal();amount=Math.exp(7+normal(r)*.17);}
      else {u=r()<.5?0:1;v=3;amount=Math.exp(6.2+normal(r)*.48);}
      if(u===v)v=(v+1)%active;
      // Retain odd historical requests: no oracle filters a clean subset.
      if(training&&r()<.025){v=personal();if(v===u)v=0;amount*=3+r()*7;}
      add(t,u,v,amount,'payment',training&&r()<.02?{settled:false}:{});
    }
    return {accounts,perm,events,truth,add,r};
  }
  function build(name='relay',size='medium',seed=42,reportDelay=1440,forwardDelay=2){
    const {n,base,days}=sizes[size]||sizes.medium,w=background(n,base,days*1440,seed),{events,accounts,truth,add,r}=w;
    const focus=[],bookmarks=[],start=days*1440*.53;
    const tag=(e,fraud=false)=>{focus.push(e.id);truth[e.id]=fraud;return e;};
    const pay=(t,u,v,a,bad=true)=>tag(add(t,u,v,a),bad);
    const [m,a,b,c,d,f,g,h,j,k]=Array.from({length:10},(_,i)=>n-10+i);
    const bookmark=(label,e)=>bookmarks.push({label,id:e.id});
    const report=(id,t)=>{const e=events.find(x=>x.id===id);events.push({id:'R'+id,t,u:-1,v:e.v,amount:0,kind:'report',reference:id});};
    function relay(at,split=false){
      const first=focus.length;let last;
      for(let i=0;i<7;i++)last=pay(at+i*.65,5+i,m,220+i*63);
      bookmark('Collection begins',events.find(e=>e.id===focus[first]));tag(add(at+4.6,-1,m,1700,'deposit'));
      if(!split){const chain=[m,a,b,c,d,f,g];let amount=2250,t=at+5;
        for(let i=0;i<chain.length-1;i++){t+=forwardDelay;last=pay(t,chain[i],chain[i+1],amount);amount*=.91;}
        bookmark('Relay request',last);
      }else{let t=at+5+forwardDelay;
        pay(t,m,a,680);pay(t+.3,m,b,720);pay(t+.6,m,c,710);t+=forwardDelay;
        pay(t,a,d,630);pay(t+.4,b,d,670);pay(t+.8,c,d,650);t+=forwardDelay;
        last=pay(t,d,f,1840);pay(t+forwardDelay,f,m,210);pay(t+forwardDelay+.5,f,g,1580);bookmark('Merge request',last);
      }
      focus.slice(first).filter(id=>truth[id]).slice(0,4).forEach(id=>report(id,Math.max(at+reportDelay,last.t+1)));
    }
    function benign(at){
      for(let i=0;i<15;i++)pay(at+i*.45,5+i%(n-12),0,20+r()*130,false);
      bookmark('Merchant payout',pay(at+9,0,3,1680,false));
      for(let i=0;i<12;i++)pay(at+40+i*.6,2,5+i,1000+r()*230,false);
      tag(add(at+87,-1,13,11000,'deposit'));bookmark('Large legitimate payment',pay(at+90,13,3,8200,false));
    }
    function takeover(at){
      const first=focus.length;bookmark('Account changes recipient',pay(at,12,j,3600));
      pay(at+.5,12,k,1900);pay(at+1,j,h,3280);pay(at+2,h,k,3050);
      // Intentionally ordinary-looking fraud: labels cannot reveal intent
      // to an anomaly model when the observations look legitimate.
      bookmark('Routine-looking fraud',pay(at+170,14,0,48));
      focus.slice(first).filter(id=>truth[id]).forEach(id=>report(id,at+Math.max(180,reportDelay)));
    }
    if(name==='relay')relay(start);
    if(name==='split')relay(start,true);
    if(name==='benign')benign(start);
    if(name==='takeover')takeover(start);
    if(name==='mixed'){benign(start);relay(start+360,true);takeover(start+900);
      for(let i=0;i<180;i++)pay(start+1200+i*.5,5+i%(n-12),i%2,18+r()*95,false);
    }
    events.sort((a,b)=>a.t-b.t||a.id.localeCompare(b.id));
    for(const e of events)if(e.kind==='payment'&&truth[e.id]===undefined)truth[e.id]=false;
    const first=events.findIndex(e=>focus.includes(e.id));
    return {name,size,seed,accounts,events,truth,focus,bookmarks,startIndex:Math.max(0,first),description:(catalog.find(x=>x.id===name)||catalog[0]).detail};
  }
  function trainingEpisodes(count=1800,n=24,length=48,seed=1731){return Array.from({length:count},(_,i)=>{const w=background(n,length,2880,seed+i*19,true);return {accounts:n,events:w.events.sort((a,b)=>a.t-b.t||a.id.localeCompare(b.id))};});}
  // Historical labels are deliberately sparse: 1 = a flagged fraud report,
  // 0 = an observed unflagged event, -1 = a fraud-like event with no flag.
  // The latter is excluded by the offline supervised trainer rather than
  // pretending that every unflagged event is clean.
  function trainingLabeled(fraudFlags=96,seed=811){
    const target=Math.max(0,Math.floor(Number(fraudFlags)||0)),names=catalog.map(x=>x.id),episodes=[],positives=[];let i=0;
    while((positives.filter(x=>x.episode<Math.max(1,Math.floor(episodes.length*.8))).length<target||episodes.length<16)&&i<200){
      const name=names[i%names.length],size=i%4===0?'medium':'small',d=build(name,size,seed+i*37,720,2),events=d.events.filter(e=>e.kind==='payment').map(e=>({...e})),episode=episodes.length;
      episodes.push({accounts:d.accounts.length,events,truth:d.truth});events.forEach((e,index)=>{if(d.truth[e.id])positives.push({episode,index});});i++;
    }
    const trainEpisodes=Math.max(1,Math.floor(episodes.length*.8)),eligible=positives.filter(x=>x.episode<trainEpisodes);
    if(target>eligible.length)throw Error('Unable to generate '+target+' flagged fraud events in the training partition.');
    const r=rng(seed+991),shuffled=eligible.slice(),holdout=positives.filter(x=>x.episode>=trainEpisodes);for(let j=shuffled.length-1;j>0;j--){const k=Math.floor(r()*(j+1));[shuffled[j],shuffled[k]]=[shuffled[k],shuffled[j]];}for(let j=holdout.length-1;j>0;j--){const k=Math.floor(r()*(j+1));[holdout[j],holdout[k]]=[holdout[k],holdout[j]];}
    const chosen=new Set(shuffled.slice(0,target).map(x=>x.episode+':'+x.index)),validationTarget=target?Math.min(holdout.length,Math.max(1,Math.floor(target*.25))):0,chosenValidation=new Set(holdout.slice(0,validationTarget).map(x=>x.episode+':'+x.index));
    let unknown=0;
    const out=episodes.map((ep,episode)=>({accounts:ep.accounts,events:ep.events.map((e,index)=>{const key=episode+':'+index,truth=!!ep.truth[e.id];let label=0;if(truth){if(chosen.has(key)||chosenValidation.has(key))label=1;else{label=-1;unknown++;}}return {...e,label};})}));
    return {version:3,seed,flags_requested:target,flags_used:target,validation_flags_used:validationTarget,unknown_fraud_events:unknown,episodes:out};
  }
  global.FraudScenarios={catalog,sizes,build,background,trainingEpisodes,trainingLabeled};
  if(global.FraudDatasets)global.FraudDatasets.register('synthetic_payments',{load:options=>global.FraudScenarios.build(options.name||'relay',options.size||'medium',options.seed??42,options.reportDelay??1440,options.forwardDelay??2)});
  if(typeof module!=='undefined')module.exports=global.FraudScenarios;
})(typeof globalThis!=='undefined'?globalThis:window);
