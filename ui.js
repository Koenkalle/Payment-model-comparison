(function(){
  'use strict';
  const root=document.getElementById('fraud-memory-demo'),$=id=>root.querySelector('#'+id),core=globalThis.FraudCore,sc=globalThis.FraudScenarios;
  const bundle=JSON.parse($('fd-model-data').textContent),models=bundle.models,ns='http://www.w3.org/2000/svg',compare=globalThis.FraudComparison;
  let model=models.find(m=>m.id===bundle.default)||models[0];
  const currencyFormat=new Intl.NumberFormat('en-IE',{style:'currency',currency:'EUR',maximumFractionDigits:0}),money=x=>currencyFormat.format(x);
  const clock=t=>'D'+(Math.floor(t/1440)+1)+' '+String(Math.floor(t/60)%24).padStart(2,'0')+':'+String(Math.floor(t)%60).padStart(2,'0');
  const percent=x=>x===null?'—':(100*x).toFixed(1)+'%';
  let data,group,entries=[],count=0,selected=0,timer=null,result,pending,prediction,follow=true;
  const comparisonCache=new compare.ComparisonCache(models,3),datasets=new Map();
  let renderVersion=0,latestTask=Promise.resolve(),busy=false,paintedGroup=null,paintedCount=-1;
  const defaultWarmup=bundle.policy?.warmup||128;
  const defaultPolicies=Object.fromEntries(models.map(m=>{
    const candidates=m.policy_validation?.unsupervised?.candidates||[],middle=candidates.length?candidates[Math.floor(candidates.length/2)][0]:10;
    return [m.id,{strategy:'shared',alpha:.02,warmup:defaultWarmup,manualTau:Number(middle)||10,falseBlockCost:1,missedFraudCost:20,objective:'f1'}];
  }));
  const modelPolicies=JSON.parse(JSON.stringify(defaultPolicies));
  const name=n=>data.accounts[n]?.name||'Outside';
  const desc=e=>e.kind==='report'?'Fraud confirmation · '+e.reference:e.kind==='deposit'?'Outside → '+name(e.v)+' · '+money(e.amount):name(e.u)+' → '+name(e.v)+' · '+money(e.amount);
  function node(tag,attrs={},parent,text=''){const e=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);e.textContent=text;if(parent)parent.appendChild(e);return e;}
  function option(parent,value,text){const e=document.createElement('option');e.value=String(value);e.textContent=text;parent.appendChild(e);}
  function stop(){if(timer)clearInterval(timer);timer=null;$('fd-play').textContent='Play';}
  function numberValue(id,fallback,min,max){const value=Number($(id).value);return Number.isFinite(value)?Math.max(min,Math.min(max,value)):fallback;}
  function readPolicyConfig(){
    const p=modelPolicies[model.id]||defaultPolicies[model.id];
    p.strategy=$('fd-model-strategy').value;
    p.alpha=numberValue('fd-model-alpha',.02,.001,.5);p.warmup=Math.floor(numberValue('fd-model-warmup',defaultWarmup,1,10000));
    p.manualTau=numberValue('fd-model-tau',p.manualTau??10,0,100000);p.falseBlockCost=numberValue('fd-model-false-cost',1,.01,10000);p.missedFraudCost=numberValue('fd-model-missed-cost',20,.01,10000);p.objective=$('fd-model-objective').value;
    modelPolicies[model.id]=p;return p;
  }
  function syncPolicyControls(){
    const p=modelPolicies[model.id]||defaultPolicies[model.id],strategy=p.strategy;
    $('fd-model-strategy').value=strategy;$('fd-model-alpha').value=String(p.alpha);$('fd-model-warmup').value=String(p.warmup);$('fd-model-tau').value=String(p.manualTau);$('fd-model-false-cost').value=String(p.falseBlockCost);$('fd-model-missed-cost').value=String(p.missedFraudCost);$('fd-model-objective').value=p.objective;
    $('fd-model-alpha').disabled=strategy!=='shared';$('fd-model-warmup').disabled=strategy!=='shared';$('fd-model-tau').disabled=strategy!=='manual';$('fd-model-false-cost').disabled=strategy!=='tuned';$('fd-model-missed-cost').disabled=strategy!=='tuned';$('fd-model-objective').disabled=strategy!=='auto';
  }
  function modelPolicyOptions(trainingMode){
    return Object.fromEntries(models.map(m=>{
      const p=modelPolicies[m.id]||defaultPolicies[m.id],fits=!!m.policy_validation?.[trainingMode],strategy=(p.strategy==='tuned'||p.strategy==='auto')&&!fits?'shared':p.strategy;
      return [m.id,{decisionPolicy:strategy,alpha:p.alpha,warmup:p.warmup,manualTau:p.manualTau,falseBlockCost:p.falseBlockCost,missedFraudCost:p.missedFraudCost,objective:p.objective}];
    }));
  }
  function policyOptions(trainingMode,warmup){
    const scope=$('fd-policy').value,options={alpha:numberValue('fd-alpha',.02,.001,.5),mode:$('fd-mode').value,trainingMode,decisionPolicy:'shared',warmup};
    if(scope==='individual')options.modelPolicies=modelPolicyOptions(trainingMode);
    if(scope==='auto')options.modelPolicies=Object.fromEntries(models.map(m=>[m.id,{decisionPolicy:'auto',objective:$('fd-objective').value}]));
    return options;
  }
  function currentPolicyLabel(state){
    if(!state)return '—';
    if(state.decisionPolicy==='shared')return 'Unlabeled budget';
    if(state.decisionPolicy==='auto')return 'Auto '+String(state.policyFit?.objective||state.objective||'F1').toUpperCase();
    if(state.decisionPolicy==='tuned')return 'Cost tune';
    if(state.decisionPolicy==='manual')return 'Fixed τ';
    return state.decisionPolicy;
  }
  function cells(id,values){const parent=$(id);parent.replaceChildren();(values||Array(8).fill(0)).forEach((v,i)=>{const e=document.createElement('span');e.className='fd-cell';e.style.opacity=String(.12+Math.abs(v)*.8);e.style.background=v<0?'var(--viz-series-2)':'var(--viz-series-1)';e.setAttribute('aria-label','State '+(i+1)+': '+v.toFixed(3));e.setAttribute('data-tooltip','State '+(i+1)+': '+v.toFixed(3));parent.appendChild(e);});}
  function row(parent,values){const tr=document.createElement('tr');values.forEach(v=>{const td=document.createElement('td');td.textContent=String(v);tr.appendChild(td);});parent.appendChild(tr);}
  function rebuild(keep=false){
    stop();const seed=Math.max(1,Math.min(99999,Math.floor(Number($('fd-seed').value)||42)));$('fd-seed').value=String(seed);
    group?.cancel();
    const datasetKey=JSON.stringify([$('fd-scenario').value,$('fd-size').value,seed,Number($('fd-delay').value)*60,Number($('fd-forward').value)]);
    if(datasets.has(datasetKey)){data=datasets.get(datasetKey);datasets.delete(datasetKey);}
    else data=sc.build(...JSON.parse(datasetKey));
    datasets.set(datasetKey,data);if(datasets.size>2)datasets.delete(datasets.keys().next().value);
    const warmup=Math.max(1,Math.floor(Number($('fd-warmup').value)||bundle.policy?.warmup||128));$('fd-warmup').value=String(warmup);
    const trainingMode=$('fd-training-mode').value==='supervised'?'supervised':'unsupervised';
    const canTune=models.every(m=>m.policy_validation?.[trainingMode]);$('fd-policy').querySelector('option[value="auto"]').disabled=!canTune;
    if(!canTune&&$('fd-policy').value==='auto')$('fd-policy').value='shared';
    const scope=$('fd-policy').value;
    $('fd-auto-controls').hidden=scope!=='auto';$('fd-individual-controls').hidden=scope!=='individual';
    $('fd-warmup').disabled=scope!=='shared';$('fd-alpha').disabled=false;
    $('fd-alpha-label').textContent=scope==='shared'?'Shared target α':'Evaluation ranking budget α';
    group=comparisonCache.acquire(data,policyOptions(trainingMode,warmup));
    count=keep?Math.min(count,data.events.length):data.startIndex;follow=true;selected=data.events[count]?.v??0;
    $('fd-account').replaceChildren();data.accounts.forEach(a=>option($('fd-account'),a.id,'#'+(a.id+1)+' · '+a.name));
    $('fd-jump').replaceChildren();option($('fd-jump'),'','Choose an event…');option($('fd-jump'),0,scope==='shared'?'Start · learn each cutoff':'Start · use configured cutoffs');
    for(const b of data.bookmarks)option($('fd-jump'),data.events.findIndex(e=>e.id===b.id),b.label);
    const report=data.events.findIndex(e=>e.kind==='report');if(report>=0)option($('fd-jump'),report,'First delayed confirmation');option($('fd-jump'),data.events.length,'End of scenario');
    $('fd-forward-value').textContent=$('fd-forward').value+' min';$('fd-delay-value').textContent=$('fd-delay').value+' h';
    $('fd-scope').textContent=data.accounts.length+' accounts · '+data.events.length.toLocaleString()+' events · '+Math.ceil(data.events.at(-1).t/1440)+' days';
    $('fd-mode-label').textContent=($('fd-mode').value==='enforce'?'Separate histories after blocks':'Identical history for all models')+' · '+(trainingMode==='supervised'?'flagged-history mode':'no-label mode');syncPolicyControls();return render();
  }
  function graph(){
    if(!result)return;const svg=$('fd-graph'),w=Math.max(300,$('fd-graph-container').clientWidth),small=w<520,H=small?340:320;
    svg.setAttribute('viewBox','0 0 '+w+' '+H);svg.setAttribute('height',String(H));svg.replaceChildren();
    node('title',{},svg,'Observed payment context around '+name(selected));
    const settled=result.state.payments,focus=selected,roots=[focus];
    if(pending?.kind==='payment'){for(const n of[pending.u,pending.v])if(!roots.includes(n))roots.push(n);}
    const keep=new Set(roots),limit=small?10:18,recent=settled.slice(-320);
    for(let hop=0;hop<2;hop++){const previous=new Set(keep);for(const e of recent.slice().reverse())if((previous.has(e.u)||previous.has(e.v))&&keep.size<limit){keep.add(e.u);if(keep.size<limit)keep.add(e.v);}}
    const ids=Array.from(keep),others=ids.filter(n=>!roots.includes(n)),pos={};
    roots.forEach((n,i)=>pos[n]=roots.length===3?[{x:w*.5,y:H*.29},{x:w*.33,y:H*.58},{x:w*.67,y:H*.58}][i]:{x:roots.length===1?w/2:w*(.34+i*.32),y:H*.49});
    others.forEach((n,i)=>{const angle=-Math.PI/2+i*2*Math.PI/Math.max(1,others.length);pos[n]={x:w/2+Math.cos(angle)*(w/2-64),y:H/2+Math.sin(angle)*(H/2-42)};});
    const defs=node('defs',{},svg);
    for(const[id,color]of[['fd-arrow','var(--border)'],['fd-proposed','var(--viz-series-2)']]){const m=node('marker',{id,markerWidth:6,markerHeight:6,refX:5,refY:3,orient:'auto',markerUnits:'userSpaceOnUse'},defs);node('path',{d:'M0 0 L6 3 L0 6 Z',fill:color},m);}
    const pairs=new Map();for(const e of recent)if(keep.has(e.u)&&keep.has(e.v))pairs.set(e.u+'-'+e.v,e);
    function edge(e,proposal){const a=pos[e.u],b=pos[e.v];if(!a||!b)return;const dx=b.x-a.x,dy=b.y-a.y,len=Math.hypot(dx,dy)||1;
      const sx=a.x+dx/len*19,sy=a.y+dy/len*19,ex=b.x-dx/len*22,ey=b.y-dy/len*22;
      const midx=(sx+ex)/2-dy*.07,midy=(sy+ey)/2+dx*.07;
      node('path',{d:`M${sx} ${sy} Q${midx} ${midy} ${ex} ${ey}`,class:proposal?'fd-request-edge':'fd-graph-edge','marker-end':proposal?'url(#fd-proposed)':'url(#fd-arrow)','data-tooltip':clock(e.t)+' · '+desc(e)},svg);
    }
    pairs.forEach(e=>edge(e,false));if(pending?.kind==='payment')edge(pending,true);
    ids.forEach(n=>{const p=pos[n],active=roots.includes(n),g=node('g',{'data-tooltip':name(n)+' · '+result.state.seen[n]+' observed events'},svg);
      node('circle',{cx:p.x,cy:p.y,r:active?18:12,fill:active?'var(--viz-series-1)':'var(--muted-foreground)','fill-opacity':active?.24:.12},g);
      node('text',{x:p.x,y:p.y+4,'text-anchor':'middle'},g,String(n+1));
      const short=name(n).length>14?name(n).slice(0,12)+'…':name(n);
      if(active)node('text',{x:p.x,y:p.y+34,'text-anchor':'middle'},g,short);
      g.addEventListener('click',()=>{selected=n;follow=false;$('fd-account').value=String(n);render(true);});
    });
    const outside=result.state.events.filter(e=>e.kind==='deposit'&&keep.has(e.v));
    $('fd-network-caption').textContent=ids.length+' / '+data.accounts.length+' accounts shown'+(outside.length?' · '+outside.length+' observed outside deposits':'');
    node('desc',{},svg,ids.map(n=>name(n)).join(', ')+'. Bounded historical neighborhood; no outside funding paths are inferred.');
  }
  function chart(){
    if(!result)return;const svg=$('fd-chart'),w=Math.max(300,$('fd-chart-container').clientWidth),height=190;
    svg.setAttribute('viewBox','0 0 '+w+' '+height);svg.setAttribute('height',String(height));svg.replaceChildren();
    const records=result.state.decisions;if(!records.length){node('text',{x:64,y:54},svg,'Process requests to see scores and the learned cutoff.');return;}
    const d3=globalThis.d3,points=records.map((r,i)=>({x:r.event.t/1440,y:r.score,tau:r.tauBefore,r,i}));
    const maxX=Math.max(.05,...points.map(p=>p.x)),maxY=Math.max(1,...points.map(p=>Math.max(p.y,p.tau||0)));
    const x=d3.scaleLinear().domain([0,maxX*1.015]).range([64,w-14]),y=d3.scaleLinear().domain([0,maxY*1.08]).range([height-42,14]);
    node('rect',{x:64,y:14,width:w-78,height:height-56,fill:'none',stroke:'var(--border)','data-chart-frame':''},svg);
    const drawAxis=(scale,vertical)=>{for(const t of scale.ticks(vertical?3:w<500?3:5))node('text',vertical?{x:56,y:scale(t)+4,'text-anchor':'end'}:{x:scale(t),y:height-24,'text-anchor':t===0?'start':'middle'},svg,String(+t.toFixed(1)));};drawAxis(x,false);drawAxis(y,true);
    node('text',{x:64,y:height-4,class:'axis-title','data-axis':'x'},svg,'Elapsed days');node('text',{x:8,y:16,class:'axis-title','data-axis':'y'},svg,'Bits');
    const thresholds=points.filter(p=>p.tau!==null),path=d3.line().x(p=>x(p.x)).y(p=>y(p.tau));
    if(thresholds.length)node('path',{d:path(thresholds),fill:'none',stroke:'var(--viz-series-1)','stroke-width':1.8},svg);
    // Every block and warm-up point is retained; allow points are thinned
    // only for rendering. All underlying decisions remain in the replay.
    const stride=Math.max(1,Math.ceil(points.length/600));
    points.filter((p,i)=>i%stride===0||p.r.decision==='BLOCK'||i===points.length-1).forEach(p=>node('circle',{cx:x(p.x),cy:y(p.y),r:p.r.decision==='BLOCK'?2.6:1.7,fill:p.r.decision==='BLOCK'?'var(--viz-series-2)':'var(--muted-foreground)','fill-opacity':p.r.decision==='BLOCK'?.9:.38,'data-tooltip':clock(p.r.event.t)+' · '+p.r.decision+' · '+p.y.toFixed(2)+' bits'},svg));
    node('title',{},svg,'Pre-decision surprise and learned cutoff over '+maxX.toFixed(1)+' days');
    node('desc',{},svg,records.length+' requests. '+(result.state.decisionPolicy==='shared'?'Cutoff learned without outcomes.':'Cutoff configured or fitted before replay and frozen.')+' Line shows its value before each request.');
  }
  function evaluate(){
    $('fd-evaluation').hidden=!$('fd-truth').checked;if(!$('fd-truth').checked||!result||busy)return;
    const d=result.state.decisions.filter(r=>r.decision!=='LEARNING'),fraud=d.filter(r=>data.truth[r.event.id]),legit=d.filter(r=>!data.truth[r.event.id]);
    const tp=fraud.filter(r=>r.decision==='BLOCK').length,fp=legit.filter(r=>r.decision==='BLOCK').length;
    const comparisonBody=$('fd-model-metrics').querySelector('tbody');comparisonBody.replaceChildren();
    let budget=0;
    for(const entry of entries){const state=entry.result.state,m=compare.metrics(state.decisions,data.truth,group.options.alpha,state.errorCosts);budget=m.budget;
      row(comparisonBody,[entry.model.label,m.tp+' / '+(m.tp+m.fn),m.fp,percent(m.precision),percent(m.blockRate),state.decisionPolicy==='tuned'?Number(m.errorCost.toFixed(2)):'—',percent(m.recallAtBudget)]);
    }
    $('fd-evaluation-protocol').textContent=(result.state.mode==='shadow'?'Matched history':'Independent blocking histories')+' · ranking uses a common '+percent(group.options.alpha)+' budget ('+budget+' requests). Error cost is shown only for cost-tuned models and uses that model’s own costs. These replay outcomes do not tune τ.';
    const body=$('fd-metrics').querySelector('tbody');body.replaceChildren();
    [['Evaluated requests',d.length],['Fraud blocked / all generated fraud',tp+' / '+fraud.length],['Legitimate requests blocked / all legitimate',fp+' / '+legit.length],['Precision among block decisions',tp+fp?(100*tp/(tp+fp)).toFixed(1)+'%':'—'],['Warm-up requests excluded',result.state.calibration.length],['Historical outcome labels used to fit τ',result.state.policyFit?result.state.policyFit.positives+result.state.policyFit.negatives:0],['Replay outcome labels used to change model or τ','0']].forEach(r=>row(body,r));
    const errors=$('fd-errors').querySelector('tbody');errors.replaceChildren();
    d.filter(r=>(r.decision==='BLOCK')!==!!data.truth[r.event.id]).slice(-6).reverse().forEach(r=>row(errors,[r.event.id+' · '+desc(r.event),data.truth[r.event.id]?'Fraud':'Legitimate',r.decision]));
  }
  function setBusy(value,message='Updating results…'){
    busy=value;root.dataset.busy=String(value);$('fd-work-status').textContent=value?message:'';
    $('fd-compare').setAttribute('aria-busy',String(value));
    $('fd-back').disabled=value||count===0;$('fd-next').disabled=value||count>=data.events.length;$('fd-run').disabled=value||count>=data.events.length;
  }
  function render(viewOnly=false){
    if(viewOnly&&!busy&&paintedGroup===group&&paintedCount===count){paint();return Promise.resolve();}
    const version=++renderVersion,requestedGroup=group,target=count;setBusy(true);
    latestTask=(async()=>{
      try{
        const next=await requestedGroup.seekAsync(target,{priorityModel:model.id,cancelled:()=>version!==renderVersion,onProgress:p=>{if(version===renderVersion)$('fd-work-status').textContent='Updating results · '+p.done+' / '+p.total+' models ready. Settings remain available.';}});
        if(!next||version!==renderVersion)return;
        entries=next;paintedGroup=requestedGroup;paintedCount=target;setBusy(false);paint();
      }catch(error){
        if(version!==renderVersion)return;
        setBusy(false);$('fd-work-status').textContent='Could not update results: '+error.message;root.dataset.error=error.message;
        throw error;
      }
    })();
    return latestTask;
  }
  async function whenIdle(){for(;;){const task=latestTask;await task;if(task===latestTask)return;}}
  function paint(){
    const active=entries.find(e=>e.model.id===model.id);result=active.result;pending=data.events[count]||null;prediction=active.prediction;
    if(follow&&pending)selected=pending.v;$('fd-account').value=String(selected);
    const tau=result.state.tau,decision=prediction?(tau===null?'LEARNING':prediction.score>tau?'BLOCK':'ALLOW'):'—';
    $('fd-step').max=String(data.events.length);$('fd-step').value=String(count);$('fd-position').textContent=count.toLocaleString()+' / '+data.events.length.toLocaleString()+' events processed';
    $('fd-back').disabled=count===0;$('fd-next').disabled=!pending;$('fd-run').disabled=!pending;
    $('fd-event-label').textContent=!pending?'Replay complete':pending.kind==='payment'?'Proposed request':pending.kind==='report'?'Incoming report':'Outside deposit';
    $('fd-event').textContent=pending?clock(pending.t)+' · '+desc(pending):'All scheduled events processed';
    const comparisonBody=$('fd-compare').querySelector('tbody');comparisonBody.replaceChildren();
    const scope=$('fd-policy').value,fit=result.state.policyFit;
    $('fd-policy-rate-heading').textContent=scope==='shared'?'Target α':'Validation / configured rate';
    for(const entry of entries){const state=entry.result.state,rate=state.policyFit?state.policyFit.impliedAlpha:null;row(comparisonBody,[entry.model.label+(entry.model.id===model.id?' · selected':''),currentPolicyLabel(state),entry.decision||'—',entry.prediction?entry.prediction.score.toFixed(2):'—',state.tau===null?'Learning':state.tau.toFixed(2),percent(rate??(state.decisionPolicy==='shared'?state.alpha:null))]);}
    if(scope==='shared')$('fd-policy-summary').textContent='Each model learns its own τ without outcome labels, targeting the same '+percent(result.state.alpha)+' budget.';
    else if(scope==='auto')$('fd-policy-summary').textContent='Every model is tuned independently for '+String($('fd-objective').value).toUpperCase()+' on held-out history. Each fitted τ and validation blocking rate is shown separately.';
    else {
      const suffix=result.state.decisionPolicy==='tuned'?' with this model’s own error costs. ':result.state.decisionPolicy==='manual'?' with this model’s fixed τ. ':result.state.decisionPolicy==='auto'?' with this model’s selected objective. ':' with this model’s own unlabeled budget. ';
      $('fd-policy-summary').textContent=model.label+' uses '+currentPolicyLabel(result.state)+suffix+'Other models retain their saved settings.';
    }
    const validation=model.policy_validation?.[result.state.trainingMode],provenance=bundle.policy.validation;
    $('fd-validation-source').textContent=validation?provenance.validationEpisodeIndices.length+' held-out historical episodes · '+validation.requests.toLocaleString()+' requests · '+validation.positives+' fraud flags · '+validation.negatives+' legitimate outcomes · '+validation.unknown+' unknown outcomes excluded from cost. Fitted rates describe this synthetic sample.':'No labelled validation sample available; shared-budget learning remains available.';
    const training=model.training||{},supervised=result.state.trainingMode==='supervised';
    const supTraining=model.supervised?.training||training;
    $('fd-method').textContent=model.architecture.description+' '+(model.parameters||0).toLocaleString()+' trained parameters.'+(supervised?' Supervised fraud head is active; flags used: '+(supTraining.fraud_labels_used??training.fraud_labels_used??0)+'.':' No fraud labels are used by the active score.');
    $('fd-training-source').textContent=supervised?(model.family==='xgboost'?'Offline XGBoost checkpoint: '+(training.fraud_labels_used??training.fraud_flags_requested??0)+' flagged events in its training partition; '+(training.validation_flags_used??0)+' held out.':'Offline logistic fraud head on this encoder: '+(supTraining.fraud_labels_used??0)+' flagged events in its training partition; '+(supTraining.validation_flags_used??0)+' held out.'):'No-label mode: fraud flags are ignored by the active score. Change offline flags with train.py --fraud-flags N, then rebuild.';
    $('fd-score-explanation').textContent=supervised?'Flagged-history mode: score = −log₂(1 − estimated flagged-fraud probability); it is an offline model score, not a calibrated production probability.':model.family==='xgboost'?'No-label mode: the displayed bits are a causal feature-rarity score; no fraud probability is used.':'No-label mode: bits = −log₂ likelihood of the observed recipient, amount bin, and timing bin; higher means less expected, not “probability of fraud”.';
    $('fd-graph-method').textContent=model.family==='xgboost'?'Selected model uses offline boosted trees over causal account/payment statistics; the network is shown as evidence, not a model input.':model.family==='temporal_family'?({dygformer:'Selected model: DyGFormer-style temporal history over eight recent account events with sinusoidal time features.',tami:'Selected model: TAMI-style directed pair history with six retained pair events and elapsed-time features.',dyg_tami:'Selected model: DyGFormer-style account history combined with TAMI-style directed pair history.',dyg_tami_gnn:'Selected model: DyGFormer history + TAMI pair state + two-hop time-respecting GNN readout.'}[model.variant]||'Selected model uses a temporal-history adapter.'):model.architecture.layers?'Selected model: '+model.architecture.layers+' graph layers · '+(model.architecture.readout==='mean'?'uniform mean':'learned attention')+' · 4 recent incidents per account.':model.hidden?'Selected model uses counterparty-aware GRU memory; it does not aggregate this neighborhood at readout.':'Neighborhood shown for inspection; this model uses account and pair statistics.';
    $('fd-memory-panels').hidden=!model.hidden;$('fd-no-memory').hidden=!!model.hidden;
    $('fd-settlement').textContent=!prediction?'No payment decision':decision==='BLOCK'?result.state.mode==='enforce'?'Transfer will not execute':'Would block; shadow replay':decision==='LEARNING'?'Allowed during calibration':'Transfer will execute';
    $('fd-calibration').textContent=result.state.decisionPolicy==='shared'?(tau===null?result.state.calibration.length+' / '+result.state.warmup+' calibration requests for this model':percent(result.state.alpha)+' target tail · separate τ per model'):fit?(currentPolicyLabel(result.state)+' fit: '+percent(fit.impliedAlpha)+' blocked · τ frozen'):currentPolicyLabel(result.state)+' = '+tau.toFixed(2)+' · τ frozen';
    $('fd-selected').textContent=name(selected)+' · '+result.state.inCount[selected]+' payment receipts · '+result.state.outCount[selected]+' completed sends';
    $('fd-reports').textContent=result.state.reports.length+' delayed confirmations received';
    cells('fd-sender-memory',prediction?.memory[0]);cells('fd-recipient-memory',prediction?.memory[1]);cells('fd-account-memory',result.state.memory[selected]);
    const parts=$('fd-parts').querySelector('tbody');parts.replaceChildren();
    if(prediction){
      const rows=supervised?[['Fraud-risk score','1 − estimated legitimate probability',prediction.parts[0].toFixed(2)],['Estimated flagged-fraud probability',(100*prediction.probability).toFixed(1)+'%',''],['Causal feature vector',(prediction.features?.length||0)+' statistics','']]:model.family==='xgboost'?[['No-label rarity score','Causal feature rarity',prediction.parts[0].toFixed(2)],['Flagged-fraud probability','Not used in this mode',''],['Causal feature vector',prediction.features.length+' statistics','']]:[['Recipient',name(pending.v),prediction.parts[0].toFixed(2)],['Amount bin',money(pending.amount)+' · bin '+(prediction.buckets[1]+1),prediction.parts[1].toFixed(2)],['Sender activity gap',prediction.gap===null?'First observed activity':prediction.gap.toFixed(1)+' min',prediction.parts[2].toFixed(2)]];
      rows.forEach(r=>row(parts,r));
    }
    const body=$('fd-history').querySelector('tbody');body.replaceChildren();result.state.decisions.filter(r=>r.event.u===selected||r.event.v===selected).slice(-5).reverse().forEach(r=>row(body,[r.event.id+' · '+desc(r.event),r.decision,r.score.toFixed(1)+' / '+(r.tauBefore===null?'learning':r.tauBefore.toFixed(1))]));
    graph();chart();evaluate();root.dataset.count=String(count);root.dataset.decision=decision;
  }
  for(const id of['fd-scenario','fd-size','fd-seed'])$(id).addEventListener('change',()=>rebuild(false));
  $('fd-model').addEventListener('change',()=>{stop();model=models.find(m=>m.id===$('fd-model').value);syncPolicyControls();return render(true);});
  for(const id of['fd-mode','fd-alpha','fd-warmup','fd-training-mode','fd-policy','fd-objective'])$(id).addEventListener('change',()=>rebuild(true));
  for(const id of['fd-model-strategy','fd-model-alpha','fd-model-warmup','fd-model-tau','fd-model-false-cost','fd-model-missed-cost','fd-model-objective'])$(id).addEventListener('change',()=>{readPolicyConfig();rebuild(true);});
  for(const id of['fd-delay','fd-forward'])$(id).addEventListener('change',()=>rebuild(false));
  $('fd-account').addEventListener('change',()=>{selected=Number($('fd-account').value);follow=false;return render(true);});
  $('fd-jump').addEventListener('change',()=>{if($('fd-jump').value==='')return;stop();count=Number($('fd-jump').value);follow=true;render();});
  $('fd-step').addEventListener('input',()=>{stop();count=Number($('fd-step').value);follow=true;render();});
  $('fd-back').addEventListener('click',()=>{stop();count=Math.max(0,count-1);follow=true;render();});
  $('fd-next').addEventListener('click',()=>{stop();count=Math.min(data.events.length,count+1);follow=true;render();});
  $('fd-run').addEventListener('click',()=>{stop();count=data.events.length;render();});
  $('fd-truth').addEventListener('change',evaluate);
  $('fd-play').addEventListener('click',()=>{if(timer){stop();return;}if(count===data.events.length)count=0;$('fd-play').textContent='Pause';timer=setInterval(()=>{if(busy)return;count++;follow=true;render();if(count>=data.events.length)stop();},800);});
  new ResizeObserver(()=>{if(!busy){graph();chart();}}).observe($('fd-graph-container'));
  models.forEach(m=>option($('fd-model'),m.id,m.label));$('fd-model').value=model.id;
  rebuild();
  root.demo={whenIdle,getPerformance:()=>({busy,comparisons:comparisonCache.entries.size,inferenceCalls:Array.from(group.runners.values()).reduce((sum,r)=>sum+r.inferenceCalls,0),...group.lastTiming}),getSnapshot:()=>({busy,count,selected,model:model.id,policyScope:$('fd-policy').value,modelPolicies:JSON.parse(JSON.stringify(modelPolicies)),mode:result?.state.mode,trainingMode:result?.state.trainingMode,decisionPolicy:result?.state.decisionPolicy,policyFit:result?.state.policyFit,rankingBudget:group.options.alpha,warmup:result?.state.warmup,comparison:entries.map(e=>({id:e.model.id,policy:currentPolicyLabel(e.result.state),score:e.prediction?.score??null,tau:e.result.state.tau,impliedAlpha:e.result.state.policyFit?.impliedAlpha??null,decision:e.decision,payments:e.result.state.payments.map(x=>x.id)})),decision:root.dataset.decision,score:prediction?.score??null,tau:result?.state.tau,accounts:data.accounts.length,events:data.events.length,processed:result?.state.events.map(e=>e.id)||[],decisions:result?.state.decisions.map(r=>({id:r.event.id,score:r.score,tau:r.tauBefore,decision:r.decision,settled:r.settled}))||[]})};
})();
