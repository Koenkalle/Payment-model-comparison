(function(){
  'use strict';
  const root=document.getElementById('fraud-memory-demo'),$=id=>root.querySelector('#'+id),core=globalThis.FraudCore,sc=globalThis.FraudScenarios;
  const bundle=JSON.parse($('fd-model-data').textContent),baseModels=bundle.models,ns='http://www.w3.org/2000/svg',compare=globalThis.FraudComparison;
  const nativeDefinitions=bundle.native_models||[],definitions=[...baseModels,...nativeDefinitions];
  const nativeClient=new globalThis.FraudNativeClient.Client();
  let models=baseModels.slice(),availableNative=[],nativeError=null,rebuildVersion=0,requestController=null,preparingNative=false;
  let model=models.find(m=>m.id===bundle.default)||models[0];
  const currencyFormat=new Intl.NumberFormat('en-IE',{style:'currency',currency:'EUR',maximumFractionDigits:0}),money=x=>currencyFormat.format(x);
  const clock=t=>'D'+(Math.floor(t/1440)+1)+' '+String(Math.floor(t/60)%24).padStart(2,'0')+':'+String(Math.floor(t)%60).padStart(2,'0');
  const percent=x=>x===null?'—':(100*x).toFixed(1)+'%';
  const tailPercent=x=>(100*x).toLocaleString('en',{maximumSignificantDigits:4})+'%';
  let importedData=null,datasetImportVersion=0;const pendingImports=new Set();
  let data,group,entries=[],count=0,selected=0,timer=null,result,pending,prediction,follow=true;
  let comparisonCache=new compare.ComparisonCache(models,3);const datasets=new Map();
  let renderVersion=0,latestTask=Promise.resolve(),busy=false,paintedGroup=null,paintedCount=-1;
  const defaultWarmup=bundle.policy?.warmup||128;
  const defaultsForMode=trainingMode=>Object.fromEntries(definitions.map(m=>{
    const candidates=m.policy_validation?.[trainingMode]?.candidates||[],middle=candidates.length?candidates[Math.floor(candidates.length/2)][0]:10;
    return [m.id,{strategy:'shared',alpha:.02,warmup:defaultWarmup,manualTau:m.comparison?(trainingMode==='supervised'?1:-Math.log2(.02)):Number(middle)||10,falseBlockCost:1,missedFraudCost:20,objective:'f1',predictionHead:m.comparison?(trainingMode==='supervised'?'fraud_linear':'empirical_tail'):'default'}];
  }));
  const policiesByMode={unsupervised:defaultsForMode('unsupervised'),supervised:defaultsForMode('supervised')};
  let modelPolicies=policiesByMode.unsupervised;
  const currentTrainingMode=()=>$('fd-training-mode').value==='supervised'?'supervised':'unsupervised';
  const nativeCapability=(id,mode=currentTrainingMode())=>globalThis.FraudNativeClient.modeCapabilities(nativeClient.models.find(m=>m.id===id),mode);
  const headLabel=id=>globalThis.FraudPredictionHeads.labels[id]||(id==='fraud_linear'?'Linear fraud classifier':id);
  function checkpointIdentity(checkpoint){
    const artifact=checkpoint.artifact,native=!!checkpoint.native_run;
    const hash=artifact?.model_sha256||checkpoint.native_run?.provenance?.artifact_model_sha256||checkpoint.checkpoint_id?.split(':')[0]||'Unavailable';
    const source=native?(artifact?.source||'unknown'):'browser';
    const label={custom:'Custom artifact',default:'Default artifact location',browser:'Bundled browser checkpoint',unknown:'Artifact source unavailable'}[source]||'Artifact source unavailable';
    return {hash,source,label,path:artifact?.path||(native?'Unavailable':checkpoint.implementation?.checkpoint||'Embedded in this page')};
  }
  function checkpointSummary(checkpoint){
    const identity=checkpointIdentity(checkpoint),name=identity.path.split(/[\\/]/).pop();
    return identity.label+(checkpoint.native_run&&checkpoint.artifact?' · '+name:'')+' · '+identity.hash.slice(0,12);
  }
  function showCheckpoint(checkpoint,trainingMode){
    const identity=checkpointIdentity(checkpoint),artifact=checkpoint.artifact,parts=[trainingMode==='supervised'?'Fraud-label training':'No-label scoring'];
    if(artifact?.encoder_training)parts.push(artifact.encoder_training==='finetune'?'Encoder fine-tuned':'Encoder frozen');
    if(artifact?.best_epoch)parts.push('Best epoch '+artifact.best_epoch+' of '+artifact.epochs_completed+' completed');
    if(artifact?.training_dataset)parts.push('Training data: '+artifact.training_dataset);
    $('fd-checkpoint').dataset.source=identity.source;
    $('fd-checkpoint-details').hidden=false;
    $('fd-checkpoint-name').textContent='Selected model · '+checkpoint.label;
    $('fd-checkpoint-source').textContent=identity.label;
    $('fd-checkpoint-training').textContent=parts.join(' · ');
    $('fd-checkpoint-path').textContent=identity.path;
    $('fd-checkpoint-id').textContent=identity.hash;
    $('fd-checkpoint-help').textContent=checkpoint.native_run?'This identifies the loaded weights used for these results. After retraining, restart serve.py with '+(trainingMode==='supervised'?'--supervised-artifact':'--artifact')+' pointing to your output directory, then reload this page.':'This checkpoint is embedded in the page at build time. Each comparison row shows its own checkpoint; select a model to inspect its full identity.';
    const parameters=checkpoint.native_run?.provenance?.model_parameters||{};
    $('fd-native-settings').hidden=!Object.keys(parameters).length;
    $('fd-native-parameters').replaceChildren();
    if(Object.keys(parameters).length){
      const fields=[...Object.entries(globalThis.PaymentInfoMetadata?.nativeParameters||{}).map(([name,definition])=>({name,...definition})),...(globalThis.PaymentInfoMetadata?.modelFields('dyg_tami_native')||[])];
      globalThis.PaymentPipelineUI?.parameterList?.($('fd-native-parameters'),parameters,fields);
    }
    globalThis.PaymentInfo?.attach($('fd-checkpoint-training'),{title:checkpoint.label+' · saved training',description:checkpoint.native_run?globalThis.PaymentInfoMetadata.explanations.native.text:'This browser checkpoint is embedded in the page. Replay settings select its saved scoring behavior without changing its trained weights.',facts:[['Training mode',trainingMode],['Trained parameters',checkpoint.parameters??'Unavailable'],['Checkpoint',identity.hash],...(checkpoint.learning_rate?[['Tree learning rate',checkpoint.learning_rate]]:[])]});
  }
  function validNativeSettings(trainingMode){
    for(const definition of nativeDefinitions){
      const capability=nativeCapability(definition.id,trainingMode),p=modelPolicies[definition.id],heads=capability.prediction_heads||[];
      if(heads.length&&!heads.includes(p.predictionHead))p.predictionHead=capability.default_head||heads[0];
      if(capability.available&&!capability.decision_policies?.includes(p.strategy))p.strategy='shared';
    }
  }
  const name=n=>data.accounts[n]?.name||'Outside';
  const desc=e=>e.kind==='report'?'Fraud confirmation · '+e.reference:e.kind==='deposit'?'Outside → '+name(e.v)+' · '+money(e.amount):name(e.u)+' → '+name(e.v)+' · '+money(e.amount);
  function node(tag,attrs={},parent,text=''){const e=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);e.textContent=text;if(parent)parent.appendChild(e);return e;}
  function option(parent,value,text){const e=document.createElement('option');e.value=String(value);e.textContent=text;parent.appendChild(e);}
  function stop(){if(timer)clearInterval(timer);timer=null;$('fd-play').textContent='Play';}
  function numberValue(id,fallback,min,max){const value=Number($(id).value);return Number.isFinite(value)?Math.max(min,Math.min(max,value)):fallback;}
  function readPolicyConfig(){
    const p=modelPolicies[model.id];
    p.strategy=$('fd-model-strategy').value;
    p.alpha=numberValue('fd-model-alpha',.02,.001,.5);p.warmup=Math.floor(numberValue('fd-model-warmup',defaultWarmup,1,10000));
    p.manualTau=numberValue('fd-model-tau',p.manualTau??10,0,100000);p.falseBlockCost=numberValue('fd-model-false-cost',1,.01,10000);p.missedFraudCost=numberValue('fd-model-missed-cost',20,.01,10000);p.objective=$('fd-model-objective').value;
    modelPolicies[model.id]=p;return p;
  }
  function syncPolicyControls(){
    const p=modelPolicies[model.id],strategy=p.strategy;
    $('fd-model-strategy').value=strategy;$('fd-model-alpha').value=String(p.alpha);$('fd-model-warmup').value=String(p.warmup);$('fd-model-tau').value=String(p.manualTau);$('fd-model-false-cost').value=String(p.falseBlockCost);$('fd-model-missed-cost').value=String(p.missedFraudCost);$('fd-model-objective').value=p.objective;
    $('fd-model-alpha').disabled=strategy!=='shared';$('fd-model-warmup').disabled=strategy!=='shared';$('fd-model-tau').disabled=strategy!=='manual';$('fd-model-false-cost').disabled=strategy!=='tuned';$('fd-model-missed-cost').disabled=strategy!=='tuned';$('fd-model-objective').disabled=strategy!=='auto';
    $('fd-model-strategy').disabled=false;
    const descriptor=nativeDefinitions.find(m=>m.id===model.id),capability=descriptor?nativeCapability(model.id):null,heads=capability?.prediction_heads||[];
    for(const id of ['shared','manual','tuned','auto'])$('fd-model-strategy').querySelector('option[value="'+id+'"]').disabled=capability?!capability.decision_policies?.includes(id):(id==='tuned'||id==='auto')&&!model.policy_validation?.[currentTrainingMode()];
    $('fd-model-head').replaceChildren();
    if(heads.length)heads.forEach(id=>option($('fd-model-head'),id,headLabel(id)));
    else option($('fd-model-head'),'default','Model default');
    $('fd-model-head').value=p.predictionHead;$('fd-model-head').disabled=!heads.length;
  }
  function modelPolicyOptions(trainingMode){
    return Object.fromEntries(definitions.map(m=>{
      const p=modelPolicies[m.id],fits=m.comparison?nativeCapability(m.id,trainingMode).decision_policies?.includes(p.strategy):!!m.policy_validation?.[trainingMode],strategy=(p.strategy==='tuned'||p.strategy==='auto')&&!fits?'shared':p.strategy;
      return [m.id,{decisionPolicy:strategy,alpha:p.alpha,warmup:p.warmup,manualTau:p.manualTau,falseBlockCost:p.falseBlockCost,missedFraudCost:p.missedFraudCost,objective:p.objective,...(m.comparison?{predictionHead:p.predictionHead}:{})}];
    }));
  }
  function policyOptions(trainingMode,warmup){
    const scope=$('fd-policy').value,options={alpha:numberValue('fd-alpha',.02,.001,.5),mode:$('fd-mode').value,trainingMode,decisionPolicy:'shared',warmup};
    if(scope==='individual')options.modelPolicies=modelPolicyOptions(trainingMode);
    if(scope==='auto')options.modelPolicies=Object.fromEntries(definitions.map(m=>[m.id,{decisionPolicy:'auto',objective:$('fd-objective').value,...(m.comparison?{predictionHead:modelPolicies[m.id].predictionHead}:{})}]));
    for(const m of nativeDefinitions){options.modelPolicies??={};options.modelPolicies[m.id]={...options.modelPolicies[m.id],predictionHead:modelPolicies[m.id].predictionHead};}
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
  function row(parent,values){const tr=document.createElement('tr');values.forEach(v=>{const td=document.createElement('td');td.textContent=String(v);tr.appendChild(td);});parent.appendChild(tr);return tr;}
  function populateModels(){
    const selectedId=$('fd-model').value||model.id;
    model=models.find(m=>m.id===selectedId)||models.find(m=>m.id===bundle.default)||models[0];
    $('fd-model').replaceChildren();
    for(const definition of definitions){
      option($('fd-model'),definition.id,definition.label);
      const item=$('fd-model').lastElementChild;
      if(item){item.disabled=!models.some(m=>m.id===definition.id);if(item.disabled)item.title=nativeError||nativeClient.error||nativeCapability(definition.id).error||'Python model is not yet available.';}
    }
    $('fd-model').value=model.id;
  }
  function activateModels(checkpoints){
    const next=[...baseModels,...checkpoints];
    if(JSON.stringify(next.map(m=>m.checkpoint_id||m.id))!==JSON.stringify(models.map(m=>m.checkpoint_id||m.id)))comparisonCache=new compare.ComparisonCache(next,3);
    models=next;populateModels();
  }
  function runtimeStatus(trainingMode){
    const omitted=availableNative.filter(m=>!nativeCapability(m.id,trainingMode).available);
    $('fd-runtime-status').textContent=nativeError||nativeClient.error||(omitted.length?omitted.map(m=>m.label+': '+nativeCapability(m.id,trainingMode).error).join(' ')+' Compatible models remain in the comparison.':availableNative.length?'All '+models.length+' models use the selected dataset and settings.':'');
  }
  function rebuild(keep=false){
    const revision=++rebuildVersion;requestController?.abort();renderVersion++;preparingNative=false;nativeError=null;
    stop();const seed=Math.max(1,Math.min(99999,Math.floor(Number($('fd-seed').value)||42)));$('fd-seed').value=String(seed);
    group?.cancel();
    const datasetKey=JSON.stringify([$('fd-scenario').value,$('fd-size').value,seed,Number($('fd-delay').value)*60,Number($('fd-forward').value)]);
    if(importedData)data=importedData;
    else if(datasets.has(datasetKey)){data=datasets.get(datasetKey);datasets.delete(datasetKey);}
    else data=globalThis.FraudDatasets.load('synthetic_payments',{name:$('fd-scenario').value,size:$('fd-size').value,seed,reportDelay:Number($('fd-delay').value)*60,forwardDelay:Number($('fd-forward').value)});
    if(!importedData){datasets.set(datasetKey,data);if(datasets.size>2)datasets.delete(datasets.keys().next().value);}
    const warmup=Math.max(1,Math.floor(Number($('fd-warmup').value)||bundle.policy?.warmup||128));$('fd-warmup').value=String(warmup);
    const trainingMode=currentTrainingMode();modelPolicies=policiesByMode[trainingMode];validNativeSettings(trainingMode);
    const wanted=availableNative.filter(m=>m.available&&nativeCapability(m.id,trainingMode).available);
    const canTune=baseModels.every(m=>m.policy_validation?.[trainingMode])&&wanted.every(m=>nativeCapability(m.id,trainingMode).decision_policies.includes('auto'));$('fd-policy').querySelector('option[value="auto"]').disabled=!canTune;
    if(!canTune&&$('fd-policy').value==='auto')$('fd-policy').value='shared';
    const scope=$('fd-policy').value;
    $('fd-auto-controls').hidden=scope!=='auto';$('fd-individual-controls').hidden=scope!=='individual';
    $('fd-warmup').disabled=scope!=='shared';$('fd-alpha').disabled=false;
    $('fd-alpha-label').textContent=scope==='shared'?'Shared target α':'Evaluation ranking budget α';
    const settings=policyOptions(trainingMode,warmup),requestedData=data;
    count=keep?Math.min(count,data.events.length):data.startIndex;follow=true;selected=data.events[count]?.v??0;
    $('fd-account').replaceChildren();data.accounts.forEach(a=>option($('fd-account'),a.id,'#'+(a.id+1)+' · '+a.name));
    $('fd-jump').replaceChildren();option($('fd-jump'),'','Choose an event…');option($('fd-jump'),0,scope==='shared'?'Start · learn each cutoff':'Start · use configured cutoffs');
    for(const b of data.bookmarks)option($('fd-jump'),data.events.findIndex(e=>e.id===b.id),b.label);
    const report=data.events.findIndex(e=>e.kind==='report');if(report>=0)option($('fd-jump'),report,'First delayed confirmation');option($('fd-jump'),data.events.length,'End of scenario');
    $('fd-forward-value').textContent=$('fd-forward').value+' min';$('fd-delay-value').textContent=$('fd-delay').value+' h';
    $('fd-scope').textContent=data.accounts.length+' accounts · '+data.events.length.toLocaleString()+' events · '+Math.ceil(data.events.at(-1).t/1440)+' days';
    $('fd-mode-label').textContent=($('fd-mode').value==='enforce'?'Separate histories after blocks':'Identical history for all models')+' · '+(trainingMode==='supervised'?'flagged-history mode':'no-label mode');
    const finish=checkpoints=>{if(revision!==rebuildVersion)return;preparingNative=false;activateModels(checkpoints);group=comparisonCache.acquire(requestedData,settings);runtimeStatus(trainingMode);syncPolicyControls();return render();};
    if(!wanted.length)return finish([]);
    preparingNative=true;setBusy(true,'Scoring the selected dataset with Python models…');requestController=new AbortController();const signal=requestController.signal;
    latestTask=(async()=>{
      try{
        const {modelPolicies:perModel,...shared}=settings;
        const checkpoints=await Promise.all(wanted.map(m=>nativeClient.predict(m,requestedData,{...shared,...perModel?.[m.id]},signal)));
        if(revision!==rebuildVersion)return;
        return finish(checkpoints);
      }catch(error){
        if(revision!==rebuildVersion||error.name==='AbortError')return;
        nativeError='Python model could not update: '+error.message+' Change a setting or recompute to retry.';
        return finish([]);
      }
    })();return latestTask;
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
    const population=group.evaluationRecords(model.id).filter(r=>r.decision!=='LEARNING'&&r.decision!=='CONTEXT');
    const d=population.filter(r=>Object.prototype.hasOwnProperty.call(data.truth,r.event.id)),fraud=d.filter(r=>data.truth[r.event.id]),legit=d.filter(r=>!data.truth[r.event.id]);
    const tp=fraud.filter(r=>r.decision==='BLOCK').length,fp=legit.filter(r=>r.decision==='BLOCK').length;
    const comparisonBody=$('fd-model-metrics').querySelector('tbody');comparisonBody.replaceChildren();
    let budget=0;
    for(const entry of entries){const state=entry.result.state,m=compare.metrics(group.evaluationRecords(entry.model.id),data.truth,group.options.alpha,state.errorCosts);budget=m.budget;
      row(comparisonBody,[entry.model.label,checkpointSummary(entry.model),m.tp+' / '+(m.tp+m.fn),m.fp,percent(m.precision),percent(m.blockRate),state.decisionPolicy==='tuned'?Number(m.errorCost.toFixed(2)):'—',percent(m.recallAtBudget)]);
    }
    $('fd-evaluation-protocol').textContent=(result.state.mode==='shadow'?'Matched history':'Independent blocking histories')+' · all models use the same evaluation requests, excluding every model’s warm-up · ranking uses a common '+percent(group.options.alpha)+' budget ('+budget+' requests). Unknown outcomes are excluded from classification. Selected dataset outcomes do not tune the head or cutoff.';
    const body=$('fd-metrics').querySelector('tbody');body.replaceChildren();
    [['Evaluated requests',population.length],['Known outcomes',d.length],['Unknown outcomes excluded from classification',population.length-d.length],['Fraud blocked / all known fraud',tp+' / '+fraud.length],['Legitimate requests blocked / all known legitimate',fp+' / '+legit.length],['Precision among known block decisions',tp+fp?(100*tp/(tp+fp)).toFixed(1)+'%':'—'],['Context and common warm-up requests excluded',result.state.decisions.length-population.length],['Historical outcome labels used to fit τ',result.state.policyFit?result.state.policyFit.positives+result.state.policyFit.negatives:0],['Replay outcome labels used to change model or τ','0']].forEach(r=>row(body,r));
    const errors=$('fd-errors').querySelector('tbody');errors.replaceChildren();
    d.filter(r=>(r.decision==='BLOCK')!==!!data.truth[r.event.id]).slice(-6).reverse().forEach(r=>row(errors,[r.event.id+' · '+desc(r.event),data.truth[r.event.id]?'Fraud':'Legitimate',r.decision]));
  }
  function setBusy(value,message='Updating results…'){
    busy=value;root.dataset.busy=String(value);$('fd-work-status').textContent=value?message:'';
    $('fd-compare').setAttribute('aria-busy',String(value));
    $('fd-checkpoint').setAttribute('aria-busy',String(value));
    $('fd-checkpoint-status').hidden=!value;
    if(value){$('fd-checkpoint-details').hidden=true;$('fd-checkpoint-name').textContent='Updating selected model…';$('fd-checkpoint-source').textContent='Pending';}
    $('fd-back').disabled=value||count===0;$('fd-next').disabled=value||count>=data.events.length;$('fd-run').disabled=value||count>=data.events.length;
  }
  function render(viewOnly=false){
    if(preparingNative)return latestTask;
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
  async function whenIdle(){for(;;){const task=latestTask;await Promise.all([task,...pendingImports]);if(task===latestTask&&!pendingImports.size)return;}}
  function paint(){
    const active=entries.find(e=>e.model.id===model.id);result=active.result;pending=data.events[count]||null;prediction=active.prediction;
    showCheckpoint(active.model,result.state.trainingMode);
    if(follow&&pending)selected=pending.v;$('fd-account').value=String(selected);
    const tau=result.state.tau,decision=prediction?(prediction.evaluationEligible===false?'CONTEXT':tau===null?'LEARNING':prediction.score>tau?'BLOCK':'ALLOW'):'—';
    $('fd-step').max=String(data.events.length);$('fd-step').value=String(count);$('fd-position').textContent=count.toLocaleString()+' / '+data.events.length.toLocaleString()+' events processed';
    $('fd-back').disabled=count===0;$('fd-next').disabled=!pending;$('fd-run').disabled=!pending;
    $('fd-event-label').textContent=!pending?'Replay complete':pending.kind==='payment'?'Proposed request':pending.kind==='report'?'Incoming report':'Outside deposit';
    $('fd-event').textContent=pending?clock(pending.t)+' · '+desc(pending):'All scheduled events processed';
    const comparisonBody=$('fd-compare').querySelector('tbody');comparisonBody.replaceChildren();
    const scope=$('fd-policy').value,fit=result.state.policyFit;
    $('fd-policy-rate-heading').textContent=scope==='shared'?'Target α':'Validation / configured rate';
    for(const entry of entries){const state=entry.result.state,rate=state.policyFit?state.policyFit.impliedAlpha:null,tr=row(comparisonBody,[entry.model.label+(entry.model.id===model.id?' · selected':''),checkpointSummary(entry.model),currentPolicyLabel(state),entry.decision||'—',entry.prediction?entry.prediction.score.toFixed(2):'—',state.tau===null?'Learning':state.tau.toFixed(2),percent(rate??(state.decisionPolicy==='shared'||state.predictionHead?state.alpha:null))]);
      for(const [index,key]of [[2,'decisionPolicy'],[5,'manualTau'],[6,'alpha']])globalThis.PaymentInfo?.attach(tr.children[index],{...globalThis.PaymentInfo.parameter(globalThis.PaymentInfoMetadata.parameters[key]),facts:[['Model',entry.model.label],['Policy',currentPolicyLabel(state)],['Current cutoff',state.tau??'Learning'],['Target α',state.alpha],...(rate===null?[]:[['Historical fitted block rate',percent(rate)]])]});
    }
    if(scope==='shared')$('fd-policy-summary').textContent='Each model learns its own τ without outcome labels, targeting the same '+percent(group.options.alpha)+' budget.';
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
    if(model.native_run){
      const native=model.native_run,reference=native.calibration.count;
      $('fd-method').textContent='Native DyGFormer + TAMI · '+headLabel(native.head.id)+' · '+currentPolicyLabel(result.state)+'.';
      if(supervised){
        const encoder=model.training?.encoder_training||native.provenance.encoder_training||native.provenance.model_parameters?.encoder_training;
        const learned=encoder==='finetune'?'The temporal encoder and fraud classifier were fine-tuned on historical fraud labels.':encoder==='frozen'?'The fraud classifier was trained on historical fraud labels with the temporal encoder held fixed.':'The native fraud classifier was trained on historical fraud labels.';
        const counts=model.training?.label_counts?.train,labels=counts?' Training outcomes: '+counts.fraud+' fraud, '+counts.legitimate+' legitimate, '+counts.unknown+' unknown context payments.':'';
        $('fd-training-source').textContent=learned+labels+' The saved weights and training normalization stay fixed when comparison data changes.';
        $('fd-validation-source').textContent=validation?'Cost and automatic cutoffs use the reserved policy-validation window, separate from model training and epoch selection. Selected dataset outcomes are used only for evaluation.':'The reserved policy-validation window lacks both known classes, so cost and automatic tuning are unavailable. Shared and fixed cutoffs remain available.';
        $('fd-score-explanation').textContent='Score = softplus(fraud logit) / ln(2) = −log₂(1 − estimated fraud probability). Fixed τ = 1 blocks probabilities above 50%; shared α is a blocking target. These estimates are not claimed to be calibrated probabilities.';
        $('fd-graph-method').textContent='Native DyGFormer + TAMI uses strictly earlier '+(result.state.mode==='enforce'?'allowed payments':'observed payment attempts')+', previous directed-pair memory and the proposed amount. The amount is inspected before settlement; deposits, reports and outcome labels are not scoring inputs.';
        $('fd-no-memory').textContent='Transformer embeddings, pair memory and the trained fraud head are computed by Python. Inspect the fraud logit and estimated probability below.';
      }else{
        $('fd-training-source').textContent='Frozen PyTorch weights trained on separate historical payments. '+reference+' historical reference transactions; the selected dataset does not retrain the encoder or head.';
        $('fd-validation-source').textContent='Cost tuning and automatic objectives use labeled historical transactions after the reference period. Selected dataset outcomes are used only for evaluation.';
        $('fd-score-explanation').textContent=(native.head.id==='empirical_tail'?'Score = −log₂ historical likelihood rank. The frozen reference ranks this transaction’s link logit.':'Score = −log₂ native link likelihood.')+' The selected policy sets τ in the same way as for the other models. Higher scores indicate unusual transactions, not fraud probabilities.';
        $('fd-graph-method').textContent='Native DyGFormer + TAMI uses strictly earlier '+(result.state.mode==='enforce'?'allowed payments':'observed payment attempts')+' and directed-pair memory. Deposits/reports and the proposed amount are not inputs to its current score.';
        $('fd-no-memory').textContent='Transformer embeddings and pair memory are computed by the Python model. Inspect the link likelihood and prediction head below.';
      }
      if(decision==='CONTEXT')$('fd-settlement').textContent='Historical context only; excluded from evaluation.';
      else if(decision==='BLOCK')$('fd-settlement').textContent='Suspected fraud · '+(result.state.mode==='enforce'?'payment blocked; later graph history excludes it':'would block; shadow comparison');
      if(!supervised)$('fd-calibration').textContent=reference+' historical head references · '+$('fd-calibration').textContent;
    }else $('fd-no-memory').textContent='This design has no learned account memory; inspect its causal statistics, temporal history, pair state, or tree features instead.';
    $('fd-selected').textContent=name(selected)+' · '+result.state.inCount[selected]+' payment receipts · '+result.state.outCount[selected]+' completed sends';
    $('fd-reports').textContent=result.state.reports.length+' delayed confirmations received';
    cells('fd-sender-memory',prediction?.memory[0]);cells('fd-recipient-memory',prediction?.memory[1]);cells('fd-account-memory',result.state.memory[selected]);
    const parts=$('fd-parts').querySelector('tbody');parts.replaceChildren();
    if(prediction&&model.native_run){
      const evidence=prediction.evidence;
      const rows=evidence.kind==='native-fraud'?[['Native fraud logit',evidence.fraud_logit.toFixed(5),''],['Estimated fraud probability',percent(evidence.fraud_probability),''],['Fraud score','−log₂(1 − estimated fraud probability)',prediction.score.toFixed(3)],['Prediction head',headLabel(evidence.head),'']]:[['Native link logit',evidence.logit.toFixed(5),''],['Native link likelihood',percent(evidence.likelihood_probability),''],['Likelihood / historical rank',percent(evidence.tail_probability),prediction.score.toFixed(3)],['Head reference',evidence.reference_count+' frozen reference transactions','']];
      rows.push(['Prospective evaluation',prediction.evaluationEligible?(model.native_run.provenance.same_dataset?'Held-out transaction':'Evaluation transaction'):'Context only','']);rows.forEach(r=>row(parts,r));
    }else if(prediction){
      const rows=supervised?[['Fraud-risk score','1 − estimated legitimate probability',prediction.parts[0].toFixed(2)],['Estimated flagged-fraud probability',(100*prediction.probability).toFixed(1)+'%',''],['Causal feature vector',(prediction.features?.length||0)+' statistics','']]:model.family==='xgboost'?[['No-label rarity score','Causal feature rarity',prediction.parts[0].toFixed(2)],['Flagged-fraud probability','Not used in this mode',''],['Causal feature vector',prediction.features.length+' statistics','']]:[['Recipient',name(pending.v),prediction.parts[0].toFixed(2)],['Amount bin',money(pending.amount)+' · bin '+(prediction.buckets[1]+1),prediction.parts[1].toFixed(2)],['Sender activity gap',prediction.gap===null?'First observed activity':prediction.gap.toFixed(1)+' min',prediction.parts[2].toFixed(2)]];
      rows.forEach(r=>{const tr=row(parts,r);if(r[0]==='Causal feature vector'&&model.family==='xgboost'&&globalThis.PaymentPipelineUI?.featureList){
        const details=document.createElement('details'),summary=document.createElement('summary'),list=document.createElement('div');summary.textContent='Inspect checkpoint features';details.append(summary,list);tr.children[1].appendChild(details);
        const definitions=globalThis.FraudXGBoost?.featureDefinitions||[];globalThis.PaymentPipelineUI?.featureList?.(list,definitions.map(definition=>definition.id),definitions);
      }});
    }
    for(const tr of parts.children){
      const label=tr.children[0]?.textContent,readout=globalThis.PaymentInfoMetadata?.readouts[label];
      if(readout){
        tr.children[0].dataset.infoReadout=label;
        const categorical=label==='Amount bin'?[['Amount boundaries (EUR)',model.amount_bins],['Amount category (zero-based)',prediction.buckets[1]],['Displayed bin number',prediction.buckets[1]+1]]:label==='Sender activity gap'?[['Gap boundaries (minutes)',model.gap_bins],['Timing category (zero-based)',prediction.buckets[2]],['First-activity category',model.gap_bins.length+1],['Elapsed gap (minutes)',prediction.gap===null?'No earlier activity':prediction.gap]]:[];
        globalThis.PaymentInfo?.attach(tr.children[0],{...readout,facts:[['Displayed value',tr.children[1]?.textContent],['Score contribution',tr.children[2]?.textContent||'Not separately reported'],...categorical]});
      }
    }
    const body=$('fd-history').querySelector('tbody');body.replaceChildren();result.state.decisions.filter(r=>r.event.u===selected||r.event.v===selected).slice(-5).reverse().forEach(r=>row(body,[r.event.id+' · '+desc(r.event),r.decision,r.score.toFixed(1)+' / '+(r.tauBefore===null?'learning':r.tauBefore.toFixed(1))]));
    graph();chart();evaluate();root.dataset.count=String(count);root.dataset.decision=decision;
  }
  function datasetStatus(next){
    $('fd-dataset-status').textContent=next?'Imported '+next.name+' · '+next.events.length+' events. Models use saved checkpoints; importing does not retrain them.':'Synthetic dataset. Models use their saved training histories.';
    if(next){
      const payments=next.events.filter(event=>event.kind==='payment').length,known=Object.keys(next.truth).length;
      $('fd-dataset-status').textContent+=' '+known+' labeled payments; '+(payments-known)+' unknown.';
      if(Number($('fd-warmup').value)>=payments){
        const suggested=[1,4,8,16,32,64,128,256,512].filter(value=>value<=Math.max(1,payments/5)).pop();
        $('fd-dataset-status').textContent+=payments>1?' Shared warm-up covers this dataset. In Threshold and replay settings, set shared warm-up to '+suggested+' to evaluate later payments.':'More than one payment is needed for evaluation after warm-up.';
      }
    }
  }
  function setDataset(document){
    const next=document===null?null:globalThis.FraudDatasets.load('payment_json',document);
    datasetImportVersion++;importedData=next;stop();
    for(const id of ['fd-scenario','fd-size','fd-seed','fd-forward','fd-delay'])$(id).disabled=!!next;
    datasetStatus(next);return rebuild(false);
  }
  function importDataset(read,message){
    const revision=++datasetImportVersion;$('fd-dataset-status').textContent=message;
    let activeRevision=revision;
    const task=(async()=>{try{const document=await read();if(revision===datasetImportVersion){const update=setDataset(document);activeRevision=datasetImportVersion;await update;}}catch(error){if(activeRevision===datasetImportVersion)$('fd-dataset-status').textContent='Dataset not loaded: '+error.message;}})();
    pendingImports.add(task);task.finally(()=>pendingImports.delete(task));return task;
  }
  const datasetControls=globalThis.FraudDatasetControls.mount(root,{importDataset,invalidate:()=>{datasetImportVersion++;}});
  globalThis.addEventListener?.('payment-info-model-schemas',()=>{if(model&&!busy&&!preparingNative)showCheckpoint(model,currentTrainingMode());});
  for(const [id,key]of Object.entries({'fd-scenario':'scenario','fd-size':'size','fd-seed':'seed','fd-model':'model','fd-model-head':'predictionHead','fd-training-mode':'trainingMode','fd-policy':'policyScope','fd-mode':'mode','fd-objective':'objective','fd-model-strategy':'decisionPolicy','fd-model-alpha':'alpha','fd-model-warmup':'warmup','fd-model-tau':'manualTau','fd-model-false-cost':'falseBlockCost','fd-model-missed-cost':'missedFraudCost','fd-model-objective':'objective','fd-alpha':'alpha','fd-warmup':'warmup','fd-forward':'forwardDelay','fd-delay':'reportDelay'}))globalThis.PaymentInfoMetadata?.attachControl(root,id,key);
  pendingImports.add(datasetControls.ready);datasetControls.ready.finally(()=>pendingImports.delete(datasetControls.ready));
  $('fd-dataset-file').addEventListener('change',()=>{
    const file=$('fd-dataset-file').files[0];if(!file)return;
    datasetControls.cancel();return importDataset(()=>file.text(),'Reading '+file.name+'…');
  });
  $('fd-dataset-reset').addEventListener('click',()=>{datasetControls.cancel();$('fd-dataset-file').value='';return setDataset(null);});
  for(const id of['fd-scenario','fd-size','fd-seed'])$(id).addEventListener('change',()=>rebuild(false));
  $('fd-model').addEventListener('change',()=>{stop();model=models.find(m=>m.id===$('fd-model').value)||model;syncPolicyControls();return render(true);});
  $('fd-model-head').addEventListener('change',()=>{modelPolicies[model.id].predictionHead=$('fd-model-head').value;return rebuild(true);});
  $('fd-recompute').addEventListener('click',()=>rebuild(true));
  for(const id of['fd-mode','fd-alpha','fd-warmup','fd-training-mode','fd-policy','fd-objective'])$(id).addEventListener('change',()=>{if(id==='fd-warmup')datasetStatus(importedData);return rebuild(true);});
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
  populateModels();
  const initialTask=rebuild();
  latestTask=(async()=>{await initialTask;if(/^https?:$/.test(globalThis.location?.protocol||''))$('fd-runtime-status').textContent='Preparing Python models and historical reference scores…';availableNative=(await nativeClient.discover()).filter(m=>m.available&&nativeDefinitions.some(d=>d.id===m.id));if(availableNative.length)return rebuild(true);runtimeStatus($('fd-training-mode').value);populateModels();})();
  function getNativeSnapshot(){
    const nativeModel=models.find(m=>m.native_run);if(!nativeModel)return null;
    const runner=group.runners.get(nativeModel.id);
    return {head:nativeModel.native_run.head.id,alpha:nativeModel.native_run.alpha,options:nativeModel.native_run.options,dataset:data.name,evidence:runner.preview()?.evidence||null,evaluation_ids:nativeModel.native_run.evaluation_ids.slice(),
      metrics:models.map(m=>({id:m.id,evaluation_ids:group.evaluationRecords(m.id).map(r=>r.event.id),...compare.metrics(group.evaluationRecords(m.id),data.truth,group.options.alpha,group.runners.get(m.id).state.errorCosts)})),
      decisions:runner.state.decisions.map(r=>({id:r.event.id,evidence:r.evidence,score:r.score,tau:r.tauBefore,decision:r.decision}))};
  }
  root.demo={setDataset,getNativeSnapshot,whenIdle,getPerformance:()=>({busy,comparisons:comparisonCache.entries.size,inferenceCalls:Array.from(group.runners.values()).reduce((sum,r)=>sum+r.inferenceCalls,0),...group.lastTiming}),getSnapshot:()=>({busy,count,selected,model:model.id,policyScope:$('fd-policy').value,modelPolicies:JSON.parse(JSON.stringify(modelPolicies)),mode:result?.state.mode,trainingMode:result?.state.trainingMode,decisionPolicy:result?.state.decisionPolicy,policyFit:result?.state.policyFit,rankingBudget:group.options.alpha,warmup:result?.state.warmup,comparison:entries.map(e=>({id:e.model.id,policy:currentPolicyLabel(e.result.state),score:e.prediction?.score??null,tau:e.result.state.tau,impliedAlpha:e.result.state.policyFit?.impliedAlpha??null,decision:e.decision,payments:e.result.state.payments.map(x=>x.id)})),decision:root.dataset.decision,score:prediction?.score??null,tau:result?.state.tau,accounts:data.accounts.length,events:data.events.length,processed:result?.state.events.map(e=>e.id)||[],decisions:result?.state.decisions.map(r=>({id:r.event.id,score:r.score,tau:r.tauBefore,decision:r.decision,settled:r.settled}))||[]})};
})();
