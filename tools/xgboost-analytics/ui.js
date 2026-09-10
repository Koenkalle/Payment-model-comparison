/* Offline, completed-run XGBoost dashboard. Selection never replays history. */
(function(){
  'use strict';
  const root=document.getElementById('xgboost-analytics');
  if(!root)return;
  const $=id=>root.querySelector('#xa-'+id),analytics=globalThis.FraudAnalytics,xgb=globalThis.FraudXGBoost;
  const ns='http://www.w3.org/2000/svg',PAGE_SIZE=25;
  const currency=new Intl.NumberFormat('en-IE',{style:'currency',currency:'EUR',maximumFractionDigits:2});
  const number=new Intl.NumberFormat('en-IE',{maximumFractionDigits:3});
  const fmt=(v,d=3)=>v===null||v===undefined||!Number.isFinite(Number(v))?'—':Number(v).toLocaleString('en-IE',{maximumFractionDigits:d});
  const signed=v=>(v>0?'+':'')+fmt(v,4);
  const pct=v=>v===null||v===undefined?'—':fmt(v*100,1)+'%';
  const clock=t=>t===null?'—':'Day '+(Math.floor(t/1440)+1)+', '+String(Math.floor(t/60)%24).padStart(2,'0')+':'+String(Math.floor(t)%60).padStart(2,'0');
  const colors={ALLOW:'var(--xa-teal)',BLOCK:'var(--xa-red)',LEARNING:'var(--xa-quiet)'};
  let importedData=null;
  let bundle,model,definitions,report=null,records=[],summary=null,selectedId=null,featureIndex=0,page=0,tab='overview';
  let busy=false,runVersion=0,latestTask=Promise.resolve(),lastError=null,dirty=false;
  function element(tag,parent,text='',attrs={}){
    const el=document.createElement(tag);if(text!==null)el.textContent=String(text);
    for(const [key,value]of Object.entries(attrs))el.setAttribute(key,String(value));
    if(parent)parent.appendChild(el);return el;
  }
  function svgNode(tag,parent,attrs={},text=''){
    const el=document.createElementNS(ns,tag);for(const [key,value]of Object.entries(attrs))el.setAttribute(key,String(value));
    if(text)el.textContent=text;if(parent)parent.appendChild(el);return el;
  }
  function text(id,value){$(id).textContent=value;}
  function row(body,values){const tr=element('tr',body);values.forEach(value=>{const td=element('td',tr);if(value&&typeof value==='object'&&value.nodeType)td.appendChild(value);else td.textContent=String(value??'—');});return tr;}
  function option(parent,value,label){element('option',parent,label,{value});}
  function badge(decision,parent){return element('span',parent,decision==='LEARNING'?'WARM-UP':decision==='BLOCK'?'BLOCKED':'ALLOWED',{class:'xa-pill '+(decision==='BLOCK'?'xa-pill-block':decision==='ALLOW'?'xa-pill-allow':'')});}
  function median(values){if(!values.length)return null;const sorted=values.slice().sort((a,b)=>a-b),m=Math.floor(sorted.length/2);return sorted.length%2?sorted[m]:(sorted[m-1]+sorted[m])/2;}
  function limits(values,fallback=[0,1]){if(!values.length)return fallback;let lo=Infinity,hi=-Infinity;for(const v of values){lo=Math.min(lo,v);hi=Math.max(hi,v);}if(lo===hi){const d=Math.max(.01,Math.abs(lo)*.1);return [lo-d,hi+d];}return [lo,hi];}
  const scale=(domain,range)=>v=>range[0]+(v-domain[0])/(domain[1]-domain[0])*(range[1]-range[0]);
  function sample(rows,max){if(rows.length<=max)return rows;return Array.from({length:max},(_,i)=>rows[Math.floor(i*(rows.length-1)/(max-1))]);}
  function valueColor(value,min,max){const p=max===min?.5:Math.max(0,Math.min(1,(value-min)/(max-min)));return 'rgb('+Math.round(52+135*p)+','+Math.round(123-54*p)+','+Math.round(172-95*p)+')';}
  function name(id){return report?.accounts[id]?.name??String(id);}
  function featureValue(value,index){
    if(value===null||value===undefined)return '—';const unit=definitions[index].unit||'';
    if(unit==='EUR'||unit==='euros'||unit==='€'||unit==='currency units')return currency.format(value);
    if(unit==='boolean'||unit==='binary')return value?'Yes':'No';
    if(unit==='minutes'||unit==='min')return fmt(value,2)+' min';
    if(unit==='ratio')return fmt(value,3)+'×';
    return number.format(value)+(unit==='count'||unit==='bin'||!unit?'':' '+unit);
  }
  function stat(parent,label,value,note){const box=element('div',parent,'',{class:'xa-stat'});element('div',box,label,{class:'xa-stat-label'});element('div',box,value,{class:'xa-stat-value'});element('div',box,note,{class:'xa-stat-note'});}
  function featureButton(index,withId=true){
    const button=element('button',null,'',{type:'button',class:'xa-link'});
    element('span',button,definitions[index].label,{class:'xa-feature-name'});
    if(withId)element('span',button,definitions[index].id,{class:'xa-feature-id'});
    button.addEventListener('click',()=>{featureIndex=index;$('feature').value=String(index);renderFeatures();setTab('features');});
    return button;
  }
  function initChart(id,width,height,title,description){const svg=$(id);svg.replaceChildren();svg.setAttribute('viewBox','0 0 '+width+' '+height);svgNode('title',svg,{},title);svgNode('desc',svg,{},description);return svg;}
  function axis(svg,x,y,domainX,domainY,box,xLabel,yLabel){
    for(let i=0;i<=4;i++){
      const yy=domainY[0]+(domainY[1]-domainY[0])*i/4,xx=domainX[0]+(domainX[1]-domainX[0])*i/4;
      svgNode('line',svg,{x1:box.left,y1:y(yy),x2:box.right,y2:y(yy),stroke:'var(--xa-line)','stroke-width':1});
      svgNode('text',svg,{x:box.left-9,y:y(yy)+4,'text-anchor':'end'},fmt(yy,2));
      svgNode('text',svg,{x:x(xx),y:box.bottom+20,'text-anchor':i===0?'start':i===4?'end':'middle'},fmt(xx,2));
    }
    svgNode('text',svg,{x:box.left,y:box.bottom+42},xLabel);
    svgNode('text',svg,{x:box.left,y:box.top-12},yLabel);
  }
  function readConfig(){
    const numeric=(id,fallback,min,max)=>{const n=Number($(id).value);return Number.isFinite(n)?Math.min(max,Math.max(min,n)):fallback;};
    return {scenario:$('scenario').value,size:$('size').value,seed:Math.floor(numeric('seed',42,1,99999)),
      decisionPolicy:$('policy').value,alpha:numeric('alpha',2,.1,50)/100,warmup:Math.floor(numeric('warmup',128,1,10000)),
      manualTau:numeric('tau',1,0,100000),falseBlockCost:numeric('false-cost',1,.01,10000),missedFraudCost:numeric('missed-cost',20,.01,10000),
      objective:$('objective').value,trainingMode:'supervised',mode:'shadow'};
  }
  function updatePolicyControls(){
    const policy=$('policy').value;
    root.querySelectorAll('[data-xa-policy]').forEach(label=>{const active=label.getAttribute('data-xa-policy')===policy;label.hidden=!active;label.querySelectorAll('input,select').forEach(el=>{el.disabled=!active;});});
    const notes={shared:'The cutoff learns from unlabeled warm-up scores, then tracks the target rate.',manual:'The configured threshold is frozen for every payment in the run.',tuned:'The cutoff minimizes the configured costs on held-out historical labels.',auto:'The cutoff optimizes the selected objective on held-out historical labels.'};
    text('policy-note',notes[policy]);
  }
  function setBusy(value){
    busy=value;root.dataset.busy=String(value);$('results').setAttribute('aria-busy',String(value));$('progress').hidden=!value;$('cancel').hidden=!value;
    text('run',value?'Restart analysis ↗':'Analyze scenario ↗');
  }
  function cancel(){
    if(!busy)return;++runVersion;setBusy(false);text('status','Analysis cancelled.'+(report?' The previous completed report remains available.':''));$('start-empty').hidden=!!report;
  }
  function run(overrides={}){
    const version=++runVersion,config={...readConfig(),...overrides};lastError=null;delete root.dataset.error;$('error').hidden=true;$('start-empty').hidden=true;dirty=false;setBusy(true);$('progress').value=0;
    text('status','Preparing scenario and historical explanations…');
    latestTask=(async()=>{
      try{
        const data=importedData||globalThis.FraudDatasets.load('synthetic_payments',{name:config.scenario,size:config.size,seed:config.seed});
        const next=await analytics.run(model,data,bundle.explanation,config,{budgetMs:8,cancelled:()=>version!==runVersion,onProgress:p=>{
          if(version!==runVersion)return;$('progress').value=p.total?p.done/p.total:1;
          text('status','Analyzing '+fmt(p.done,0)+' / '+fmt(p.total,0)+' events · capturing original decisions and feature contributions…');
        }});
        if(!next||version!==runVersion)return null;
        report=next;selectedId=null;page=0;setBusy(false);$('results').hidden=false;
        const elapsed=report.timing?.durationMs;
        text('status','Analysis complete · '+fmt(report.finalPosition,0)+' events processed'+(Number.isFinite(elapsed)?' in '+fmt(elapsed/1000,2)+' s':'')+(report.timing?.cacheHit?' · captured model scores reused.':'.'));
        renderReport();return report;
      }catch(error){
        if(version!==runVersion)return null;lastError=error.message;root.dataset.error=lastError;setBusy(false);text('error','Could not complete the analysis: '+lastError);$('error').hidden=false;text('status',report?'The previous completed report remains available.':'Analysis did not produce a report.');$('start-empty').hidden=!!report;return null;
      }
    })();return latestTask;
  }
  async function whenIdle(){for(;;){const task=latestTask;await task;if(task===latestTask)return;}}
  function setDataset(document){
    const next=document===null?null:globalThis.FraudDatasets.load('payment_json',document);
    importedData=next;
    for(const id of ['scenario','size','seed'])$(id).disabled=!!next;
    text('dataset-status',next?'Imported '+next.name+' · '+next.events.length+' events. This checkpoint was trained on synthetic history; import does not retrain it.':'Synthetic dataset. This browser checkpoint was trained on synthetic history.');
    return run();
  }
  function filters(){return {query:$('search').value,decision:$('decision').value,outcome:$('outcome').value,sort:$('sort').value,direction:$('direction').value};}
  function setFilters(changes={}){
    const map={query:'search',decision:'decision',outcome:'outcome',sort:'sort',direction:'direction'};
    for(const [key,value]of Object.entries(changes))if(map[key])$(map[key]).value=value;
    page=0;renderPopulation();return records;
  }
  function setTab(next,focus=false){
    tab=next;for(const name of ['overview','features','decisions']){const active=name===next;$(name).hidden=!active;$('tab-'+name).setAttribute('aria-selected',String(active));$('tab-'+name).setAttribute('tabindex',active?'0':'-1');}if(focus)$('tab-'+next).focus();
  }
  function selectPayment(id,show=true){
    if(!records.some(r=>r.event.id===id))return false;selectedId=id;renderPayments();renderInspector();if(show)setTab('decisions');return true;
  }
  function renderReport(){
    const config=report.configuration,scenario=globalThis.FraudScenarios.catalog.find(s=>s.id===config.scenario);
    text('report-title',scenario?.name||config.scenario||'Completed scenario');
    const policyNames={shared:'Unlabeled budget',manual:'Fixed threshold',tuned:'Historical cost tuning',auto:'Historical '+String(config.objective||'f1').toUpperCase()};
    text('run-meta',(config.size||'Custom')+' network · '+fmt(report.accounts.length,0)+' accounts'+(config.seed==null?'':' · seed '+config.seed)+' · '+policyNames[config.decisionPolicy]+' · final event: '+clock(report.finalTime));
    const provenance=$('provenance');provenance.replaceChildren();const dl=element('dl',provenance);
    const reference=report.reference||{};
    for(const [key,value]of [['Checkpoint',report.model.checkpointId||report.model.id],['Checkpoint SHA-256',report.model.checkpointHash||bundle.explanation.checkpoint_sha256||'Unavailable'],['Feature schema',report.featureSchemaVersion??bundle.explanation.feature_schema_version],['Reference',reference.id||bundle.explanation.reference_id],['Reference population',reference.convention||reference.description||'Unweighted fitting-row counts; not native XGBoost Hessian cover'],['Reference details',JSON.stringify(reference)],['Completed events',report.finalPosition],['Final cutoff, bits',fmt(report.finalTau,6)]]){element('dt',dl,key);element('dd',dl,String(value??'Unavailable'));}
    renderPopulation();
  }
  function renderPopulation(){
    if(!report)return;records=analytics.filter(report,filters());summary=analytics.summarize(report,records);
    if(!records.some(r=>r.event.id===selectedId))selectedId=records[0]?.event.id??null;
    page=Math.min(page,Math.max(0,Math.ceil(records.length/PAGE_SIZE)-1));
    text('population',fmt(records.length,0)+' / '+fmt(report.records.length,0)+' payments');
    renderOverview();renderFeatures();renderPayments();renderInspector();
  }
  function renderOverview(){
    const m=summary.metrics,n=records.length,known=m.labeled||0,assessed=m.eligible||0,blocked=records.filter(r=>r.decision==='BLOCK').length;
    const holder=$('overview-stats');holder.replaceChildren();
    stat(holder,'Payments',fmt(n,0),'In the displayed population');
    stat(holder,'Assessed',fmt(assessed,0),fmt(known,0)+' with known outcomes');
    stat(holder,'Blocked',fmt(blocked,0),'Recorded policy decisions');
    stat(holder,'Block rate',assessed?pct(m.blockRate):'—',fmt(blocked,0)+' / '+fmt(assessed,0)+' assessed');
    stat(holder,'Precision',known?pct(m.precision):'—',fmt(m.tp,0)+' / '+fmt(m.tp+m.fp,0)+' labeled blocks');
    stat(holder,'Recall',known?pct(m.recall):'—',fmt(m.tp,0)+' / '+fmt(m.tp+m.fn,0)+' labeled fraud');
    stat(holder,'F1',known?fmt(m.f1,3):'—','Known assessed outcomes only');
    stat(holder,'Warm-up',fmt(m.warmup,0),'Allowed; excluded from assessment');
    text('metric-note','Classification uses '+fmt(known,0)+' known assessed outcomes. '+fmt(m.assessedUnknown||0,0)+' assessed payments have unknown outcomes. Block rate includes all '+fmt(assessed,0)+' assessed payments.'+(known?'':' Classification metrics are unavailable for this population.'));
    for(const key of ['tp','fp','fn','tn']){const td=$(key);td.replaceChildren();element('span',td,fmt(m[key]||0,0));element('small',td,{tp:'fraud caught',fp:'false positives',fn:'false negatives',tn:'benign allowed'}[key]);}
    text('confusion-note',known?fmt(known,0)+' labeled assessed payments · '+fmt(m.warmup,0)+' warm-up payments excluded.':'No labeled assessed payments in this population.');
    text('cutoff-note','Final learned / configured cutoff: '+fmt(report.finalTau,5)+' bits. Each cohort uses its original threshold; the final cutoff is not applied retrospectively.');
    renderDistribution();
  }
  function renderDistribution(){
    const count=18,max=Math.max(.05,...records.map(r=>r.score))*1.001,width=620,height=260;
    const bins=Array.from({length:count},(_,i)=>({lo:max*i/count,hi:max*(i+1)/count,ALLOW:0,BLOCK:0,LEARNING:0}));
    for(const r of records)bins[Math.min(count-1,Math.floor(r.score/max*count))][r.decision]++;
    const top=Math.max(1,...bins.map(b=>b.ALLOW+b.BLOCK+b.LEARNING)),box={left:48,right:607,top:28,bottom:204};
    const x=scale([0,max],[box.left,box.right]),y=scale([0,top],[box.bottom,box.top]);
    const svg=initChart('score-chart',width,height,'Score distribution',records.length+' payments, grouped by their original recorded decisions. Exact counts follow in the distribution table.');
    axis(svg,x,y,[0,max],[0,top],box,'Model score, bits','Payments');
    if(!records.length)svgNode('text',svg,{x:width/2,y:115,'text-anchor':'middle'},'No payments in this population');
    for(const bin of bins){let cumulative=0;for(const decision of ['ALLOW','LEARNING','BLOCK']){const value=bin[decision];if(value){const bar=svgNode('rect',svg,{x:x(bin.lo)+1,y:y(cumulative+value),width:Math.max(1,x(bin.hi)-x(bin.lo)-2),height:y(cumulative)-y(cumulative+value),fill:colors[decision],rx:1});svgNode('title',bar,{},fmt(bin.lo,3)+'–'+fmt(bin.hi,3)+' bits: '+value+' '+decision);}cumulative+=value;}}
    const body=$('score-table').querySelector('tbody');body.replaceChildren();for(const b of bins)row(body,[fmt(b.lo,3)+'–'+fmt(b.hi,3),b.ALLOW,b.BLOCK,b.LEARNING]);
  }
  function renderFeatures(){
    if(!report)return;const n=records.length,query=$('feature-search').value.trim().toLowerCase();
    text('feature-population',fmt(n,0)+' displayed payments · '+definitions.length+' features · mean absolute contribution across this population');
    const ranked=summary.featureStats.slice().sort((a,b)=>b.meanAbs-a.meanAbs||a.index-b.index),max=Math.max(1e-12,...ranked.map(f=>f.meanAbs));
    const contributionMax=Math.max(.001,...ranked.flatMap(f=>[Math.abs(f.min||0),Math.abs(f.max||0)]));
    const visible=ranked.filter(f=>!query||Object.values(definitions[f.index]).join(' ').toLowerCase().includes(query));
    const body=$('importance-table').querySelector('tbody');body.replaceChildren();
    const sampled=sample(records,90);
    for(const f of visible){
      const tr=element('tr',body,'',{'data-feature':f.id,class:f.index===featureIndex?'xa-selected':''});
      const nameCell=element('td',tr);nameCell.appendChild(featureButton(f.index));
      const importance=element('div',element('td',tr),'',{class:'xa-importance'}),bar=element('div',importance,'',{class:'xa-bar'});element('span',bar).style.width=(100*f.meanAbs/max)+'%';element('span',importance,n?fmt(f.meanAbs,4):'—',{class:'xa-mono'});
      const distribution=svgNode('svg',element('td',tr),{viewBox:'0 0 220 34',class:'xa-beeswarm',role:'img','aria-label':definitions[f.index].label+' SHAP range '+fmt(f.min,4)+' to '+fmt(f.max,4)+' log-odds'});
      svgNode('line',distribution,{x1:110,y1:2,x2:110,y2:32,stroke:'var(--xa-line)'});
      const x=scale([-contributionMax,contributionMax],[6,214]);
      sampled.forEach((r,i)=>{const value=r.explanation.contributions[f.index],dot=svgNode('circle',distribution,{cx:x(value),cy:17+((i*37)%23-11),r:2.2,fill:valueColor(r.readableValues[f.index],f.minValue,f.maxValue),'fill-opacity':.7});svgNode('title',dot,{},r.event.id+': '+signed(value)+' log-odds; '+featureValue(r.readableValues[f.index],f.index));});
      element('td',tr,featureValue(median(records.map(r=>r.readableValues[f.index])),f.index),{class:'xa-right'});
    }
    $('feature-empty').hidden=visible.length>0;renderFeatureDetail();
  }
  function renderFeatureDetail(){
    const def=definitions[featureIndex],stats=summary.featureStats[featureIndex];text('definition-title',def.label);text('definition-description',def.description);text('dependence-title',def.label);
    const definition=$('definition');definition.replaceChildren();const dl=element('dl',definition);
    for(const [key,value]of [['Feature ID',def.id],['Unit',def.unit||'Model units'],['Source',def.source||'Observed history before this payment'],['Window',def.window||'All prior observed history'],['Transform',def.transform||'Identity']]){element('dt',dl,key);element('dd',dl,value);}
    const holder=$('feature-stats');holder.replaceChildren();const table=element('table',holder),body=element('tbody',table);
    row(body,['Minimum readable value',featureValue(stats.minValue,featureIndex)]);row(body,['Median readable value',featureValue(median(records.map(r=>r.readableValues[featureIndex])),featureIndex)]);row(body,['Maximum readable value',featureValue(stats.maxValue,featureIndex)]);row(body,['Mean contribution',fmt(stats.mean,5)+' log-odds']);row(body,['Mean absolute contribution',records.length?fmt(stats.meanAbs,5)+' log-odds':'—']);
    const sampled=sample(records,600),values=records.map(r=>r.readableValues[featureIndex]),contributions=records.map(r=>r.explanation.contributions[featureIndex]);
    const dx=limits(values),dy=limits(contributions),box={left:60,right:544,top:34,bottom:236},x=scale(dx,[box.left,box.right]),y=scale(dy,[box.bottom,box.top]);
    const svg=initChart('dependence-chart',560,290,def.label+' versus contribution',sampled.length+' of '+records.length+' displayed payments. Horizontal axis shows readable feature values; vertical axis shows raw-margin SHAP contribution.');
    axis(svg,x,y,dx,dy,box,'Readable value ('+(def.unit||'unitless')+')','SHAP, log-odds');
    if(dy[0]<=0&&dy[1]>=0)svgNode('line',svg,{x1:box.left,x2:box.right,y1:y(0),y2:y(0),stroke:'var(--xa-quiet)','stroke-dasharray':'3 4'});
    for(const r of sampled){const v=r.readableValues[featureIndex],phi=r.explanation.contributions[featureIndex];const dot=svgNode('circle',svg,{cx:x(v),cy:y(phi),r:r.event.id===selectedId?4.5:2.8,fill:valueColor(v,stats.minValue,stats.maxValue),'fill-opacity':.68,stroke:r.event.id===selectedId?'var(--foreground)':'none','stroke-width':1.3});svgNode('title',dot,{},r.event.id+' · '+featureValue(v,featureIndex)+' · '+signed(phi)+' log-odds');dot.addEventListener('click',()=>selectPayment(r.event.id));}
    if(!records.length)svgNode('text',svg,{x:300,y:135,'text-anchor':'middle'},'No payments in this population');
    text('dependence-note',fmt(sampled.length,0)+' / '+fmt(records.length,0)+' payments plotted. Statistics use the entire displayed population. Readable values are shown here; the model’s transformed input is available in each payment inspector.');
  }
  function renderPayments(){
    const body=$('payment-table').querySelector('tbody');body.replaceChildren();const start=page*PAGE_SIZE,view=records.slice(start,start+PAGE_SIZE);
    for(const r of view){const e=r.event,tr=element('tr',body,'',{'data-payment-id':e.id,class:e.id===selectedId?'xa-selected':''});
      const first=element('td',tr),button=element('button',first,e.id,{type:'button',class:'xa-link','aria-label':'Inspect payment '+e.id,'aria-pressed':String(e.id===selectedId)});button.addEventListener('click',()=>selectPayment(e.id));element('div',first,clock(e.t),{class:'xa-small xa-muted'});
      const accounts=element('td',tr);element('div',accounts,name(e.u));element('div',accounts,'→ '+name(e.v),{class:'xa-small xa-muted'});
      for(const value of [currency.format(e.amount),fmt(r.score,5),fmt(r.tauBefore,5)])element('td',tr,value,{class:'xa-right'});
      badge(r.decision,element('td',tr));element('td',tr,r.label===null?'Unknown':r.label===1?'Fraud':'Benign');
    }
    $('payment-empty').hidden=records.length>0;text('page-label',records.length?'Payments '+fmt(start+1,0)+'–'+fmt(Math.min(records.length,start+PAGE_SIZE),0)+' of '+fmt(records.length,0)+' · page '+(page+1)+' / '+Math.ceil(records.length/PAGE_SIZE):'0 payments');
    $('prev-page').disabled=page===0;$('next-page').disabled=start+PAGE_SIZE>=records.length;
  }
  function renderInspector(){
    const record=records.find(r=>r.event.id===selectedId);$('inspector').hidden=!record;if(!record)return;
    const e=record.event;root.dataset.selected=e.id;text('inspector-title',e.id+' · '+name(e.u)+' → '+name(e.v));
    text('inspector-meta',clock(e.t)+' · '+currency.format(e.amount)+' · outcome: '+(record.label===null?'unknown':record.label===1?'fraud':'benign')+' · '+(record.settled?'transfer executed':'transfer not settled'));
    $('inspector-decision').replaceChildren();badge(record.decision,$('inspector-decision'));
    const stats=$('inspector-stats');stats.replaceChildren();stat(stats,'Model probability',pct(record.probability),'Uncalibrated model estimate');stat(stats,'Score',fmt(record.score,5),'Bits · −log₂(1 − probability)');stat(stats,'Original threshold',fmt(record.tauBefore,5),'Bits · captured before decision');stat(stats,'Raw model margin',fmt(record.rawMargin,5),'Log-odds · additive explanation');
    text('original-decision',record.decision==='LEARNING'?'This payment was allowed during warm-up, before a threshold was available.':'Original decision: '+fmt(record.score,6)+(record.decision==='BLOCK'?' > ':' ≤ ')+fmt(record.tauBefore,6)+' bits → '+(record.decision==='BLOCK'?'blocked':'allowed')+'. All transfers continue in observed-history mode.');
    renderWaterfall(record);
    const body=$('values-table').querySelector('tbody');body.replaceChildren();
    definitions.forEach((def,i)=>{const tr=row(body,[featureButton(i),featureValue(record.readableValues[i],i),fmt(record.features[i],7),signed(record.explanation.contributions[i])]);Array.from(tr.children).slice(1).forEach(td=>td.classList.add('xa-right'));tr.lastChild.classList.add(record.explanation.contributions[i]>0?'xa-high':record.explanation.contributions[i]<0?'xa-good':'xa-muted');tr.setAttribute('data-feature',def.id);});
    renderTree(record);
  }
  function renderWaterfall(record){
    const explanation=record.explanation,ranked=definitions.map((f,i)=>({label:f.label,index:i,value:explanation.contributions[i]})).sort((a,b)=>Math.abs(b.value)-Math.abs(a.value)||a.index-b.index);
    const shown=ranked.slice(0,10),rest=ranked.slice(10);if(rest.length)shown.push({label:rest.length+' other features',value:rest.reduce((sum,f)=>sum+f.value,0)});
    let current=explanation.baseline;const bars=[{label:'Expected reference margin',start:0,end:current,total:true}];
    for(const f of shown){bars.push({...f,start:current,end:current+f.value});current+=f.value;}bars.push({label:'Payment raw margin',start:0,end:record.rawMargin,total:true});
    const h=bars.length*28+58,width=950,lo=Math.min(0,...bars.flatMap(b=>[b.start,b.end])),hi=Math.max(0,...bars.flatMap(b=>[b.start,b.end]));
    const pad=Math.max(.1,(hi-lo)*.1),dx=[lo-pad,hi+pad],x=scale(dx,[260,854]);
    const svg=initChart('waterfall',width,h,'Payment '+record.event.id+' contribution waterfall','Reference margin '+fmt(explanation.baseline,6)+' plus all feature contributions equals '+fmt(record.rawMargin,6)+'. Exact contributions for all features are in the following table.');
    svgNode('line',svg,{x1:x(0),y1:14,x2:x(0),y2:h-34,stroke:'var(--xa-line)','stroke-dasharray':'3 4'});
    bars.forEach((b,i)=>{const y=14+i*28;svgNode('text',svg,{x:248,y:y+14,'text-anchor':'end',class:'xa-chart-label'},b.label);const color=b.total?'var(--xa-quiet)':b.value>=0?'var(--xa-red)':'var(--xa-teal)';
      const rect=svgNode('rect',svg,{x:Math.min(x(b.start),x(b.end)),y,width:Math.max(1,Math.abs(x(b.end)-x(b.start))),height:20,fill:color,rx:2});svgNode('title',rect,{},b.label+': '+(b.total?fmt(b.end,6):signed(b.value))+' log-odds');
      svgNode('text',svg,{x:866,y:y+14,fill:color},b.total?fmt(b.end,5):signed(b.value));
      if(i>0&&!b.total)svgNode('line',svg,{x1:x(b.start),x2:x(b.start),y1:y-8,y2:y,stroke:'var(--xa-quiet)','stroke-dasharray':'2 2'});
    });
    for(let i=0;i<=4;i++){const value=dx[0]+(dx[1]-dx[0])*i/4;svgNode('text',svg,{x:x(value),y:h-15,'text-anchor':'middle'},fmt(value,2));}
    const total=explanation.contributions.reduce((a,b)=>a+b,0),residual=Math.abs(explanation.baseline+total-record.rawMargin);
    text('additivity',fmt(explanation.baseline,6)+' reference + '+signed(total)+' contribution = '+fmt(record.rawMargin,6)+' raw margin · additive residual '+residual.toExponential(1));
  }
  function renderTree(record){
    if(!record)return;const treeIndex=Number($('tree').value)||0,tree=model.trees[treeIndex],path=xgb.treePath(model,record.features,treeIndex),active=new Set(path.steps.map(s=>s.path));active.add(path.leafPath);
    const list=$('tree-path');list.replaceChildren();for(const step of path.steps)element('li',list,definitions[step.feature].label+': model input '+fmt(step.value,7)+(step.direction==='left'?' ≤ ':' > ')+fmt(step.threshold,7)+' → '+step.direction+' branch.');element('li',list,'Reached leaf '+fmt(path.leaf,7)+' × learning rate '+fmt(model.learning_rate,4)+' = '+signed(path.weightedLeaf)+' log-odds.');
    text('tree-summary','Tree '+(treeIndex+1)+' / '+model.trees.length+' · weighted leaf '+signed(path.weightedLeaf)+' log-odds');
    const nodes=[];let depthMax=0;
    function collect(node,key='root',depth=0,lo=0,hi=1020){const item={node,key,depth,x:(lo+hi)/2,y:32+depth*86};nodes.push(item);depthMax=Math.max(depthMax,depth);if(node.leaf===undefined){collect(node.left,key==='root'?'L':key+'L',depth+1,lo,item.x);collect(node.right,key==='root'?'R':key+'R',depth+1,item.x,hi);}return item;}
    collect(tree);const svg=initChart('tree-chart',1020,depthMax*86+100,'Tree '+(treeIndex+1)+' path for '+record.event.id,'The highlighted branch path is listed in text immediately below this tree.');
    const byKey=new Map(nodes.map(n=>[n.key,n]));
    for(const item of nodes){if(item.key==='root')continue;const parent=byKey.get(item.key.length===1?'root':item.key.slice(0,-1));const highlighted=active.has(item.key)&&active.has(parent.key);svgNode('path',svg,{d:'M '+parent.x+' '+(parent.y+21)+' L '+item.x+' '+(item.y-20),stroke:highlighted?'var(--xa-teal)':'var(--xa-line)','stroke-width':highlighted?2.5:1.2,fill:'none'});}
    for(const item of nodes){const isLeaf=item.node.leaf!==undefined,onPath=active.has(item.key),group=svgNode('g',svg);svgNode('rect',group,{x:item.x-58,y:item.y-20,width:116,height:42,rx:5,fill:onPath?'color-mix(in srgb,var(--xa-teal) 12%,var(--xa-panel))':'var(--xa-panel)',stroke:onPath?'var(--xa-teal)':'var(--xa-line)','stroke-width':onPath?1.8:1});
      const full=isLeaf?'Leaf '+fmt(item.node.leaf,5):definitions[item.node.feature].label;
      const label=full.length>18?full.slice(0,17)+'…':full;svgNode('text',group,{x:item.x,y:item.y-2,'text-anchor':'middle',class:onPath?'xa-chart-label':''},label);svgNode('text',group,{x:item.x,y:item.y+13,'text-anchor':'middle'},isLeaf?'× '+fmt(model.learning_rate,3)+' = '+fmt(item.node.leaf*model.learning_rate,4):'≤ '+fmt(item.node.threshold,6));svgNode('title',group,{},full+(isLeaf?'':'; model input ≤ '+item.node.threshold)+(onPath?'; payment path':''));
    }
  }
  function download(kind){
    if(!report)return;const value=kind==='csv'?analytics.toCSV(report,records):analytics.toJSON(report,records),content=typeof value==='string'?value:JSON.stringify(value,null,2);
    const url=URL.createObjectURL(new Blob([content],{type:kind==='csv'?'text/csv;charset=utf-8':'application/json'}));
    const a=element('a',root,'',{href:url,download:'xgboost-'+(report.configuration.scenario||'scenario')+'-'+report.configuration.size+'-'+report.configuration.seed+'-'+records.length+'-payments.'+kind});a.hidden=true;a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
    text('status','Exported '+fmt(records.length,0)+' / '+fmt(report.records.length,0)+' payments as '+kind.toUpperCase()+'.');
  }
  try{
    bundle=JSON.parse($('model-data').textContent);model=bundle.models.find(m=>m.id==='xgboost')||bundle.models[0];definitions=xgb.featureDefinitions;
    text('checkpoint',model.trees.length+' trees · '+definitions.length+' features');
    definitions.forEach((f,i)=>option($('feature'),i,f.label));model.trees.forEach((tree,i)=>option($('tree'),i,'Tree '+(i+1)));
    if(bundle.policy?.warmup)$('warmup').value=String(bundle.policy.warmup);
    if(!model.policy_validation?.supervised){for(const name of ['tuned','auto']){const el=$('policy').querySelector('option[value="'+name+'"]');el.disabled=true;el.textContent+=' · validation unavailable';}}
    updatePolicyControls();
    $('config-form').addEventListener('submit',e=>{e.preventDefault();run();});
    $('config-form').addEventListener('change',()=>{updatePolicyControls();dirty=true;if(!busy)text('status','Settings changed. Analyze scenario to create a report with this configuration.');});
    $('cancel').addEventListener('click',cancel);
    $('dataset-file').addEventListener('change',async()=>{try{const file=$('dataset-file').files[0];if(file)await setDataset(await file.text());}catch(error){text('dataset-status','Dataset not loaded: '+error.message);}});
    $('dataset-reset').addEventListener('click',()=>{ $('dataset-file').value='';return setDataset(null);});
    for(const id of ['decision','outcome','sort','direction'])$(id).addEventListener('change',()=>{page=0;renderPopulation();});
    $('search').addEventListener('input',()=>{page=0;renderPopulation();});
    $('reset-filters').addEventListener('click',()=>setFilters({query:'',decision:'all',outcome:'all'}));
    $('feature-search').addEventListener('input',renderFeatures);
    $('feature').addEventListener('change',()=>{featureIndex=Number($('feature').value);renderFeatures();});
    $('tree').addEventListener('change',()=>renderTree(records.find(r=>r.event.id===selectedId)));
    $('prev-page').addEventListener('click',()=>{page--;renderPayments();});$('next-page').addEventListener('click',()=>{page++;renderPayments();});
    $('export-csv').addEventListener('click',()=>download('csv'));$('export-json').addEventListener('click',()=>download('json'));
    const tabs=['overview','features','decisions'];for(const name of tabs){$('tab-'+name).addEventListener('click',()=>setTab(name));$('tab-'+name).addEventListener('keydown',e=>{let next;if(['ArrowRight','ArrowDown'].includes(e.key))next=(tabs.indexOf(tab)+1)%tabs.length;else if(['ArrowLeft','ArrowUp'].includes(e.key))next=(tabs.indexOf(tab)+tabs.length-1)%tabs.length;else if(e.key==='Home')next=0;else if(e.key==='End')next=tabs.length-1;else return;e.preventDefault();setTab(tabs[next],true);});}
    root.demo={setDataset,whenIdle,run,cancel,setFilters,selectPayment,setTab,get report(){return report;},getSnapshot:()=>({busy,dirty,error:lastError,tab,count:report?.records.length||0,filteredCount:records.length,featureCount:definitions.length,selectedId,finalPosition:report?.finalPosition??0,finalTime:report?.finalTime??null,metrics:summary?.metrics??null,configuration:report?.configuration??null,page,reportVersion:report?.version??null}),exportCSV:()=>report?analytics.toCSV(report,records):null,exportJSON:()=>report?analytics.toJSON(report,records):null};
    run();
  }catch(error){lastError=error.message;root.dataset.error=lastError;text('error','Could not initialize analytics: '+lastError);$('error').hidden=false;text('status','Analysis is unavailable. Rebuild the offline page to include its model and explanation data.');$('start-empty').hidden=false;}
})();
