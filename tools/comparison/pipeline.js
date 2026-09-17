/* Evaluate immutable training runs against an explicitly shared population. */
(function(global){
  'use strict';
  const root=global.document.getElementById('trained-comparison'),ui=global.PaymentPipelineUI;
  if(!root||!ui)return;
  const $=id=>root.querySelector('#tc-'+id),el=ui.element;
  const state={datasets:[],runs:[],models:[],jobs:[],selected:new Set(),dataset:null,result:null,job:null,page:0,busy:false,revision:0};
  const pending=new Set(),query=new URLSearchParams(global.location.search);
  function track(promise){pending.add(promise);promise.finally(()=>pending.delete(promise)).catch(()=>{});return promise;}
  function status(message){$('status').textContent=message;}
  function choices(){return Array.from(state.selected);}
  function update(){
    $('compare').disabled=state.busy||!state.dataset||!state.selected.size;
    const selected=state.datasets.find(row=>row.id===state.dataset);
    $('dataset-info').textContent=selected?selected.name+' · '+ui.number(selected.rows)+' transactions · '+selected.views.join(', ')+' views':'';
    $('train-link').href=ui.link('trainer.html',{dataset:state.dataset});
    $('data-link').href=ui.link('data-lab.html',{dataset:state.dataset});
    if(state.dataset)ui.remember('dataset',state.dataset);
  }
  function invalidate(){state.revision++;state.busy=false;state.job=null;state.result=null;$('result').hidden=true;status('Selection updated. Compare selected models to evaluate this population.');update();}
  function runDetails(parent,run){
    if(!run)return;
    const details=el('details',parent,null,{class:'tc-run-details'});
    el('summary',details,'Features and saved settings');
    const dataset=state.datasets.find(row=>row.id===run.dataset_id),fields=state.models.find(row=>row.id===run.model_id)?.parameters||[];
    if(run.feature_names?.length){el('p',details,'Training input features',{class:'text-small'});ui.featureList?.(el('div',details,null,{'data-run-features':run.id}),run.feature_names,dataset?.feature_definitions||[]);}
    if(run.parameters){el('p',details,'Saved training parameters',{class:'text-small'});ui.parameterList?.(el('div',details,null,{'data-run-parameters':run.id}),run.parameters,fields);}
  }
  function renderRuns(){
    const container=$('runs');container.replaceChildren();
    if(!state.runs.length){el('p',container,'No trained models are ready. Open Model trainer to create a run.');return;}
    for(const run of state.runs){
      const label=el('label',container,null,{class:'tc-run'}),input=el('input',label,null,{type:'checkbox',value:run.id,'aria-label':run.name});
      input.checked=state.selected.has(run.id);
      const info=el('div',label),counts=run.partition_counts||{};
      el('strong',info,run.name);el('span',info,ui.human(run.model_id)+' · '+run.dataset_name,{class:'text-small'});
      el('span',info,Object.entries(counts).map(([name,count])=>ui.human(name)+': '+ui.number(count)).join(' · '),{class:'tc-badge'});
      el('code',info,'Run '+run.id.slice(0,12)+' · '+ui.date(run.created_at));
      runDetails(info,run);
      input.addEventListener('change',()=>{input.checked?state.selected.add(run.id):state.selected.delete(run.id);invalidate();});
    }
  }
  function renderHistory(){
    const select=$('history'),previous=select.value;select.replaceChildren();
    for(const job of state.jobs.filter(row=>row.kind==='comparison')){
      el('option',select,ui.date(job.created_at)+' · '+ui.human(job.status)+' · '+(job.result?.dataset_name||job.payload?.dataset_id||'').slice(0,55),{value:job.id});
    }
    if(Array.from(select.options).some(row=>row.value===previous))select.value=previous;
    $('open').disabled=!select.value;
  }
  async function refresh(initial=false){
    $('refresh').disabled=true;
    try{
      const [catalog,models,jobs,schemas]=await Promise.all([ui.request('/datasets'),ui.request('/runs'),ui.request('/jobs'),ui.request('/models')]);
      state.datasets=catalog.datasets;state.runs=models.runs;state.jobs=jobs.jobs;state.models=schemas.models;
      global.PaymentInfoMetadata?.setModelSchemas(state.models);
      const requested=initial?(query.get('dataset')||ui.recall('dataset')):state.dataset;
      state.dataset=state.datasets.some(row=>row.id===requested)?requested:state.datasets[0]?.id||null;
      if(initial)state.selected=new Set((query.get('runs')||'').split(',').filter(Boolean));
      state.selected=new Set(choices().filter(id=>state.runs.some(row=>row.id===id)));
      if(initial&&!state.selected.size)state.runs.filter(row=>row.dataset_id===state.dataset).slice(0,2).forEach(row=>state.selected.add(row.id));
      const select=$('dataset');select.replaceChildren();
      for(const dataset of state.datasets)el('option',select,dataset.name+' · '+ui.number(dataset.rows)+' rows',{value:dataset.id});
      if(state.dataset)select.value=state.dataset;
      renderRuns();renderHistory();$('controls').hidden=false;update();
      if(!state.busy)status(state.runs.length?'Select ready models and start a comparison.':'Create a training run to compare its held-out predictions here.');
      if(initial&&query.get('job'))await openJob(query.get('job'));
    }catch(error){status(error.message);}finally{$('refresh').disabled=false;}
  }
  const percent=value=>value===null||value===undefined?'—':ui.number(100*value,1)+'%';
  function renderPredictions(){
    const result=state.result;if(!result)return;
    const head=$('predictions').querySelector('thead'),body=$('predictions').querySelector('tbody');head.replaceChildren();body.replaceChildren();
    const tr=el('tr',head);['Transaction','Time (seconds)','Known outcome',...result.models.map(row=>row.label+' · probability / decision')].forEach(label=>el('th',tr,label));
    const start=state.page*25,rows=result.rows.slice(start,start+25);
    for(const row of rows){const line=el('tr',body);el('td',line,row.id);el('td',line,ui.number(row.timestamp_seconds,2));el('td',line,row.label===null?'Unknown':row.label===1?'Fraud':'Legitimate');
      for(const model of result.models){const value=row.predictions[model.run_id];el('td',line,percent(value.probability)+' · '+value.decision);}
    }
    $('page').textContent=(start+1)+'–'+Math.min(start+25,result.rows.length)+' of '+ui.number(result.rows.length)+' transactions';
    $('prev').disabled=state.page===0;$('next').disabled=start+25>=result.rows.length;
  }
  function renderResult(result){
    state.result=result;state.page=0;$('result').hidden=false;
    $('population').textContent=result.dataset_name+' · '+ui.number(result.row_count)+' '+(result.population==='common-held-out-test'?'common held-out test transactions':'transactions from a separate dataset')+' · '+ui.number(result.known)+' known outcomes · '+ui.number(result.unknown)+' unknown. No training or validation rows are included.';
    const body=$('metrics').querySelector('tbody'),bars=$('bars');body.replaceChildren();bars.replaceChildren();
    for(const model of result.models){
      const metric=model.metrics,tr=el('tr',body),name=el('td',tr);el('strong',name,model.label);el('div',name,model.run_id.slice(0,12),{class:'tc-badge'});
      runDetails(name,state.runs.find(run=>run.id===model.run_id));
      [percent(metric.precision),percent(metric.recall),percent(metric.f1),ui.number(metric.roc_auc,3),ui.number(metric.average_precision,3),ui.number(metric.fp),ui.number(metric.fn),percent(metric.block_rate),percent(model.probability_threshold)].forEach(value=>el('td',tr,value));
      global.PaymentInfo?.attach(tr.lastElementChild,{...global.PaymentInfo.parameter(global.PaymentInfoMetadata.parameters.probability_threshold),facts:[['Saved run',model.label],['Probability cutoff',model.probability_threshold]]});
      const bar=el('div',bars,null,{class:'tc-bar'});el('span',bar,model.label);const track=el('div',bar,null,{class:'tc-track'}),fill=el('div',track,null,{class:'tc-fill'});fill.style.width=(100*(metric.f1||0))+'%';el('span',bar,percent(metric.f1));
    }
    renderPredictions();
  }
  async function watch(job,revision){
    state.job=job;
    while(revision===state.revision){
      state.job=job;status(ui.human(job.status)+' · '+(job.message||ui.human(job.stage)));
      if(job.status==='succeeded'){state.busy=false;renderResult(job.result);status('Comparison complete. This report and its model identities are saved on the server.');update();return;}
      if(job.status==='failed')throw Error(job.error||'Comparison failed.');
      await new Promise(resolve=>global.setTimeout(resolve,600));
      if(revision!==state.revision)return;
      job=(await ui.request('/jobs/'+encodeURIComponent(job.id))).job;
    }
  }
  async function openJob(id){
    const revision=++state.revision;state.busy=true;update();
    try{
      const {job}=await ui.request('/jobs/'+encodeURIComponent(id));if(revision!==state.revision)return;
      if(job.kind!=='comparison')throw Error('Choose a saved comparison job.');
      state.dataset=job.payload.dataset_id;state.selected=new Set(job.payload.run_ids);
      $('dataset').value=state.dataset;$('partition').value=job.payload.partition;renderRuns();update();
      await watch(job,revision);
    }
    catch(error){if(revision===state.revision)status(error.message);}finally{if(revision===state.revision){state.busy=false;update();}}
  }
  async function compare(){
    const revision=++state.revision;state.busy=true;state.result=null;$('result').hidden=true;update();status('Submitting comparison…');
    const payload={dataset_id:state.dataset,run_ids:choices(),partition:$('partition').value};
    try{
      const {job}=await ui.request('/compare',payload);if(revision!==state.revision)return;
      const url=new URL(global.location.href);url.search=new URLSearchParams({dataset:state.dataset,runs:payload.run_ids.join(','),job:job.id});global.history.replaceState(null,'',url);
      state.jobs.unshift(job);renderHistory();await watch(job,revision);
    }catch(error){if(revision===state.revision)status(error.message);}finally{if(revision===state.revision){state.busy=false;update();}}
  }
  $('dataset').addEventListener('change',()=>{state.dataset=$('dataset').value;invalidate();});
  global.PaymentInfoMetadata?.attachControl(root,'tc-dataset','dataset');
  global.PaymentInfoMetadata?.attachControl(root,'tc-partition','partition');
  $('partition').addEventListener('change',invalidate);
  $('refresh').addEventListener('click',()=>track(refresh()));
  $('compare').addEventListener('click',()=>track(compare()));
  $('open').addEventListener('click',()=>track(openJob($('history').value)));
  $('prev').addEventListener('click',()=>{state.page--;renderPredictions();});
  $('next').addEventListener('click',()=>{state.page++;renderPredictions();});
  $('download').addEventListener('click',()=>{if(!state.result)return;const url=URL.createObjectURL(new Blob([JSON.stringify(state.result,null,2)],{type:'application/json'})),a=el('a',root,null,{href:url,download:'comparison-'+state.result.id+'.json'});a.click();a.remove();global.setTimeout(()=>URL.revokeObjectURL(url),1000);});
  root.demo={async whenIdle(){while(pending.size)await Promise.allSettled(Array.from(pending));},getSnapshot(){return {dataset_id:state.dataset,run_ids:choices(),job:state.job,result:state.result,busy:state.busy};}};
  track(refresh(true));
})(globalThis);
