(function(){
  'use strict';
  const root=document.getElementById('model-trainer');if(!root)return;
  const U=globalThis.PaymentPipelineUI,$=id=>root.querySelector('#mt-'+id),e=U.element;
  let datasets=[],models=[],runs=[],jobs=[],selectedDataset=null,pending=0,lastError=null,version=0,submitting=false,pollTimer=null,polling=false,idleWaiters=[];
  const selectedRuns=new Set(),initialId=new URLSearchParams(location.search).get('dataset')||U.recall('dataset');
  function status(message){$('status').textContent=message;}
  function error(value){lastError=value?.message||value||null;$('error').textContent=lastError||'';$('error').hidden=!lastError;}
  const activeJobs=()=>jobs.some(job=>(!job.kind||job.kind==='training')&&['queued','running'].includes(job.status));
  function notifyIdle(){root.dataset.busy=String(pending>0||polling||activeJobs());if(!pending&&!polling&&!activeJobs()){const waiting=idleWaiters;idleWaiters=[];waiting.forEach(resolve=>resolve());}}
  async function track(action){pending++;root.dataset.busy='true';try{return await action();}catch(cause){error(cause);}finally{pending--;root.dataset.busy=String(pending>0);notifyIdle();}}
  const whenIdle=()=>pending||polling||activeJobs()?new Promise(resolve=>idleWaiters.push(resolve)):Promise.resolve();
  const dataset=()=>datasets.find(item=>item.id===selectedDataset),model=()=>models.find(item=>item.id===$('model').value);
  const inContext=()=>model()?.training_mode==='in_context';
  function splitValue(){return {train:Number($('train-percent').value)/100,validation:Number($('validation-percent').value)/100};}
  function updateSplit(){
    const split=splitValue(),test=1-split.train-split.validation,count=dataset()?.rows||0,trainLabel=inContext()?'Context':'Training';
    const valid=Number.isFinite(test)&&split.train>0&&split.validation>0&&test>0;
    const trainCount=Math.floor(count*split.train),validationCount=Math.floor(count*split.validation),testCount=count-trainCount-validationCount;
    for(const [name,fraction,n] of [['train',split.train,trainCount],['validation',split.validation,validationCount],['test',test,testCount]]){$(name+'-bar').style.width=(valid?fraction*100:0)+'%';$(name+'-count').textContent=(name==='train'?trainLabel:U.human(name))+' '+U.number(fraction*100,1)+'%'+(count?' · ~'+U.number(Math.max(0,n)):'');}
    $('split-bar').setAttribute('aria-label',valid?U.number(split.train*100)+'% '+trainLabel.toLowerCase()+', '+U.number(split.validation*100)+'% validation, '+U.number(test*100)+'% test':'Invalid data split');
    $('validation-percent').setCustomValidity(valid?'':trainLabel+' and validation must leave at least 1% of the data for testing.');
    $('split-note').textContent=!valid?trainLabel+' and validation must leave data for testing.':'Counts are approximate; records sharing a timestamp stay together. The saved run records the exact partition counts.'+(inContext()?' The context size parameter limits how many labeled records the model retains from the context partition.':'');
    $('train').disabled=submitting||!valid||!model()?.available||!dataset();
  }
  function updateModel(){
    const item=model(),context=inContext();U.parameters($('parameters'),item?.parameters,'mt-param');
    $('model-description').textContent=[item?.description||([item?.view?U.human(item.view)+' features':null,item?.label].filter(Boolean).join(' · ')),item?.validation_note].filter(Boolean).join(' ');
    $('model-usage').textContent=item?.usage_note||'';$('model-usage').hidden=!item?.usage_note;
    $('config-title').textContent=context?'Configure model context':'Configure training';
    $('parameters-title').textContent=context?'Context and inference parameters':'Model parameters';
    $('train-label').textContent=context?'Context (%)':'Training (%)';
    $('split-description').textContent=context?'Use labeled examples from the earliest records as context for the pretrained model. Its weights stay fixed. Use the next records for validation and keep the final records for testing. Test outcomes never enter the context or threshold selection.':'Fit on the earliest records, use the next records for validation, and keep the final records for testing. Test outcomes never tune the fitted model.';
    $('train').textContent=context?'Prepare and save model':'Train and save model';
    $('train-help').textContent=context?'Context preparation and evaluation run on the server. The saved model retains its context for future predictions. You can leave this page and return to its progress or completed model.':'Training runs on the server. You can leave this page and return to its progress or completed model.';
    for(const [id,title,description,guidance] of [
      ['train-percent',context?'Context share':'Training share',context?'The earliest chronological share supplies labeled context examples for the pretrained model. Its weights remain fixed.':'The earliest chronological share supplies examples used to fit the model.',context?'A larger share offers more candidate context examples, but the context-size parameter still caps the retained set. Leave enough later records to validate and test the model.':'A larger share supplies more fitting examples but leaves fewer records for validation and testing. Keep all compared models on the same split.'],
      ['validation-percent','Validation share','The following chronological share is reserved for validation, including automatic cutoff selection and model-specific early stopping.','Too few validation records, especially too few fraud cases, can make selection unstable. The final remaining share stays held out for testing.'],
    ]){
      const input=$(id),field={name:id,label:title,type:'integer',min:Number(input.min),max:Number(input.max),default:Number(input.defaultValue),description,info:{sections:[{title:'Choosing a share',text:guidance},{title:'Timestamp boundaries',text:'Transactions sharing a timestamp stay together. The preview counts are approximate; saved runs record the exact partitions. Training and validation must leave a nonempty test partition.'}]}};
      globalThis.PaymentInfo.attach(input.closest('label'),globalThis.PaymentInfo.parameter(field));
    }
    $('model-unavailable').hidden=!item||item.available;$('model-unavailable').textContent=item?.reason||'This model is unavailable for the selected dataset.';updateSplit();
  }
  function renderDataset(){
    const item=dataset();$('dataset-panel').hidden=!item;$('no-data').hidden=!!datasets.length;$('inspect').href=U.link('data-lab.html',{dataset:item?.id});
    if(!item){$('dataset-description').textContent='Prepare a dataset in the data lab.';return;}
    $('dataset-description').textContent=U.number(item.rows)+' rows · '+U.number(U.labels(item).known)+' known labels';$('dataset-title').textContent=item.name;$('dataset-kind').textContent=item.kind==='features'?'Feature dataset':item.kind==='generator'?'Generated dataset':'Fixed dataset';U.datasetStats($('dataset-stats'),item);
    const numeric=(item.views||[]).includes('numeric'),graph=(item.views||[]).includes('graph');
    $('dataset-note').textContent=(numeric&&graph?'Numeric and graph models can use this dataset.':numeric?'This dataset supports models that use numeric features.':'Model availability is based on the dataset views.')+' Unknown labels are excluded from supervised fitting, labeled context, and label-based evaluation.';
    U.featureList($('dataset-features'),item.feature_names,item.feature_definitions);
    updateSplit();
  }
  function metricText(metrics){
    if(!metrics||typeof metrics!=='object')return 'Test metrics unavailable';
    const pairs=[['ROC AUC','roc_auc'],['PR AUC','pr_auc'],['F1','f1'],['Precision','precision'],['Recall','recall']].filter(([,key])=>metrics[key]!==undefined&&metrics[key]!==null);
    return pairs.length?pairs.map(([label,key])=>label+' '+U.number(metrics[key],3)).join(' · '):Object.entries(metrics).filter(([,value])=>typeof value==='number').slice(0,5).map(([key,value])=>U.human(key)+' '+U.number(value,3)).join(' · ')||'Test metrics unavailable';
  }
  function updateComparison(){
    const selected=runs.filter(run=>selectedRuns.has(run.id||run.run_id)&&run.dataset_id===selectedDataset);
    const ready=selected.length>0;
    $('compare').setAttribute('aria-disabled',String(!ready));$('compare').tabIndex=ready?0:-1;$('compare').href=ready?U.link('index.html',{dataset:selectedDataset,runs:selected.map(run=>run.id||run.run_id).join(',')}):'index.html';
    $('selection-note').textContent=!selected.length?'Select one or more ready models.':U.number(selected.length)+' saved '+(selected.length===1?'model':'models')+' · shared test records';
  }
  function renderRuns(){
    $('runs').replaceChildren();const visible=runs.filter(run=>run.dataset_id===selectedDataset&&(run.status==='ready'||!run.status));
    const validIds=new Set(visible.map(run=>run.id||run.run_id));for(const id of selectedRuns)if(!validIds.has(id))selectedRuns.delete(id);
    $('runs-empty').hidden=visible.length>0;$('ready-count').textContent=U.number(visible.length)+' models';
    for(const run of visible){
      const id=run.id||run.run_id,card=e('article',$('runs'),null,{class:'pl-job'}),line=e('div',card,null,{class:'pl-line'}),label=e('label',line,null,{class:'pl-check'}),checkbox=e('input',label,null,{type:'checkbox','aria-label':'Compare '+(run.name||run.label||id)});
      checkbox.checked=selectedRuns.has(id);checkbox.addEventListener('change',()=>{if(checkbox.checked)selectedRuns.add(id);else selectedRuns.delete(id);updateComparison();});e('strong',label,run.name||run.label||U.human(run.model_id));
      e('span',line,'Ready',{class:'pl-badge'});e('p',card,U.human(run.model_id)+' · '+U.date(run.created_at),{class:'pl-muted pl-small pl-gap'});
      e('p',card,metricText(run.test_metrics||run.metrics?.test||run.metrics),{class:'pl-run-metrics'});
      const details=e('details',card);e('summary',details,'Run configuration and test partition');
      const provenance=run.model_provenance,contextFacts=provenance?.context_rows===undefined?[]:[
        ['Context used',U.number(provenance.context_rows)+' of '+U.number(provenance.training_rows)+' eligible training rows'],
        ['Pretrained model',[provenance.repository,provenance.release?'release '+provenance.release:null].filter(Boolean).join(' · ')],
        ['Weights revision',provenance.revision],
      ];
      const facts=e('dl',details,null,{class:'pl-facts'});U.facts(facts,[['Run ID',id],['Split',run.split],['Exact row counts',run.partition_counts],...contextFacts,['Threshold',run.threshold===undefined?'—':U.number(run.threshold,5)+' '+(run.threshold_units||'')],['Threshold source',run.threshold_source],['Model checksum',run.model_sha256||run.checksum||'—']]);
      if(run.feature_names?.length){e('dt',facts,'Input features');U.featureList(e('dd',facts),run.feature_names,dataset()?.feature_definitions);}
      e('dt',facts,'Parameters');U.parameterList(e('dd',facts),run.parameters,models.find(item=>item.id===run.model_id)?.parameters);
      e('a',details,'Compare this model ↗',{class:'pl-text-link pl-small',href:U.link('index.html',{dataset:run.dataset_id,runs:id})});
    }
    updateComparison();
  }
  function renderJobs(){
    const expanded=new Set([...$('jobs').querySelectorAll('details[open]')].map(node=>node.dataset.job));
    $('jobs').replaceChildren();const training=jobs.filter(job=>!job.kind||job.kind==='training'),active=training.filter(job=>['queued','running'].includes(job.status));
    $('jobs-empty').hidden=training.length>0;$('active-count').textContent=active.length?U.number(active.length)+' active '+(active.length===1?'job':'jobs'):'No active jobs';
    for(const job of training.slice(0,12)){
      const card=e('article',$('jobs'),null,{class:'pl-job'}),line=e('div',card,null,{class:'pl-line'});e('strong',line,job.name||job.payload?.name||job.result?.run?.name||U.human(job.model_id||job.payload?.model_id||job.configuration?.model_id)||'Model training');e('span',line,U.human(job.status),{class:'pl-badge'});
      e('p',card,job.error?.message||job.error||job.message||U.human(job.stage)||'Waiting for the server…',{class:job.status==='failed'?'pl-error pl-gap':'pl-muted pl-small pl-gap'});
      if(['queued','running'].includes(job.status)){const progress=e('progress',card,null,{'aria-label':'Training job progress',max:'100'});if(typeof job.progress==='number')progress.value=job.progress<=1?job.progress*100:job.progress;}
      if(job.status==='succeeded'&&(job.result?.run_id||job.run_id))e('a',card,'Open completed model comparison ↗',{class:'pl-text-link pl-small',href:U.link('index.html',{dataset:job.result?.run?.dataset_id||job.dataset_id,runs:job.result?.run_id||job.run_id})});
      const details=e('details',card,null,{'data-job':job.id});details.open=expanded.has(job.id);e('summary',details,'Job details');e('p',details,job.id,{class:'pl-mono pl-gap'});
      if(job.logs?.length)e('pre',details,job.logs.map(log=>typeof log==='string'?log:(log.at?U.date(log.at)+'  ':'')+log.message).join('\n'));
    }
    notifyIdle();
  }
  async function loadModels(){
    const revision=++version,modelId=$('model').value;models=[];$('model').replaceChildren();selectedDataset=$('dataset').value||null;selectedRuns.clear();U.remember('dataset',selectedDataset||'');renderDataset();renderRuns();$('train').disabled=true;$('model').disabled=true;
    if(!selectedDataset){models=[];$('model').replaceChildren();updateModel();return;}
    const result=await U.request('/models?'+new URLSearchParams({dataset_id:selectedDataset}));if(revision!==version)return;
    models=result.models||[];$('model').replaceChildren();for(const item of models)e('option',$('model'),(item.label||U.human(item.id))+(item.available?'':' · unavailable'),{value:item.id});
    const chosen=models.find(item=>item.id===modelId&&item.available)||models.find(item=>item.available)||models[0];if(chosen)$('model').value=chosen.id;$('model').disabled=false;updateModel();renderRuns();history.replaceState(null,'',U.link('trainer.html',{dataset:selectedDataset}));
    status('Training workspace ready. Choose a model and its chronological split.');
  }
  function schedulePoll(){clearTimeout(pollTimer);if(activeJobs())pollTimer=setTimeout(poll,1200);}
  async function poll(){
    if(polling)return;polling=true;
    try{
      const result=await U.request('/jobs'),previous=new Set(jobs.filter(job=>['queued','running'].includes(job.status)).map(job=>job.id));jobs=result.jobs||[];renderJobs();
      if(jobs.some(job=>previous.has(job.id)&&job.status==='succeeded')){runs=(await U.request('/runs')).runs||[];renderRuns();status('Job completed. The saved model is ready for comparison.');}
      if(jobs.some(job=>previous.has(job.id)&&job.status==='failed'))status('A training job failed. Its error and retained job details are shown below.');
    }catch(cause){status(cause.message+' Training jobs remain on the server; retrying.');}
    finally{polling=false;schedulePoll();notifyIdle();}
  }
  async function refresh(){
    error(null);$('refresh').disabled=true;
    try{
      const [dataResult,jobResult]=await Promise.allSettled([U.request('/datasets'),U.request('/jobs')]);
      // List artifacts after job state so a newly completed job is never shown
      // without its ready model merely because two requests raced.
      const [runResult]=await Promise.allSettled([U.request('/runs')]),results=[dataResult,runResult,jobResult];
      if(dataResult.status==='fulfilled')datasets=dataResult.value.datasets||[];if(runResult.status==='fulfilled')runs=runResult.value.runs||[];if(jobResult.status==='fulfilled')jobs=jobResult.value.jobs||[];
      renderJobs();renderRuns();schedulePoll();const failure=results.find(result=>result.status==='rejected');if(failure)throw failure.reason;
      const current=selectedDataset||initialId;$('dataset').replaceChildren();for(const item of datasets)e('option',$('dataset'),item.name+' · '+U.number(item.rows)+' rows',{value:item.id});
      if(datasets.some(item=>item.id===current))$('dataset').value=current;$('fields').disabled=!datasets.length;await loadModels();
    }finally{$('refresh').disabled=false;}
  }
  async function train(){
    if(submitting)return;error(null);updateSplit();if(!$('form').reportValidity()||!model()?.available)return;
    const context=inContext(),request={dataset_id:selectedDataset,model_id:$('model').value,parameters:U.values($('parameters')),split:splitValue(),name:$('name').value.trim()||undefined};submitting=true;updateSplit();status(context?'Submitting context preparation job…':'Submitting training job…');
    try{const result=await U.request('/train',request);jobs=[result.job,...jobs.filter(job=>job.id!==result.job.id)];renderJobs();schedulePoll();status(context?'Context preparation submitted. Progress is saved on the server.':'Training submitted. Progress is saved on the server.');}
    finally{submitting=false;updateSplit();}
  }
  $('refresh').addEventListener('click',()=>track(refresh));$('dataset').addEventListener('change',()=>track(loadModels));$('model').addEventListener('change',updateModel);
  for(const id of ['train-percent','validation-percent'])$(id).addEventListener('input',updateSplit);
  $('form').addEventListener('submit',event=>{event.preventDefault();track(train);});$('compare').addEventListener('click',event=>{if($('compare').getAttribute('aria-disabled')==='true')event.preventDefault();});
  globalThis.addEventListener('pagehide',()=>clearTimeout(pollTimer));
  globalThis.addEventListener('pageshow',event=>{if(event.persisted)track(refresh);});
  root.demo={whenIdle,getSnapshot:()=>({busy:pending>0||polling||activeJobs(),error:lastError,datasetId:selectedDataset,modelId:$('model').value,models:models.map(item=>({id:item.id,available:item.available})),jobs:jobs.map(job=>({id:job.id,status:job.status,error:job.error,runId:job.result?.run_id||job.run_id})),runs:runs.filter(run=>run.dataset_id===selectedDataset),selectedRuns:[...selectedRuns]}),refresh:()=>track(refresh)};
  track(refresh);
})();
