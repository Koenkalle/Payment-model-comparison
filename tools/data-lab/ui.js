(function(){
  'use strict';
  const root=document.getElementById('data-lab');if(!root)return;
  const U=globalThis.PaymentPipelineUI,$=id=>root.querySelector('#dl-'+id),e=U.element;
  const graph=new globalThis.DatasetGraphExplorer($('graph-panel'),{closeId:'dl-graph-close',onClose:()=>{$('graph-open').setAttribute('aria-expanded','false');$('graph-open').focus();}});
  let catalog={datasets:[],sources:[],generators:[],limits:{}},selected=null,detail=null,pending=0,selectionVersion=0,lastError=null,idleWaiters=[];
  let featureCatalog=null,featureSelection=new Set(),featureRevision=0,featureLoading=false,featureSaving=false,featureError=null;
  const initialId=new URLSearchParams(location.search).get('dataset')||U.recall('dataset');
  function status(message){$('status').textContent=message;}
  function error(value){lastError=value?.message||value||null;$('error').textContent=lastError||'';$('error').hidden=!lastError;}
  async function track(action){pending++;root.dataset.busy='true';try{return await action();}catch(cause){error(cause);status('The action could not be completed. Check the message above and retry.');}finally{pending--;root.dataset.busy=String(pending>0);if(!pending){const waiting=idleWaiters;idleWaiters=[];waiting.forEach(resolve=>resolve());}}}
  const whenIdle=()=>pending?new Promise(resolve=>idleWaiters.push(resolve)):Promise.resolve();
  function updateSource(){const source=catalog.sources.find(item=>item.id===$('source').value);$('source-description').textContent=source?.description||'';$('currency-field').hidden=!source?.requires_conversion;$('currency').required=!!source?.requires_conversion&&$('method').value==='import';$('currency').disabled=!$('currency').required;}
  function updateGenerator(){const generator=catalog.generators.find(item=>item.id===$('generator').value);$('generator-description').textContent=generator?.description||'';U.parameters($('parameters'),generator?.parameters,'dl-param');}
  function updateMethod(){const importing=$('method').value==='import';$('generator-section').hidden=importing;$('import-section').hidden=!importing;$('file').required=importing;$('create').textContent=importing?'Import and store dataset':'Generate and store dataset';for(const input of $('generator-section').querySelectorAll('input,select'))input.disabled=importing;for(const input of $('import-section').querySelectorAll('input,select'))input.disabled=!importing;updateSource();}
  function renderLibrary(){
    $('library').replaceChildren();$('library-empty').hidden=catalog.datasets.length>0;$('library-count').textContent=U.number(catalog.datasets.length)+' datasets';
    for(const dataset of catalog.datasets){
      const row=e('tr',$('library'),null,{'aria-selected':String(dataset.id===selected)}),cell=e('td',row),button=e('button',cell,dataset.name,{type:'button',class:'pl-text-link'});
      button.addEventListener('click',()=>track(()=>inspect(dataset.id)));
      e('div',cell,U.human(dataset.source||dataset.generator),{class:'pl-muted pl-help'});e('td',row,U.number(dataset.rows));e('td',row,dataset.kind==='features'?'Feature dataset':dataset.kind==='generator'?'Generated':'Fixed');
    }
  }
  function featureIds(){return [...new Set([...(featureCatalog?.enabled_features||[]),...(featureCatalog?.features||[]).map(feature=>feature.id)])].filter(id=>featureSelection.has(id));}
  function featureDirty(){const saved=featureCatalog?.enabled_features||[];return featureSelection.size!==saved.length||saved.some(id=>!featureSelection.has(id));}
  const graphFeature=feature=>feature.kind!=='source'&&feature.group==='graph';
  const paymentFeature=feature=>feature.kind!=='source'&&!graphFeature(feature);
  function updateFeatureState(){
    const dirty=featureDirty(),available=(featureCatalog?.features||[]).filter(feature=>feature.available!==false),saved=featureCatalog?.enabled_features||[];
    $('feature-count').textContent=featureCatalog?U.number(featureSelection.size)+' selected · '+U.number(available.length)+' available':'';
    $('feature-fields').disabled=featureLoading||featureSaving;
    $('feature-save').disabled=!dirty||!featureSelection.size||featureLoading||featureSaving;
    $('feature-reset').disabled=!dirty||featureLoading||featureSaving;
    $('feature-add').disabled=!available.some(feature=>paymentFeature(feature)&&!featureSelection.has(feature.id));
    $('feature-add-graph').disabled=!available.some(feature=>graphFeature(feature)&&!featureSelection.has(feature.id));
    $('feature-state').dataset.dirty=String(dirty);
    $('feature-state').textContent=featureSaving?'Calculating and storing the selected features…':dirty?(featureSelection.size?'Unsaved selection: '+U.number(featureSelection.size)+' columns. Save or discard changes before training.':'No columns selected. Select at least one feature to save.'):'Saved selection: '+U.number(saved.length)+' columns ready for training.';
    const trainBlocked=dirty||featureSaving;
    $('train').setAttribute('aria-disabled',String(trainBlocked));$('train').tabIndex=trainBlocked?-1:0;
    $('train').title=trainBlocked?'Save or discard feature changes before training.':'Train models using this saved dataset.';
    for(const row of $('feature-list').children){const enabled=featureSelection.has(row.dataset.featureId);row.dataset.enabled=String(enabled);row.querySelector('input').checked=enabled;}
  }
  function featureMetric(parent,label,value){const item=e('div',parent);e('dt',item,label);e('dd',item,value);}
  function featureNumber(value,digits=5){return typeof value==='number'&&value!==0&&Math.abs(value)<10**-digits?value.toExponential(3):U.number(value,digits);}
  function renderFeatureInformation(parent,feature){
    e('p',parent,feature.info?.description||feature.description||'Original numeric column from the source dataset.',{class:'pl-muted pl-small'});
    for(const section of feature.info?.sections||[]){e('h3',parent,section.title,{class:'pl-space'});e('p',parent,section.text,{class:'pl-muted pl-small'});}
    const facts=e('dl',parent,null,{class:'pl-facts'}),requirements=feature.requirements;
    U.facts(facts,[['Origin',feature.kind==='source'?'Source column':'Derived · '+U.human(feature.group||'payment')],['Calculation',feature.transform||'Preserved from the source dataset'],['Unit',feature.unit||'Numeric'],...(feature.method?[['Method',feature.method]]:[]),...(feature.orientation?[['Graph orientation',feature.orientation]]:[]),...(feature.readout?[['Readout',feature.readout]]:[]),...(feature.parameters?[['Parameters',feature.parameters]]:[]),...(feature.approximation?[['Approximation',feature.approximation]]:[]),...(feature.window?[['History window',feature.window]]:[]),...(requirements?.length?[['Requires',Array.isArray(requirements)?requirements.join(', '):requirements]]:[]),...(feature.amount_bins?[['Amount bins',feature.amount_bins.join(', ')]]:[]),...(feature.amount_unit_assumption?[['Bin currency',feature.amount_unit_assumption]]:[]),...(feature.version?[['Definition',String(feature.version)]]:[])]);
    if(feature.references?.length){
      const references=Array.isArray(feature.references)?feature.references:[feature.references],links=e('dd');
      for(const reference of references){const url=typeof reference==='string'?reference:reference.url;if(!/^https?:\/\//i.test(url||''))continue;if(links.childNodes.length)e('span',links,' · ');e('a',links,typeof reference==='string'?reference:reference.title||url,{href:url,target:'_blank',rel:'noopener noreferrer',class:'pl-text-link'});}
      if(links.childNodes.length){e('dt',facts,'References');facts.append(links);}
    }
    const stats=feature.statistics;
    if(!stats){e('p',parent,feature.unavailable_reason||'Statistics are unavailable for this feature.',{class:'pl-muted pl-small pl-gap'});return;}
    const metrics=e('dl',parent,null,{class:'dl-feature-metrics'}),missing=Number(stats.missing||0),rows=featureCatalog.row_count||0;
    for(const [label,value] of [['Minimum',featureNumber(stats.min)],['Maximum',featureNumber(stats.max)],['Mean',featureNumber(stats.mean)],['Std. deviation',featureNumber(stats.std)],['Valid values',U.number(stats.count)],['Missing',U.number(missing)+(rows?' ('+U.number(100*missing/rows,1)+'%)':'')],['Distinct values',U.number(stats.unique)],['Median',featureNumber(stats.quantiles?.p50)]])featureMetric(metrics,label,value);
    if(stats.histogram?.length){
      const distribution=e('div',parent),peak=Math.max(...stats.histogram.map(bin=>bin.count),1);
      e('div',distribution,'Value distribution · '+U.number(featureCatalog.row_count)+' rows',{class:'pl-help'});
      const bars=e('div',distribution,null,{class:'dl-feature-distribution',role:'img','aria-label':stats.histogram.map(bin=>featureNumber(bin.min,4)+' to '+featureNumber(bin.max,4)+': '+U.number(bin.count)+' rows').join('; ')});
      for(const bin of stats.histogram){const bar=e('span',bars,null,{class:'dl-feature-bar',title:featureNumber(bin.min)+'–'+featureNumber(bin.max)+': '+U.number(bin.count)+' rows'});bar.style.height=100*bin.count/peak+'%';}
      const scale=e('div',distribution,null,{class:'dl-feature-scale'});e('span',scale,featureNumber(stats.min));e('span',scale,featureNumber(stats.max));
    }
    e('p',parent,'Middle 50%: '+featureNumber(stats.quantiles?.p25)+'–'+featureNumber(stats.quantiles?.p75)+'. Standard deviation uses all valid rows.',{class:'pl-help pl-gap'});
  }
  function renderFeatureList(){
    if(!featureCatalog)return;
    const search=$('feature-search').value.trim().toLowerCase(),filter=$('feature-filter').value;
    const expanded=new Set([...$('feature-list').querySelectorAll('details[open]')].map(node=>node.parentElement.dataset.featureId));
    const features=featureCatalog.features.filter(feature=>(filter==='all'||(filter==='enabled'?featureSelection.has(feature.id):filter==='graph'?graphFeature(feature):feature.kind===filter))&&[feature.id,feature.label,feature.description,feature.group,feature.method].some(value=>String(value||'').toLowerCase().includes(search)));
    $('feature-list').replaceChildren();$('feature-empty').hidden=features.length>0;
    $('feature-empty').textContent=featureCatalog.features.length?'No features match this search.':'No numeric model input features are available for this dataset.';
    $('feature-visible').textContent=U.number(features.length)+' of '+U.number(featureCatalog.features.length)+' features · Expand a feature for its calculation and statistics.';
    for(const feature of features){
      const label=feature.label||U.human(feature.id),row=e('div',$('feature-list'),null,{class:'dl-feature','data-feature-id':feature.id});
      const input=e('input',row,null,{type:'checkbox','aria-label':'Include '+label,'data-feature-id':feature.id});input.checked=featureSelection.has(feature.id);input.disabled=feature.available===false;
      input.addEventListener('change',()=>{if(input.checked)featureSelection.add(feature.id);else featureSelection.delete(feature.id);updateFeatureState();if($('feature-filter').value==='enabled')renderFeatureList();});
      const disclosure=e('details',row),summary=e('summary',disclosure),heading=e('span',summary,label,{class:'dl-feature-heading'});
      e('span',heading,feature.kind==='source'?'Source':graphFeature(feature)?'Graph':'Derived',{class:'pl-badge'});
      e('span',summary,feature.id,{class:'dl-feature-name pl-mono pl-muted'});
      globalThis.PaymentInfo.attach(summary,globalThis.PaymentInfo.feature(feature),{buttonParent:heading});
      if(feature.available===false){const reasonId='dl-unavailable-'+featureCatalog.features.indexOf(feature);input.setAttribute('aria-describedby',reasonId);e('span',summary,feature.unavailable_reason||'Required source data is unavailable.',{id:reasonId,class:'dl-feature-unavailable'});}
      let populated=false;const populate=()=>{if(populated||!disclosure.open)return;populated=true;renderFeatureInformation(e('div',disclosure,null,{class:'dl-feature-details'}),feature);};
      disclosure.addEventListener('toggle',populate);if(expanded.has(feature.id)){disclosure.open=true;populate();}
    }
    updateFeatureState();
  }
  function featureFailure(cause){featureError=cause?.message||String(cause);$('feature-error').textContent=featureError;$('feature-error').hidden=false;$('feature-retry').hidden=!!featureCatalog;}
  async function loadFeatures(id=selected,selection=selectionVersion){
    const revision=++featureRevision;featureLoading=true;featureError=null;$('feature-error').hidden=true;$('feature-retry').hidden=true;$('feature-loading').hidden=false;updateFeatureState();
    try{
      const result=await U.request('/datasets/'+encodeURIComponent(id)+'/features');if(selection!==selectionVersion||revision!==featureRevision)return;
      featureCatalog=result;featureCatalog.features||=[];featureCatalog.enabled_features||=featureCatalog.features.filter(feature=>feature.enabled).map(feature=>feature.id);
      featureSelection=new Set(featureCatalog.enabled_features);$('feature-fields').hidden=false;
      const newer=featureCatalog.latest_recipe_version&&featureCatalog.latest_recipe_version!==featureCatalog.recipe_version;
      $('feature-update').hidden=!newer;$('feature-update').replaceChildren();
      if(newer){$('feature-update').textContent='New feature definitions are available. This saved dataset keeps its original calculations. ';e('a',$('feature-update'),'Open the original source to choose from the latest features.',{href:U.link('data-lab.html',{dataset:featureCatalog.source_dataset_id}),class:'pl-text-link'});}
      $('feature-replay').textContent=featureCatalog.replay_note||'Payment history features use only previously observed transactions, before the current payment. Labels are excluded from calculations.';
      $('feature-name').placeholder=(detail?.dataset.name||'Dataset')+' · features';renderFeatureList();renderPreviewFeatures();
    }catch(cause){if(selection===selectionVersion&&revision===featureRevision)featureFailure(cause);}
    finally{if(selection===selectionVersion&&revision===featureRevision){featureLoading=false;$('feature-loading').hidden=true;updateFeatureState();}}
  }
  async function saveFeatures(){
    if(!featureCatalog||!featureDirty()||!featureSelection.size||featureSaving)return;
    if(!$('feature-name').reportValidity())return;
    const id=selected,selection=selectionVersion,body={features:featureIds(),name:$('feature-name').value.trim()||undefined};
    featureSaving=true;featureError=null;$('feature-error').hidden=true;updateFeatureState();
    try{
      const result=await U.request('/datasets/'+encodeURIComponent(id)+'/features',body);
      if(selection!==selectionVersion)return;
      featureSaving=false;const refreshed=await refresh(result.dataset.id,selection);if(refreshed!==false)status('Saved “'+result.dataset.name+'”. Its selected feature columns are ready for model training.');
    }catch(cause){if(selection===selectionVersion)featureFailure(cause);}
    finally{if(selection===selectionVersion){featureSaving=false;updateFeatureState();}}
  }
  function renderDetail(){
    const dataset=detail.dataset,sample=detail.sample||{};
    $('inspector-empty').hidden=true;$('inspector').hidden=false;$('preview').hidden=false;$('inspector-title').textContent=dataset.name;
    $('train').href=U.link('trainer.html',{dataset:dataset.id});U.datasetStats($('stats'),dataset);$('views').replaceChildren();
    for(const view of dataset.views||[])e('span',$('views'),U.human(view)+' view',{class:'pl-badge'});
    const supportsGraph=(dataset.views||[]).some(view=>view==='graph'||view==='labeled_graph');
    $('graph-open').disabled=!supportsGraph;$('graph-reason').textContent=supportsGraph?'Explore a bounded slice and expand connections.':'Graph view unavailable: this dataset has no stable relational identities.';
    graph.setDataset(dataset);$('graph-open').setAttribute('aria-expanded','false');
    U.facts($('facts'),[['Source',U.human(dataset.source||dataset.generator)],['Stored',U.date(dataset.created_at)],['Time range',dataset.time_range?U.number(dataset.time_range.start,3)+'–'+U.number(dataset.time_range.end,3)+' '+(dataset.time_range.unit||'seconds'):'—'],['Dataset ID',dataset.id]]);
    if(dataset.parent_dataset_id){e('dt',$('facts'),'Derived from');e('a',e('dd',$('facts')),catalog.datasets.find(item=>item.id===dataset.parent_dataset_id)?.name||dataset.parent_dataset_id,{href:U.link('data-lab.html',{dataset:dataset.parent_dataset_id}),class:'pl-text-link'});}
    if(dataset.source_dataset_id&&dataset.source_dataset_id!==dataset.id){e('dt',$('facts'),'Original source');const origin=e('dd',$('facts'));e('a',origin,catalog.datasets.find(item=>item.id===dataset.source_dataset_id)?.name||dataset.source_dataset_id,{href:U.link('data-lab.html',{dataset:dataset.source_dataset_id}),class:'pl-text-link'});e('div',origin,'Open the source to create a dataset using the latest available feature definitions.',{class:'pl-help'});}
    if(dataset.feature_recipe_version){e('dt',$('facts'),'Feature recipe');e('dd',$('facts'),dataset.feature_recipe_version);}
    const settings=Object.entries(dataset.parameters||{}).map(([key,value])=>[U.human(key),value]);U.facts($('settings'),settings);$('no-settings').hidden=settings.length>0;
    $('provenance').textContent=JSON.stringify({fingerprint:dataset.fingerprint,schema:dataset.schema,...dataset.provenance},null,2);
    const columns=sample.columns||[],rows=sample.rows||[];
    $('sample-head').replaceChildren();$('sample-body').replaceChildren();const header=e('tr',$('sample-head'));
    for(const name of ['ID','Time','Outcome',...columns])e('th',header,name,{scope:'col'});
    renderPreviewFeatures();
    for(const row of rows){const tr=e('tr',$('sample-body'));const values=Array.isArray(row.features)?row.features:columns.map(name=>row.features?.[name]);for(const value of [row.id,U.number(row.time,3),row.label===1?'Fraud':row.label===0?'Legitimate':'Unknown',...values])e('td',tr,typeof value==='number'?featureNumber(value):value??'—');}
    $('preview-count').textContent=U.number(rows.length)+' of '+U.number(dataset.rows)+' rows · '+U.number(columns.length)+' features';
  }
  function renderPreviewFeatures(){
    const definitions=new Map((featureCatalog?.features||detail?.dataset.feature_definitions||[]).map(feature=>[feature.id,feature]));
    const headers=[...$('sample-head').querySelectorAll('th')].slice(3);
    for(const [index,name] of (detail?.sample.columns||[]).entries()){
      const feature=definitions.get(name);if(feature&&headers[index])globalThis.PaymentInfo.attach(headers[index],globalThis.PaymentInfo.feature(feature));
    }
  }
  async function inspect(id){
    const revision=++selectionVersion;selected=id;detail=null;U.remember('dataset',id);renderLibrary();error(null);status('Loading dataset preview…');
    featureRevision++;featureCatalog=null;featureSelection=new Set();featureLoading=false;featureSaving=false;featureError=null;
    $('features').hidden=true;$('feature-fields').hidden=true;$('feature-update').hidden=true;$('feature-list').replaceChildren();$('feature-search').value='';$('feature-filter').value='all';$('feature-name').value='';updateFeatureState();
    graph.setDataset(null);$('graph-open').setAttribute('aria-expanded','false');
    $('inspector').hidden=true;$('preview').hidden=true;$('inspector-empty').hidden=false;$('inspector-empty').textContent='Loading dataset settings and records…';
    try{const result=await U.request('/datasets/'+encodeURIComponent(id));if(revision!==selectionVersion)return;detail=result;renderDetail();$('features').hidden=false;history.replaceState(null,'',U.link('data-lab.html',{dataset:id}));status('Dataset ready. Inspect its shared features or continue to model training.');await loadFeatures(id,revision);}
    catch(cause){if(revision!==selectionVersion)return;$('inspector-empty').textContent='The preview could not be loaded. Select the dataset or refresh to try again.';throw cause;}
  }
  async function refresh(preferred,expectedSelection){
    error(null);$('refresh').disabled=true;
    try{
      catalog=await U.request('/datasets');catalog.datasets||=[];catalog.sources||=[];catalog.generators||=[];catalog.limits||={};
      if(expectedSelection!==undefined&&expectedSelection!==selectionVersion){renderLibrary();return false;}
      const generatorId=$('generator').value,sourceId=$('source').value,parameterValues=new Map([...$('parameters').querySelectorAll('input,select')].map(input=>[input.name,input.value]));
      for(const [id,items] of [['generator',catalog.generators],['source',catalog.sources]]){$(id).replaceChildren();for(const item of items)e('option',$(id),item.label||U.human(item.id),{value:item.id});}
      if(catalog.generators.some(item=>item.id===generatorId))$('generator').value=generatorId;else if(catalog.generators.some(item=>item.id==='handbook_generator'))$('generator').value='handbook_generator';if(catalog.sources.some(item=>item.id===sourceId))$('source').value=sourceId;
      updateGenerator();if($('generator').value===generatorId)for(const input of $('parameters').querySelectorAll('input,select'))if(parameterValues.has(input.name))input.value=parameterValues.get(input.name);updateMethod();$('create-fields').disabled=false;
      $('upload-limit').textContent='CSV upload limit: '+U.number((catalog.limits.max_upload_bytes||6291456)/1048576,1)+' MiB.';
      const id=[preferred,selected,initialId,catalog.datasets[0]?.id].find(candidate=>catalog.datasets.some(item=>item.id===candidate));renderLibrary();
      if(id)await inspect(id);else status('The library is ready. Generate or import a dataset to begin.');
    }finally{$('refresh').disabled=false;}
  }
  async function create(){
    error(null);const importing=$('method').value==='import';
    if(!$('create-form').reportValidity())return;
    const body={name:$('name').value.trim()||undefined};let path;
    if(importing){
      const file=$('file').files[0];if(!file)throw Error('Choose a CSV file to import.');
      if(file.size>(catalog.limits.max_upload_bytes||6291456))throw Error('This file exceeds the '+U.number((catalog.limits.max_upload_bytes||6291456)/1048576,1)+' MiB upload limit.');
      body.source=$('source').value;body.name||=file.name;body.release=$('release').value.trim()||undefined;
      if(!$('currency-field').hidden)body.amount_to_eur=Number($('currency').value);
      body.csv=await file.text();path='/datasets/import';
    }else{body.generator=$('generator').value;body.parameters=U.values($('parameters'));path='/datasets/generate';}
    $('create-fields').disabled=true;status(importing?'Importing and storing dataset…':'Generating and storing dataset…');
    try{const result=await U.request(path,body);await refresh(result.dataset.id);status('Stored “'+result.dataset.name+'”. Inspect it below or continue to model training.');}
    finally{$('create-fields').disabled=false;}
  }
  $('refresh').addEventListener('click',()=>track(()=>refresh()));$('generator').addEventListener('change',()=>{updateGenerator();updateMethod();});$('source').addEventListener('change',updateSource);$('method').addEventListener('change',updateMethod);
  $('create-form').addEventListener('submit',event=>{event.preventDefault();track(create);});
  $('feature-search').addEventListener('input',renderFeatureList);$('feature-filter').addEventListener('change',renderFeatureList);
  for(const [id,predicate] of [['feature-add',feature=>featureSelection.has(feature.id)||paymentFeature(feature)],['feature-add-graph',feature=>featureSelection.has(feature.id)||graphFeature(feature)],['feature-all',()=>true],['feature-source',feature=>feature.kind==='source']])$(id).addEventListener('click',()=>{featureSelection=new Set(featureCatalog.features.filter(feature=>feature.available!==false&&predicate(feature)).map(feature=>feature.id));renderFeatureList();});
  $('feature-reset').addEventListener('click',()=>{featureSelection=new Set(featureCatalog.enabled_features);featureError=null;$('feature-error').hidden=true;renderFeatureList();});
  $('feature-save').addEventListener('click',()=>track(saveFeatures));$('feature-retry').addEventListener('click',()=>track(()=>loadFeatures()));
  $('train').addEventListener('click',event=>{if(featureDirty()||featureSaving)event.preventDefault();});
  $('graph-open').addEventListener('click',()=>{graph.open();$('graph-open').setAttribute('aria-expanded','true');$('graph-panel').scrollIntoView({behavior:'smooth',block:'start'});});
  root.demo={whenIdle:async()=>{await whenIdle();await graph.whenIdle();},graph,getSnapshot:()=>({busy:pending>0,error:lastError,selectedDatasetId:selected,datasets:catalog.datasets.map(item=>({id:item.id,name:item.name,rows:item.rows,kind:item.kind})),detail:detail?.dataset||null,features:{datasetId:featureCatalog?.dataset_id||null,selected:featureIds(),saved:featureCatalog?.enabled_features||[],dirty:featureDirty(),loading:featureLoading,saving:featureSaving,error:featureError,catalog:featureCatalog?.features||[]}}),refresh:()=>track(()=>refresh()),selectDataset:id=>track(()=>inspect(id))};
  track(()=>refresh());
})();
