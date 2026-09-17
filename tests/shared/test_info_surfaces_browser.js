/* Fresh source-template integration: no generated pages or model training.
 * PLAYWRIGHT_MODULE=/path/to/playwright node tests/shared/test_info_surfaces_browser.js
 */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const ROOT=path.resolve(__dirname,'../..'),registry=require('../../tools/registry.json');
const source=file=>fs.readFileSync(path.join(ROOT,file),'utf8');
function pageSource(id){
  const tool=registry.tools.find(tool=>tool.id===id),bundle=JSON.parse(source('model-bundle.json'));
  if(tool.model_ids!=='all'){bundle.models=bundle.models.filter(model=>tool.model_ids.includes(model.id));bundle.default=bundle.models[0].id;}
  if(tool.explanation)bundle.explanation=JSON.parse(source(tool.explanation));
  if(tool.native_models)bundle.native_models=require('../../models/registry.json').models.filter(model=>tool.native_models.includes(model.id));
  const data='<script type="application/json" id="'+tool.model_data_id+'">'+JSON.stringify(bundle).replace(/</g,'\\u003c')+'</script>';
  let fragment=source(tool.template),at=Math.max(fragment.lastIndexOf('</div>'),fragment.lastIndexOf('</main>'));
  fragment=fragment.includes('<!-- TOOL_DATA -->')?fragment.replace('<!-- TOOL_DATA -->',()=>data):fragment.slice(0,at)+data+fragment.slice(at);
  const scripts=[...require('../../shared/runtime/plugin-scripts')(),...registry.shared_scripts,...tool.scripts];
  fragment+=scripts.map(file=>'<script>'+source(file)+'</script>').join('');
  return source('shared/ui/standalone-shell.html').replace('<!-- SHARED_UI_SCRIPTS -->',()=>registry.ui_scripts.map(file=>'<script>'+source(file)+'</script>').join('')).replace('<!-- FRAUD_DEMO_FRAGMENT -->',()=>fragment);
}
async function main(){
  const browser=await chromium.launch({headless:true,args:['--no-sandbox'],...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
  const page=await browser.newPage({viewport:{width:1440,height:1050}}),errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  const dataset={id:'dataset-info',name:'Feature audit dataset',rows:40,views:['numeric'],feature_names:['graph_sender_pagerank'],feature_definitions:[{id:'graph_sender_pagerank',label:'Sender PageRank',description:'Importance from earlier payment links.',method:'Sparse historical PageRank',approximation:'Numerical accuracy applies to the stored snapshot.',unit:'probability'}]};
  const run={id:'run-info',name:'Saved audit model',model_id:'xgboost_native',dataset_id:dataset.id,dataset_name:dataset.name,feature_names:dataset.feature_names,parameters:{trees:24},created_at:'2026-01-01',partition_counts:{train:24,validation:8,test:8}};
  const model={run_id:run.id,label:run.name,model_id:run.model_id,probability_threshold:.4,metrics:{precision:1,recall:.5,f1:2/3,roc_auc:.8,average_precision:.8,fp:0,fn:1,block_rate:.2}};
  const result={id:'comparison-info',dataset_id:dataset.id,dataset_name:dataset.name,population:'common-held-out-test',row_count:1,known:1,unknown:0,models:[model],rows:[{id:'payment-info',timestamp_seconds:100,label:1,predictions:{[run.id]:{probability:.7,decision:'BLOCK'}}}]};
  const job={id:'job-info',kind:'comparison',status:'succeeded',created_at:'2026-01-01',payload:{dataset_id:dataset.id,run_ids:[run.id],partition:'test'},result};
  const responses={
    '/api/models':{models:[]},'/api/datasets':{datasets:[]},
    '/api/pipeline/datasets':{datasets:[dataset]},'/api/pipeline/runs':{runs:[run]},'/api/pipeline/jobs':{jobs:[job]},'/api/pipeline/jobs/job-info':{job},
    '/api/pipeline/models':{models:[{id:run.model_id,parameters:[{name:'trees',label:'Trees',type:'integer',default:100,min:1,info:{description:'Number of trees fitted in the saved model.',sections:[{title:'Tradeoff',text:'More trees increase fitting work.'}]}}]}]},
  };
  await page.route('http://info.test/**',route=>{
    const url=new URL(route.request().url());
    if(url.pathname==='/comparison')return route.fulfill({contentType:'text/html',body:pageSource('comparison')});
    if(url.pathname==='/analytics')return route.fulfill({contentType:'text/html',body:pageSource('xgboost-analytics')});
    return route.fulfill({contentType:'application/json',body:JSON.stringify(responses[url.pathname]||{})});
  });
  const popup=page.locator('.payment-info-window');
  const dismiss=async()=>{await page.keyboard.press('Escape');await page.mouse.move(0,0);};
  const show=async(selector,expected)=>{
    await dismiss();await page.locator(selector).first().hover();await popup.waitFor({state:'visible'});
    assert.match(await popup.textContent(),expected);await dismiss();
  };
  try{
    await page.goto('http://info.test/analytics');
    await page.evaluate(()=>document.getElementById('xgboost-analytics').demo.whenIdle());
    assert.equal(await page.locator('#xgboost-analytics').getAttribute('data-error'),null);
    assert.equal(await page.locator('[data-info-parameter]').count(),11);
    const before=await page.evaluate(()=>{window.originalReport=document.getElementById('xgboost-analytics').demo.report;return document.getElementById('xa-policy').value;});
    await show('label[for="xa-policy"]',/Decision policy.*How it is used.*historical validation/s);
    await show('label[for="xa-alpha"]',/Target block rate.*guaranteed.*Displayed value.*2/s);
    await page.locator('#xa-tab-features').click();
    await show('#xa-importance-table .xa-link',/Feature ID.*Transform.*History window/s);
    await show('#xa-importance-table tbody tr:first-child .xa-beeswarm circle:last-of-type',/Payment.*Readable value.*SHAP contribution/s);
    await show('#xa-dependence-chart circle:last-of-type',/Feature attribution.*Model input/s);
    await page.locator('#xa-tab-decisions').click();
    await show('#xa-values-table tbody tr:first-child td:nth-child(3)',/Model input.*SHAP contribution/s);
    await show('#xa-waterfall rect[data-payment-info]',/margin|contribution|SHAP/i);
    await show('#xa-tree-chart g[data-info-tree-node="root"]',/Tree split.*Split threshold.*Selected payment input/s);
    assert.equal(await page.evaluate(()=>window.originalReport===document.getElementById('xgboost-analytics').demo.report),true,'Inspecting help keeps the completed report unchanged.');
    assert.equal(await page.locator('#xa-policy').inputValue(),before);
    await page.setViewportSize({width:390,height:844});
    await page.locator('#xa-tree-chart g[data-info-tree-node="root"]').focus();await popup.waitFor({state:'visible'});
    const box=await popup.boundingBox();assert(box.x>=0&&box.x+box.width<=391);await dismiss();

    await page.setViewportSize({width:1440,height:1050});
    await page.goto('http://info.test/comparison?job=job-info');
    await page.evaluate(async()=>{await document.getElementById('fraud-memory-demo').demo.whenIdle();await document.getElementById('trained-comparison').demo.whenIdle();});
    assert.equal(await page.locator('#tc-result').isVisible(),true);
    await show('label[for="tc-partition"]',/Evaluation data.*intersection.*same transaction IDs/s);
    await page.locator('#tc-runs details summary').click();
    await show('#tc-runs [data-feature-id="graph_sender_pagerank"]',/Sender PageRank.*Sparse historical PageRank.*snapshot/s);
    await show('#tc-runs [data-parameter-name="trees"]',/Number of trees fitted.*Tradeoff.*Saved value.*24/s);
    await page.locator('#tc-metrics details summary').click();
    await show('#tc-metrics [data-feature-id="graph_sender_pagerank"]',/Sparse historical PageRank/);
    await show('#tc-metrics tbody tr:first-child td:last-child',/Saved probability cutoff.*Probability cutoff.*0.4/s);
    await show('label[for="fd-policy"]',/Policy scope.*separately learned cutoffs/s);
    await page.locator('#fd-parts').evaluate(table=>{for(let parent=table.parentElement;parent;parent=parent.parentElement)if(parent.tagName==='DETAILS')parent.open=true;});
    await show('#fd-parts [data-info-readout="Amount bin"]',/Checkpoint amount category.*zero-based integer category.*not divided.*Amount boundaries \(EUR\)/s);
    await show('#fd-parts [data-info-readout="Sender activity gap"]',/Checkpoint timing category.*without a logarithmic transform.*separate first-activity category.*Gap boundaries \(minutes\)/s);
    await page.locator('#fd-model').selectOption('xgboost');await page.evaluate(()=>document.getElementById('fraud-memory-demo').demo.whenIdle());
    await page.locator('#fd-parts details summary').click();
    await show('#fd-parts [data-feature-id="sender_gap"]',/Sender activity gap.*History window/s);
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({test:'comparison and XGBoost extended info',passed:true,checks:['fresh source templates','parameter controls','feature values and attributions','chart marks','tree splits','saved run settings','saved comparison feature lists','mobile popup','unchanged predictions']}));
  }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
