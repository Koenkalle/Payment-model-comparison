/* Catalog-driven trainer UI: in-context preparation uses the existing job API. */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs/promises'),path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const ROOT=path.resolve(__dirname,'../..');

async function main(){
  const [template,info,client,ui]=await Promise.all([
    fs.readFile(path.join(ROOT,'tools/model-trainer/template.html'),'utf8'),
    fs.readFile(path.join(ROOT,'shared/ui/info-windows.js'),'utf8'),
    fs.readFile(path.join(ROOT,'tools/pipeline/client.js'),'utf8'),
    fs.readFile(path.join(ROOT,'tools/model-trainer/ui.js'),'utf8'),
  ]);
  const browser=await chromium.launch({headless:true,args:['--no-sandbox'],...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
  try{
    const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[],requests=[];
    page.on('pageerror',error=>errors.push(error.message));
    const data={id:'numeric-example',name:'Stored tabular examples',kind:'fixed',rows:200,known:200,fraud:40,views:['numeric']};
    const tabfm={
      id:'tabfm',label:'TabFM',view:'numeric',available:true,training_mode:'in_context',
      description:'Pretrained tabular foundation model using labeled context with fixed weights.',
      usage_note:'Google pretrained weights are for non-commercial, non-production use. The first run downloads the weights; saved runs retain labeled context.',
      parameters:{
        max_context_rows:{type:'integer',default:100,min:2,max:2048},
        n_estimators:{type:'integer',default:1,min:1,max:8},
        inference_batch_size:{type:'integer',default:128,min:1,max:256},
        seed:{type:'integer',default:42,min:0,max:2147483647},
        decision_threshold:{type:'number',default:null,nullable:true,min:0,max:1},
      },
    };
    const logistic={id:'logistic_regression',label:'Logistic regression',view:'numeric',available:true,parameters:{C:{type:'number',default:1,min:.01,max:100}}};
    let savedRun=null,job=null;
    // Serve the editable trainer sources; no pretrained weights or local server
    // are required to check capability metadata and the browser/API boundary.
    await page.route('https://pipeline.test/**',async route=>{
      const request=route.request(),url=new URL(request.url());
      if(url.pathname==='/trainer.html')return route.fulfill({contentType:'text/html; charset=utf-8',body:'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"></head><body>'+template+'<script>'+info+'</script><script>'+client+'</script><script>'+ui+'</script></body></html>'});
      let response;
      if(url.pathname==='/api/pipeline/datasets')response={datasets:[data]};
      else if(url.pathname==='/api/pipeline/models'){
        assert.equal(url.searchParams.get('dataset_id'),data.id);response={models:[logistic,tabfm]};
      }else if(url.pathname==='/api/pipeline/runs')response={runs:savedRun?[savedRun]:[]};
      else if(url.pathname==='/api/pipeline/jobs')response={jobs:job?[{...job,status:'succeeded',result:{run_id:savedRun.id}}]:[]};
      else if(url.pathname==='/api/pipeline/train'){
        assert.equal(request.method(),'POST');const payload=request.postDataJSON();requests.push(payload);
        savedRun={...payload,id:'tabfm-run',status:'ready',partition_counts:{train:100,validation:40,test:60},test_metrics:{roc_auc:.8},
          model_provenance:{context_rows:40,training_rows:95,repository:'google/tabfm-1.0.0-pytorch',release:'1.0.0',revision:'77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83'}};
        job={id:'context-job',kind:'training',status:'queued',payload};response={job};
      }else throw Error('Unexpected browser request: '+request.url());
      await route.fulfill({contentType:'application/json',body:JSON.stringify(response)});
    });
    const idle=async()=>{await page.waitForFunction(()=>!!document.getElementById('model-trainer')?.demo);await page.evaluate(()=>document.getElementById('model-trainer').demo.whenIdle());};
    await page.goto('https://pipeline.test/trainer.html?dataset='+data.id);await idle();
    assert.equal(await page.locator('#mt-model').inputValue(),'logistic_regression');
    assert.equal(await page.locator('#mt-model-usage').isVisible(),false);
    await page.locator('#mt-model').selectOption('tabfm');
    assert.match(await page.locator('#mt-model-description').textContent(),/foundation model/);
    assert.match(await page.locator('#mt-model-usage').textContent(),/non-commercial, non-production/);
    assert.match(await page.locator('#mt-split-description').textContent(),/weights stay fixed/);
    assert.match(await page.locator('#mt-split-description').textContent(),/Test outcomes never enter the context/);
    assert.equal(await page.locator('#mt-train-label').textContent(),'Context (%)');
    assert.match(await page.locator('#mt-train-count').textContent(),/^Context 60%/);
    assert.match(await page.locator('#mt-split-bar').getAttribute('aria-label'),/^60% context/);
    assert.equal(await page.locator('#mt-train').textContent(),'Prepare and save model');
    await page.locator('#mt-param-max_context_rows').fill('40');
    await page.locator('#mt-train-percent').fill('50');
    await page.locator('#mt-name').fill('TabFM context 40');
    await page.locator('#mt-train').click();await idle();
    assert.deepEqual(requests,[{dataset_id:data.id,model_id:'tabfm',parameters:{max_context_rows:40,n_estimators:1,inference_batch_size:128,seed:42,decision_threshold:null},split:{train:.5,validation:.2},name:'TabFM context 40'}]);
    assert.equal(await page.locator('#mt-runs input[type=checkbox]').count(),1);
    await page.locator('#mt-runs details summary').click();
    assert.match(await page.locator('#mt-runs details').textContent(),/40 of 95 eligible training rows/);
    assert.match(await page.locator('#mt-runs details').textContent(),/google\/tabfm-1\.0\.0-pytorch · release 1\.0\.0/);
    assert.match(await page.locator('#mt-runs details').textContent(),/77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83/);
    await page.locator('#mt-runs input[type=checkbox]').check();
    assert.match(await page.locator('#mt-compare').getAttribute('href'),/runs=tabfm-run/);
    assert.equal(await page.locator('#mt-compare').getAttribute('aria-disabled'),'false');
    await page.setViewportSize({width:390,height:844});
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Context controls and usage notes fit mobile.');
    await page.locator('#mt-model').selectOption('logistic_regression');
    assert.equal(await page.locator('#mt-train').textContent(),'Train and save model');
    assert.equal(await page.locator('#mt-train-label').textContent(),'Training (%)');
    assert.match(await page.locator('#mt-train-count').textContent(),/^Training 50%/);
    assert.match(await page.locator('#mt-split-description').textContent(),/^Fit on the earliest records/);
    assert.equal(await page.locator('#mt-model-usage').isVisible(),false);
    assert.equal(await page.locator('#mt-param-max_context_rows').count(),0);
    tabfm.available=false;tabfm.reason='TabFM requires Python 3.11+ and requirements-tabfm.txt. Install the optional dependencies and refresh.';
    await page.locator('#mt-refresh').click();await idle();await page.locator('#mt-model').selectOption('tabfm');
    assert.equal(await page.locator('#mt-train').isDisabled(),true);
    assert.match(await page.locator('#mt-model-unavailable').textContent(),/requirements-tabfm\.txt/);
    assert.equal(await page.locator('#mt-model-usage').isVisible(),true,'The license remains visible before optional setup.');
    assert.deepEqual(errors,[]);
    console.log('PASS TabFM trainer browser: dynamic catalog, context split, license, typed job submission, ready comparison, model switching, optional setup and mobile.');
  }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
