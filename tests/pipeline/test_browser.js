/* Real browser: saved data -> actual model fits -> shared evaluation -> restart. */
'use strict';
const assert=require('node:assert/strict'),path=require('node:path'),os=require('node:os'),fs=require('node:fs/promises');
const {spawn}=require('node:child_process'),{once}=require('node:events');
const {pathToFileURL}=require('node:url');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const ROOT=path.resolve(__dirname,'../..');

async function start(directory){
  const code="import sys\nfrom types import SimpleNamespace\nfrom framework.comparison_service import make_server\nfrom framework.pipeline_service import PipelineService\np=PipelineService(sys.argv[1])\ns=make_server(SimpleNamespace(models=lambda: {'models':[]}),port=0,pipeline_service=p)\nprint(s.server_port,flush=True)\ns.serve_forever()";
  const child=spawn(process.env.PYTHON_BINARY||'python',['-c',code,directory],{cwd:ROOT,stdio:['ignore','pipe','pipe']});
  let logs='';child.stderr.on('data',chunk=>logs+=chunk);
  const port=await new Promise((resolve,reject)=>{
    const timeout=setTimeout(()=>{child.kill();reject(Error('Pipeline server startup timeout: '+logs));},30000);
    child.stdout.once('data',chunk=>{clearTimeout(timeout);resolve(Number(String(chunk).trim()));});
    child.once('error',error=>{clearTimeout(timeout);reject(error);});
    child.once('exit',code=>{clearTimeout(timeout);reject(Error('Pipeline server exited '+code+': '+logs));});
  });
  return {child,base:'http://127.0.0.1:'+port,logs:()=>logs};
}
async function stop(server){if(server?.child.exitCode===null){const exited=once(server.child,'exit');server.child.kill();await exited;}}

async function main(){
  const directory=await fs.mkdtemp(path.join(os.tmpdir(),'payment-pipeline-browser-'));let server,browser;
  try{
    server=await start(directory);browser=await chromium.launch({headless:true,args:['--no-sandbox']});
    const page=await browser.newPage({viewport:{width:1440,height:1050}}),errors=[];
    page.on('pageerror',error=>errors.push(error.message));
    const idle=async id=>{await page.waitForFunction(id=>!!document.getElementById(id)?.demo,id);await page.evaluate(id=>document.getElementById(id).demo.whenIdle(),id);};
    const snapshot=id=>page.evaluate(id=>document.getElementById(id).demo.getSnapshot(),id);
    await page.goto(server.base+'/data-lab.html');await idle('data-lab');
    assert.equal((await snapshot('data-lab')).datasets.length,1,'Fixed dataset is already stored.');
    await page.locator('#dl-generator').selectOption('handbook_generator');
    await page.locator('#dl-param-transactions').fill('200');await page.locator('#dl-param-fraud_rate').fill('0.3');
    await page.locator('#dl-param-customer_count').fill('8');await page.locator('#dl-param-terminal_count').fill('4');
    await page.locator('#dl-name').fill('Browser benchmark');await page.locator('#dl-create').click();await idle('data-lab');
    const data=await snapshot('data-lab');assert.equal(data.error,null);assert.equal(data.detail.rows,200);assert.equal(data.detail.parameters.customer_count,8);
    assert.equal(await page.locator('#dl-sample-body tr').count(),20);
    const datasetId=data.selectedDatasetId;
    if(process.env.PIPELINE_SCREENSHOTS)await page.screenshot({path:path.join(process.env.PIPELINE_SCREENSHOTS,'pipeline-data.png'),fullPage:true});
    await page.locator('#dl-train').click();await idle('model-trainer');
    assert.equal((await snapshot('model-trainer')).datasetId,datasetId);
    await page.locator('#mt-model').selectOption('logistic_regression');
    await page.locator('#mt-param-C').fill('0.2');await page.locator('#mt-name').fill('Logistic C=0.2');
    await page.locator('#mt-train').click();await idle('model-trainer');
    let training=await snapshot('model-trainer');assert.equal(training.error,null);assert.equal(training.runs.length,1,JSON.stringify(training));
    assert.equal(training.runs[0].parameters.class_weight,null,'Null select values retain their type.');
    await page.locator('#mt-param-C').fill('2');await page.locator('#mt-train-percent').fill('50');
    await page.locator('#mt-param-decision_threshold').fill('0.4');
    await page.locator('#mt-name').fill('Logistic C=2');await page.locator('#mt-train').click();await idle('model-trainer');
    training=await snapshot('model-trainer');assert.equal(training.runs.length,2,JSON.stringify(training));
    assert.equal(training.runs[0].split.train,.5);
    assert.equal(training.runs[0].parameters.decision_threshold,.4);
    const native=training.models.find(row=>row.id==='dyg_tami_native');
    if(native?.available){
      await page.locator('#mt-model').selectOption('dyg_tami_native');
      await page.locator('#mt-param-epochs').fill('1');await page.locator('#mt-param-num_heads').selectOption('1');
      await page.locator('#mt-param-max_input_sequence_length').selectOption('8');
      await page.locator('#mt-name').fill('DyGFormer one epoch');await page.locator('#mt-train').click();await idle('model-trainer');
      training=await snapshot('model-trainer');assert.equal(training.runs.length,3,JSON.stringify(training));
      assert.equal(training.runs[0].parameters.num_heads,1,'Numeric select parameters retain their type.');
    }
    for(const checkbox of await page.locator('#mt-runs input[type=checkbox]').all())await checkbox.check();
    assert.equal(await page.locator('#mt-compare').getAttribute('aria-disabled'),'false','Different splits may use their common heldout population.');
    if(process.env.PIPELINE_SCREENSHOTS)await page.screenshot({path:path.join(process.env.PIPELINE_SCREENSHOTS,'pipeline-trainer.png'),fullPage:true});
    await page.locator('#mt-compare').click();await idle('trained-comparison');
    assert.equal((await snapshot('trained-comparison')).run_ids.length,training.runs.length);
    await page.locator('#tc-compare').click();await idle('trained-comparison');
    let comparison=await snapshot('trained-comparison');
    assert.equal(comparison.job.status,'succeeded',JSON.stringify(comparison.job));
    assert.equal(comparison.result.models.length,training.runs.length);assert.equal(comparison.result.row_count,40);
    assert.equal(comparison.result.training_rows_in_evaluation,0);assert.equal(comparison.result.validation_rows_in_evaluation,0);
    assert(comparison.result.rows.every(row=>Object.keys(row.predictions).length===training.runs.length));
    assert.equal(await page.locator('#tc-metrics tbody tr').count(),training.runs.length);
    await page.locator('#tc-next').click();assert.match(await page.locator('#tc-page').textContent(),/26–40 of 40/);
    const comparisonPath=new URL(page.url()).pathname+new URL(page.url()).search;
    const downloaded=page.waitForEvent('download',{timeout:5000});await page.locator('#tc-download').click();
    const report=JSON.parse(await fs.readFile(await (await downloaded).path(),'utf8'));
    assert.equal(report.row_count,40);assert.equal(report.models.length,training.runs.length);
    // Exercise explicit rejection of in-sample evaluation through the UI.
    await page.locator('#tc-partition').selectOption('all');await page.locator('#tc-compare').click();await idle('trained-comparison');
    assert.equal((await snapshot('trained-comparison')).job.status,'failed');
    assert.match(await page.locator('#tc-status').textContent(),/held.out|training|original|separate|same/i);
    await page.goto(server.base+comparisonPath);await idle('trained-comparison');
    assert.equal((await snapshot('trained-comparison')).result.row_count,40,'Saved result survives page reload.');
    if(process.env.PIPELINE_SCREENSHOTS)await page.screenshot({path:path.join(process.env.PIPELINE_SCREENSHOTS,'pipeline-comparison.png'),fullPage:false});
    for(const [file,id] of [['data-lab.html','data-lab'],['trainer.html','model-trainer'],['index.html','trained-comparison']]){
      await page.setViewportSize({width:390,height:844});await page.goto(server.base+'/'+file+'?dataset='+datasetId);await idle(id);
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),file+' fits mobile.');
    }
    // Fixed numeric-only data can enter the same training workspace.
    await page.goto(server.base+'/data-lab.html');await idle('data-lab');
    await page.locator('#dl-method').selectOption('import');await page.locator('#dl-source').selectOption('ulb');
    const header=['Time',...Array.from({length:28},(_,i)=>'V'+(i+1)),'Amount','Class'];
    const csv=header.join(',')+'\n'+Array.from({length:100},(_,i)=>[i,...Array.from({length:28},(_,j)=>i%2+j),i+1,i%2].join(',')).join('\n');
    await page.locator('#dl-file').setInputFiles({name:'ulb-smoke.csv',mimeType:'text/csv',buffer:Buffer.from(csv)});
    await page.locator('#dl-create').click();await idle('data-lab');assert.equal((await snapshot('data-lab')).error,null);
    await page.locator('#dl-train').click();await idle('model-trainer');
    const tabular=await snapshot('model-trainer');assert.equal(tabular.models.find(row=>row.id==='dyg_tami_native').available,false);
    await page.locator('#mt-model').selectOption('logistic_regression');await page.locator('#mt-train').click();await idle('model-trainer');
    assert.equal((await snapshot('model-trainer')).runs.length,1);
    await stop(server);server=await start(directory);
    await page.goto(server.base+comparisonPath);await idle('trained-comparison');
    comparison=await snapshot('trained-comparison');assert.equal(comparison.result.row_count,40,'Dataset, runs and results survive server restart.');
    assert.deepEqual(errors,[]);
    await page.goto(pathToFileURL(path.join(ROOT,'data-lab.html')).href);await idle('data-lab');
    assert.match((await snapshot('data-lab')).error,/python serve.py/);
    console.log('PASS pipeline browser: fixed library, parameterized generation, inspection, real model training, mixed model comparison, splits, rejected overlap, ULB, mobile, reload, server restart and offline guidance.');
  }catch(error){if(server)console.error(server.logs().slice(-8000));throw error;}
  finally{await browser?.close();await stop(server);await fs.rm(directory,{recursive:true,force:true});}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
