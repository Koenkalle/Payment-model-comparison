/* Actual supervised native weights, normal controls and per-mode settings.
 * Requires the trained supervised artifact. The helper starts serve.py unless
 * NATIVE_BASE_URL points to an existing local comparison server.
 */
'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const {startServer,verify}=require('./test_native_browser');
const ROOT=path.resolve(__dirname,'../..'),ID='dyg_tami_native';
async function main(){
  const server=await startServer(),browser=await chromium.launch({headless:true,args:['--no-sandbox'],...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})}).catch(error=>{server.stop();throw error;});
  const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[],runs=[],responses=[];
  page.on('pageerror',error=>errors.push(error.message));
  page.on('response',response=>{if(new URL(response.url()).pathname==='/api/compare'&&response.ok())responses.push(response.json().then(run=>runs.push(run)));});
  const idle=async()=>{await page.evaluate(()=>document.getElementById('fraud-memory-demo').demo.whenIdle());await Promise.all(responses);};
  const state=()=>page.evaluate(()=>document.getElementById('fraud-memory-demo').demo.getSnapshot());
  const native=()=>page.evaluate(()=>document.getElementById('fraud-memory-demo').demo.getNativeSnapshot());
  const change=async(id,value)=>{await page.locator('#'+id).evaluate((element,next)=>{element.value=String(next);element.dispatchEvent(new Event('change',{bubbles:true}));},value);await idle();};
  const end=async()=>{if(!(await page.locator('#fd-run').isDisabled())){await page.locator('#fd-run').click();await idle();}};
  const check=async()=>{
    const snapshot=await native(),current=await state();assert.ok(snapshot,await page.locator('#fd-runtime-status').textContent());
    const run=runs.slice().reverse().find(candidate=>candidate.dataset.name===snapshot.dataset&&JSON.stringify(candidate.options)===JSON.stringify(snapshot.options)&&snapshot.decisions.every(row=>candidate.predictions.find(value=>value.id===row.id)?.logit===(row.evidence.fraud_logit??row.evidence.logit)));
    assert.ok(run,'Displayed native predictions need a matching real service response: '+JSON.stringify({dataset:snapshot.dataset,options:snapshot.options,decisions:snapshot.decisions.length,busy:current.busy,recent:runs.slice(-3).map(candidate=>({dataset:candidate.dataset.name,options:candidate.options,payments:candidate.predictions.length}))}));verify(snapshot,current,run);return {snapshot,current,run};
  };
  try{
    await page.goto(server.url);await idle();
    console.log('PASS local service discovery and initial comparison');
    await change('fd-mode','shadow');
    await change('fd-size','small');await change('fd-model',ID);await change('fd-model-head','fixed_likelihood');
    await change('fd-policy','individual');await change('fd-model-strategy','manual');await change('fd-model-tau',3);
    const unsupervised=await check();assert.equal(unsupervised.snapshot.head,'fixed_likelihood');
    await change('fd-training-mode','supervised');await end();
    const first=await check();assert.equal(first.current.model,ID);assert.equal(first.current.comparison.length,10);
    assert.equal(first.snapshot.head,'fraud_linear');assert.equal(first.snapshot.options.trainingMode,'supervised');
    assert.equal(first.snapshot.options.decisionPolicy,'shared','New training mode starts with independent policy settings.');
    console.log('PASS first complete supervised native run');
    assert.deepEqual(await page.locator('#fd-model-head option').evaluateAll(options=>options.map(option=>option.value)),['fraud_linear']);
    assert.match(await page.locator('#fd-training-source').textContent(),/fraud.*classifier|classifier.*fraud/i);
    assert.match(await page.locator('#fd-score-explanation').textContent(),/softplus/);
    assert.match(await page.locator('#fd-graph-method').textContent(),/proposed amount/);
    assert.doesNotMatch(await page.locator('#fd-training-source').textContent(),/Offline logistic fraud head|historical reference transactions/);
    await change('fd-model-strategy','manual');assert.equal((await native()).options.manualTau,1);
    await change('fd-model-tau',.75);await check();
    await change('fd-training-mode','unsupervised');const restored=await check();
    assert.equal(restored.snapshot.head,'fixed_likelihood');assert.equal(restored.snapshot.options.manualTau,3);
    await change('fd-training-mode','supervised');assert.equal((await native()).options.manualTau,.75);

    for(const scenario of ['takeover','mixed']){await change('fd-scenario',scenario);await end();await check();}
    await change('fd-seed',314);await end();await check();
    await change('fd-policy','shared');await change('fd-warmup',32);await check();
    await change('fd-policy','individual');await change('fd-model-strategy','tuned');assert.equal((await check()).snapshot.options.decisionPolicy,'tuned');
    await change('fd-model-strategy','auto');await change('fd-model-objective','f2');assert.equal((await check()).current.policyFit.objective,'f2');
    await change('fd-model-strategy','manual');await change('fd-model-tau',0);await change('fd-mode','shadow');const shadow=await check();
    assert.equal(shadow.snapshot.options.mode,'shadow');
    await change('fd-mode','enforce');const enforced=await check();
    assert.equal(enforced.snapshot.options.mode,'enforce');
    assert.ok(enforced.current.comparison.find(row=>row.id===ID).payments.length<shadow.current.comparison.find(row=>row.id===ID).payments.length);
    assert.ok(enforced.snapshot.decisions.some((row,index)=>row.evidence.fraud_logit!==shadow.snapshot.decisions[index].evidence.fraud_logit));
    await change('fd-mode','shadow');
    console.log('PASS supervised mode retention, two scenarios and four policies with genuine enforcement history');

    const data=JSON.parse(fs.readFileSync(path.join(ROOT,'datasets/takeover-medium-42.json'),'utf8'));
    await page.locator('#fd-dataset-file').setInputFiles({name:'supervised-payments.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(data))});await idle();await end();
    const imported=await check();assert.equal(imported.current.accounts,64);
    const original=imported.snapshot.decisions.map(row=>row.evidence.fraud_logit);
    data.truth=Object.fromEntries(Object.entries(data.truth).map(([id,label])=>[id,!label]));
    await page.locator('#fd-dataset-file').setInputFiles({name:'relabeled-payments.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(data))});await idle();await end();
    assert.deepEqual((await check()).snapshot.decisions.map(row=>row.evidence.fraud_logit),original);
    await page.locator('#fd-dataset-reset').click();await idle();
    await page.evaluate(()=>{const mode=document.getElementById('fd-training-mode');for(const value of ['unsupervised','supervised','unsupervised','supervised']){mode.value=value;mode.dispatchEvent(new Event('change',{bubbles:true}));}});
    await idle();await end();const final=await check();assert.equal(final.snapshot.options.trainingMode,'supervised');

    // Missing one artifact is a mode capability, never a fallback to another model task.
    const missing=await browser.newPage();
    await missing.route('**/api/models',async route=>{const response=await route.fetch(),catalog=await response.json();const model=catalog.models.find(value=>value.id===ID);model.capabilities.training_mode_capabilities.supervised={available:false,error:'Supervised checkpoint is missing.',prediction_heads:[],decision_policies:[],default_head:'fraud_linear'};model.capabilities.training_modes=['unsupervised'];await route.fulfill({response,json:catalog});});
    await missing.goto(server.url);await missing.evaluate(()=>document.getElementById('fraud-memory-demo').demo.whenIdle());
    await missing.locator('#fd-training-mode').selectOption('supervised');await missing.evaluate(()=>document.getElementById('fraud-memory-demo').demo.whenIdle());
    assert.equal(await missing.evaluate(()=>document.getElementById('fraud-memory-demo').demo.getNativeSnapshot()),null);
    assert.ok(await missing.locator('#fd-model option[value="'+ID+'"]').evaluate(option=>option.disabled));
    assert.match(await missing.locator('#fd-runtime-status').textContent(),/Supervised checkpoint is missing/);
    await missing.locator('#fd-training-mode').selectOption('unsupervised');await missing.evaluate(()=>document.getElementById('fraud-memory-demo').demo.whenIdle());
    assert.ok(await missing.evaluate(()=>document.getElementById('fraud-memory-demo').demo.getNativeSnapshot()));await missing.close();

    const proposed=final.run.dataset.events.findLastIndex(event=>event.kind==='payment');
    await page.locator('#fd-step').evaluate((element,index)=>{element.value=String(index);element.dispatchEvent(new Event('input',{bubbles:true}));},proposed);await idle();await check();
    assert.match(await page.locator('#fd-parts').textContent(),/Native fraud logit/);
    assert.match(await page.locator('#fd-parts').textContent(),/Estimated fraud probability/);
    assert.doesNotMatch(await page.locator('#fd-parts').textContent(),/Native link|historical rank|frozen reference transactions/);
    await page.locator('#fd-truth').evaluate(element=>{element.checked=true;element.dispatchEvent(new Event('change',{bubbles:true}));});
    if(process.env.NATIVE_SCREENSHOTS){fs.mkdirSync(process.env.NATIVE_SCREENSHOTS,{recursive:true});await page.screenshot({path:path.join(process.env.NATIVE_SCREENSHOTS,'native-supervised-comparison.png'),fullPage:true});}
    await page.setViewportSize({width:390,height:844});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
    if(process.env.NATIVE_SCREENSHOTS)await page.screenshot({path:path.join(process.env.NATIVE_SCREENSHOTS,'native-supervised-mobile.png'),fullPage:true});
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({test:'native supervised comparison browser',passed:true,models:10,requests:runs.length,checks:['fraud probabilities','stable scores','per-mode head and policy retention','scenarios and imports','four policies','enforced history','label isolation','rapid mode cancellation','missing artifact capability','responsive layout']},null,2));
  }finally{await browser.close();server.stop();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
