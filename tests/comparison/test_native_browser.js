/* Real PyTorch service and browser integration through the ordinary settings.
 * Build first, then run with PLAYWRIGHT_MODULE=/path/to/playwright node this-file.
 * Set NATIVE_BASE_URL to reuse a server, otherwise this test starts serve.py.
 */
'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {spawn}=require('node:child_process');
const {pathToFileURL}=require('node:url');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const ROOT=path.resolve(__dirname,'../..'),NATIVE_ID='dyg_tami_native';

function close(actual,expected,message){assert.ok(Math.abs(actual-expected)<=1e-11*Math.max(1,Math.abs(expected)),`${message}: expected ${expected}, got ${actual}`);}
function verify(snapshot,state,run){
  assert.ok(snapshot,'The native model must remain in the comparison.');
  assert.equal(state.comparison.length,10);
  assert.equal(snapshot.head,run.head);
  assert.deepEqual(snapshot.options,run.options);
  const byId=new Map(run.predictions.map(row=>[row.id,row]));
  let tau=run.policy_state.tau;const warmup=[];
  for(const row of snapshot.decisions){
    const expected=byId.get(row.id),logit=expected.logit;
    if(run.options.trainingMode==='supervised'){
      assert.equal(row.evidence.kind,'native-fraud');assert.equal(row.evidence.fraud_logit,logit);
      close(row.evidence.fraud_probability,logit>=0?1/(1+Math.exp(-logit)):Math.exp(logit)/(1+Math.exp(logit)),'Estimated fraud probability');
      close(row.score,(Math.max(logit,0)+Math.log1p(Math.exp(-Math.abs(logit))))/Math.LN2,'Stable fraud score');
      assert.equal(Object.hasOwn(row.evidence,'tail_probability'),false);
    }else{
      assert.equal(row.evidence.logit,logit);
      const tail=run.head==='empirical_tail'?(1+run.heads.empirical_tail.reference_logits.filter(value=>value<=logit).length)/(run.heads.empirical_tail.reference_logits.length+1):Math.max(Number.MIN_VALUE,logit>=0?1/(1+Math.exp(-logit)):Math.exp(logit)/(1+Math.exp(logit)));
      close(row.evidence.tail_probability,tail,'Independent likelihood head');close(row.score,-Math.log2(tail),'Anomaly bits');
    }
    if(tau===null)assert.equal(row.tau,null);else close(row.tau,tau,'Chronological policy threshold');
    const decision=!expected.evaluationEligible?'CONTEXT':tau===null?'LEARNING':row.score>tau?'BLOCK':'ALLOW';
    assert.equal(row.decision,decision);
    if(run.options.decisionPolicy==='shared'&&expected.evaluationEligible){
      if(decision==='LEARNING'){
        warmup.push(row.score);
        if(warmup.length>=run.options.warmup)tau=warmup.slice().sort((a,b)=>a-b)[Math.ceil((1-run.options.alpha)*warmup.length)-1];
      }else tau+=run.options.eta*((decision==='BLOCK'?1:0)-run.options.alpha);
    }
  }
  const nativeState=state.comparison.find(row=>row.id===NATIVE_ID);
  assert.deepEqual(nativeState.payments,snapshot.decisions.filter(row=>run.options.mode==='shadow'||row.decision!=='BLOCK').map(row=>row.id));
  const cohort=snapshot.metrics[0].evaluation_ids;
  for(const metric of snapshot.metrics)assert.deepEqual(metric.evaluation_ids,cohort,'Every model uses the same evaluated requests.');
}

async function startServer(){
  if(process.env.NATIVE_BASE_URL)return {url:process.env.NATIVE_BASE_URL,stop(){}};
  const child=spawn(process.env.PYTHON_BINARY||'python',['serve.py','--port','0'],{cwd:ROOT,stdio:['ignore','pipe','pipe']});
  let output='';
  const url=await new Promise((resolve,reject)=>{
    const timeout=setTimeout(()=>reject(Error('Native service did not start: '+output)),60000);
    const consume=chunk=>{output+=chunk;const match=output.match(/Comparison: (http:\/\/[^\s]+)/);if(match){clearTimeout(timeout);resolve(match[1]);}};
    child.stdout.on('data',consume);child.stderr.on('data',consume);
    child.once('error',error=>{clearTimeout(timeout);reject(error);});
    child.once('exit',code=>{clearTimeout(timeout);reject(Error('Native service exited '+code+': '+output));});
  }).catch(error=>{child.kill();throw error;});
  return {url,stop(){child.kill();}};
}

async function main(){
  const server=await startServer();
  const browser=await chromium.launch({headless:true,args:['--no-sandbox'],...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})}).catch(error=>{server.stop();throw error;});
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[],runs=[],pendingResponses=[];
  page.on('pageerror',error=>errors.push(error.message));
  page.on('response',response=>{if(new URL(response.url()).pathname==='/api/compare'&&response.ok())pendingResponses.push(response.json().then(run=>runs.push(run)));});
  const idle=async()=>{await page.evaluate(()=>document.getElementById('fraud-memory-demo').demo.whenIdle());await Promise.all(pendingResponses);};
  const state=()=>page.evaluate(()=>document.getElementById('fraud-memory-demo').demo.getSnapshot());
  const native=()=>page.evaluate(()=>document.getElementById('fraud-memory-demo').demo.getNativeSnapshot());
  const change=async(id,value)=>{
    await page.locator('#'+id).evaluate((element,next)=>{element.value=String(next);element.dispatchEvent(new Event('change',{bubbles:true}));},value);await idle();
  };
  const runToEnd=async()=>{await page.locator('#fd-run').click();await idle();};
  const activeRun=snapshot=>runs.slice().reverse().find(run=>JSON.stringify(run.options)===JSON.stringify(snapshot.options)&&run.dataset.name===snapshot.dataset&&snapshot.decisions.every(row=>run.predictions.find(value=>value.id===row.id)?.logit===row.evidence.logit));
  const check=async()=>{const snapshot=await native(),current=await state();assert.ok(snapshot,'Native service failure: '+await page.locator('#fd-runtime-status').textContent());const run=activeRun(snapshot);assert.ok(run,'The displayed results must come from a matching real service response.');verify(snapshot,current,run);return {snapshot,current,run};};
  const overflow=async()=>assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Page must fit the viewport.');
  try{
    await page.goto(server.url);await idle();
    assert.equal(await page.locator('#fd-native-file, #fd-native-demo, #fd-native-head').count(),0,'No separate native dataset or settings flow.');
    assert.equal(await page.locator('#fd-model option').count(),10);
    assert.ok(await page.locator('#fd-model option[value="'+NATIVE_ID+'"]').isEnabled());
    await page.locator('#fd-model').selectOption(NATIVE_ID);await idle();
    assert.ok(await page.locator('#fd-mode').isEnabled());
    assert.ok(await page.locator('#fd-training-mode').isEnabled());
    await change('fd-size','small');await change('fd-warmup','32');
    await change('fd-mode','shadow');await runToEnd();await check();

    const scenarios=await page.locator('#fd-scenario option').evaluateAll(options=>options.map(option=>option.value));
    const signatures=new Set();
    for(const scenario of scenarios){
      await change('fd-scenario',scenario);await runToEnd();
      const {run}=await check();signatures.add(JSON.stringify(run.dataset.events));
      console.log('PASS native scenario '+scenario+' through the normal controls');
    }
    assert.equal(signatures.size,scenarios.length);
    await change('fd-seed',314);await runToEnd();const reseeded=await check();
    assert.ok(!signatures.has(JSON.stringify(reseeded.run.dataset.events)),'A new seed must generate and score a different dataset.');
    await change('fd-size','large');await runToEnd();const large=await check();
    assert.equal(large.current.accounts,112);
    assert.ok(large.run.predictions.length>reseeded.run.predictions.length);
    await change('fd-size','small');await change('fd-forward',31);await change('fd-delay',6);await runToEnd();await check();

    const baseline=(await native()).decisions.map(row=>row.evidence.logit);
    await change('fd-model-head','fixed_likelihood');await check();
    assert.deepEqual((await native()).decisions.map(row=>row.evidence.logit),baseline,'Changing only the head in shadow mode preserves native encoder output.');
    console.log('PASS native dataset size, seed, timing and swappable head');
    await change('fd-alpha',.05);const budget=await check();assert.equal(budget.snapshot.options.alpha,.05);
    await change('fd-policy','individual');await change('fd-model-strategy','manual');await change('fd-model-tau',0);
    const manual=await check();assert.equal(manual.snapshot.options.decisionPolicy,'manual');assert.equal(manual.snapshot.options.manualTau,0);
    assert.ok(manual.snapshot.decisions.some(row=>row.decision==='BLOCK'));
    await change('fd-mode','enforce');const enforced=await check();
    assert.ok(enforced.current.comparison.find(row=>row.id===NATIVE_ID).payments.length<manual.current.comparison.find(row=>row.id===NATIVE_ID).payments.length);
    assert.ok(enforced.snapshot.decisions.some((row,index)=>row.evidence.logit!==manual.snapshot.decisions[index].evidence.logit),'Blocking must change subsequent real native inference.');
    await change('fd-mode','shadow');await change('fd-model-strategy','tuned');const tuned=await check();assert.equal(tuned.snapshot.options.decisionPolicy,'tuned');
    await change('fd-model-missed-cost',50);await check();
    await change('fd-model-strategy','auto');await change('fd-model-objective','f2');const automatic=await check();assert.equal(automatic.current.policyFit.objective,'f2');
    await change('fd-policy','auto');await check();
    console.log('PASS shared, individual, cost and automatic policies with real enforcement history');

    await change('fd-policy','shared');await change('fd-model-head','empirical_tail');
    const imported=JSON.parse(fs.readFileSync(path.join(ROOT,'datasets/benign-large-42.json'),'utf8'));
    await page.locator('#fd-dataset-file').setInputFiles({name:'new-payments.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(imported))});await idle();await runToEnd();
    const loaded=await check();assert.equal(loaded.current.accounts,imported.accounts.length);
    assert.match(await page.locator('#fd-dataset-status').textContent(),/Imported/);
    const importLogits=loaded.snapshot.decisions.map(row=>row.evidence.logit);
    const relabeled=JSON.parse(JSON.stringify(imported));relabeled.truth=Object.fromEntries(Object.entries(relabeled.truth).map(([id,value])=>[id,!value]));
    await page.locator('#fd-dataset-file').setInputFiles({name:'relabeled-payments.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(relabeled))});await idle();await runToEnd();
    assert.deepEqual((await check()).snapshot.decisions.map(row=>row.evidence.logit),importLogits);
    console.log('PASS ordinary dataset import and evaluation-label isolation');
    await page.locator('#fd-dataset-reset').click();await idle();

    await page.evaluate(()=>{const seed=document.getElementById('fd-seed');for(const value of [71,72,73]){seed.value=String(value);seed.dispatchEvent(new Event('change',{bubbles:true}));}});
    await idle();await runToEnd();const latest=await check();
    assert.equal(await page.locator('#fd-seed').inputValue(),'73');
    const expected=await page.evaluate(()=>globalThis.FraudDatasets.load('synthetic_payments',{name:document.getElementById('fd-scenario').value,size:document.getElementById('fd-size').value,seed:73,reportDelay:Number(document.getElementById('fd-delay').value)*60,forwardDelay:Number(document.getElementById('fd-forward').value)}));
    assert.deepEqual(latest.run.dataset.events,expected.events);
    await page.locator('#fd-truth').evaluate(element=>{element.checked=true;element.dispatchEvent(new Event('change',{bubbles:true}));});
    await overflow();
    if(process.env.NATIVE_SCREENSHOTS){fs.mkdirSync(process.env.NATIVE_SCREENSHOTS,{recursive:true});await page.screenshot({path:path.join(process.env.NATIVE_SCREENSHOTS,'native-unified-comparison.png'),fullPage:true});}
    await page.setViewportSize({width:390,height:844});await overflow();
    if(process.env.NATIVE_SCREENSHOTS)await page.screenshot({path:path.join(process.env.NATIVE_SCREENSHOTS,'native-unified-mobile.png'),fullPage:true});
    const offline=await browser.newPage({offline:true});
    await offline.goto(pathToFileURL(path.join(ROOT,'index.html')).href);
    await offline.evaluate(()=>document.getElementById('fraud-memory-demo').demo.whenIdle());
    assert.equal(await offline.evaluate(()=>document.getElementById('fraud-memory-demo').demo.getSnapshot().comparison.length),9);
    assert.ok(await offline.locator('#fd-model option[value="'+NATIVE_ID+'"]').evaluate(option=>option.disabled));
    assert.match(await offline.locator('#fd-runtime-status').textContent(),/python serve\.py/);
    await offline.close();
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({test:'native comparison unified browser flow',passed:true,models:10,scenarios,requests:runs.length,checks:['actual PyTorch inference','all scenarios','size and seed changes','ordinary JSON import','head swapping','shared/manual/cost/auto policies','real enforcement history','truth isolation','latest settings win','common evaluation population','mobile layout']},null,2));
  }finally{await browser.close();server.stop();}
}
module.exports={startServer,verify};
if(require.main===module)main().catch(error=>{console.error(error);process.exitCode=1;});
