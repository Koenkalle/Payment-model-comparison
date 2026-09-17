/* Check the displayed identity against real default and custom artifact responses. */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const {startServer}=require('./test_native_browser');
const ROOT=path.resolve(__dirname,'../..'),ID='dyg_tami_native';

async function main(){
  const artifact=fs.mkdtempSync(path.join(os.tmpdir(),'comparison-custom-checkpoint-'));
  const original=path.join(ROOT,'models/native-dyg-tami-fraud');
  for(const file of ['model.npz','manifest.json'])fs.copyFileSync(path.join(original,file),path.join(artifact,file));
  const calibration=JSON.parse(fs.readFileSync(path.join(original,'calibration.config.json')));
  calibration.dataset.path=path.resolve(original,calibration.dataset.path);
  fs.writeFileSync(path.join(artifact,'calibration.config.json'),JSON.stringify(calibration));
  const saved=JSON.parse(fs.readFileSync(path.join(artifact,'manifest.json')));
  let server,browser;
  try{
    server=await startServer(['--supervised-artifact',artifact]);
    browser=await chromium.launch({headless:true,args:['--no-sandbox'],...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
    const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
    page.on('pageerror',error=>errors.push(error.message));
    const idle=()=>page.evaluate(()=>document.getElementById('fraud-memory-demo').demo.whenIdle());
    const change=async(id,value)=>{await page.locator('#'+id).evaluate((element,next)=>{element.value=String(next);element.dispatchEvent(new Event('change',{bubbles:true}));},value);await idle();};
    const text=id=>page.locator('#'+id).textContent();
    await page.goto(server.url);await idle();
    console.log('PASS native discovery with custom supervised artifact');
    await change('fd-size','small');await change('fd-model',ID);
    const link=JSON.parse(fs.readFileSync(path.join(ROOT,'models/native-dyg-tami/manifest.json')));
    assert.equal(await text('fd-checkpoint-source'),'Default artifact location');
    assert.equal(await text('fd-checkpoint-id'),link.model_sha256);
    assert.equal(await text('fd-checkpoint-path'),path.join(ROOT,'models/native-dyg-tami'));
    await change('fd-training-mode','supervised');
    assert.equal(await text('fd-checkpoint-source'),'Custom artifact');
    assert.equal(await text('fd-checkpoint-id'),saved.model_sha256);
    assert.equal(await text('fd-checkpoint-path'),artifact);
    assert.match(await text('fd-checkpoint-training'),new RegExp('Fraud-label training.*Best epoch '+saved.best_epoch));
    assert.match(await text('fd-checkpoint-help'),/--supervised-artifact/);
    assert.equal(await page.locator('#fd-checkpoint').getAttribute('aria-busy'),'false');
    const nativeRow=page.locator('#fd-compare tbody tr').filter({hasText:'DyGFormer + TAMI · native'});
    assert.match(await nativeRow.textContent(),new RegExp(saved.model_sha256.slice(0,12)));
    assert.match(await nativeRow.textContent(),/Custom artifact/);
    await page.getByText('Evaluate against known outcomes',{exact:true}).click();
    await page.locator('#fd-truth').check();
    assert.match(await page.locator('#fd-model-metrics').textContent(),new RegExp(saved.model_sha256.slice(0,12)));
    // Selecting another model must never leave the native model's identity in
    // the selected-model panel. Its row remains identifiable in the comparison.
    await change('fd-model','statistics');
    assert.equal(await text('fd-checkpoint-source'),'Bundled browser checkpoint');
    assert.notEqual(await text('fd-checkpoint-id'),saved.model_sha256);
    assert.match(await nativeRow.textContent(),/Custom artifact/);
    await change('fd-model',ID);
    await change('fd-training-mode','unsupervised');
    assert.equal(await text('fd-checkpoint-id'),link.model_sha256);
    await change('fd-training-mode','supervised');
    assert.equal(await text('fd-checkpoint-id'),saved.model_sha256);
    // A failed update excludes the native model and clears its selected identity.
    await page.route('**/api/compare',route=>route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({error:'Checkpoint unavailable for this test'})}));
    await change('fd-seed',314);
    assert.match(await text('fd-runtime-status'),/Checkpoint unavailable/);
    assert.equal(await text('fd-checkpoint-source'),'Bundled browser checkpoint');
    await page.unroute('**/api/compare');await change('fd-seed',42);await change('fd-model',ID);
    const capture=async(name)=>{if(process.env.CHECKPOINT_SCREENSHOTS){fs.mkdirSync(process.env.CHECKPOINT_SCREENSHOTS,{recursive:true});await page.screenshot({path:path.join(process.env.CHECKPOINT_SCREENSHOTS,name+'.png'),fullPage:true});}};
    await capture('checkpoint-desktop');
    await page.setViewportSize({width:375,height:900});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Checkpoint paths must wrap on mobile.');
    assert.ok(await page.locator('#fd-checkpoint-id').isVisible());await capture('checkpoint-mobile');
    assert.deepEqual(errors,[]);
    console.log('PASS checkpoint identities: actual custom/default artifacts, training modes, selected models, comparison rows, failure recovery and mobile layout.');
  }finally{
    if(browser)await browser.close();if(server)server.stop();fs.rmSync(artifact,{recursive:true,force:true});
  }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
