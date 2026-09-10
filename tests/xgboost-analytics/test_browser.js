/* Real-browser offline checks. Build pages first; Playwright is an optional dev tool.
 * PLAYWRIGHT_MODULE=/path/to/playwright node tests/xgboost-analytics/test_browser.js
 * ANALYTICS_SCREENSHOTS=/tmp/screenshots additionally captures desktop/mobile views.
 */
'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const ROOT=path.resolve(__dirname,'../..');
const target=process.argv[2]||path.join(ROOT,'xgboost-analytics.html');
const snapshots=process.env.ANALYTICS_SCREENSHOTS;
async function main(){
  const browser=await chromium.launch({headless:true,args:['--no-sandbox'],...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
  const context=await browser.newContext({offline:true,viewport:{width:1440,height:1000},acceptDownloads:true});
  const page=await context.newPage(),errors=[],external=[];
  page.on('pageerror',error=>errors.push(error.message));
  page.on('request',request=>{if(/^https?:/.test(request.url()))external.push(request.url());});
  const state=()=>page.evaluate(()=>document.getElementById('xgboost-analytics').demo.getSnapshot());
  const idle=()=>page.evaluate(()=>document.getElementById('xgboost-analytics').demo.whenIdle());
  const capture=async name=>{if(snapshots){fs.mkdirSync(snapshots,{recursive:true});await page.screenshot({path:path.join(snapshots,name+'.png'),fullPage:true});}};
  const noOverflow=async()=>assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Page must not overflow the viewport; wide tables scroll internally.');
  try{
    await page.goto(pathToFileURL(path.resolve(target)).href);await idle();
    const initial=await state();assert.equal(initial.error,null);assert.ok(initial.count>1000);assert.equal(initial.count,initial.filteredCount);assert.equal(initial.featureCount,27);assert.equal(initial.configuration.trainingMode,'supervised');assert.equal(initial.configuration.mode,'shadow');assert.ok(initial.finalPosition>initial.count);
    assert.equal(await page.locator('#xa-importance-table tbody tr').count(),27);assert.equal(await page.locator('#xa-values-table tbody tr').count(),27);assert.equal(await page.locator('#xa-tree option').count(),32);await noOverflow();await capture('analytics-overview-desktop');
    assert.equal(await page.locator('input[type=range]').count(),0,'Analytics must have no replay slider.');
    await page.locator('#xa-tab-features').click();await capture('analytics-features-desktop');
    await page.locator('#xa-feature').selectOption('26');assert.match(await page.locator('#xa-definition-title').textContent(),/recipient/i);
    await page.locator('#xa-feature-search').fill('preceding 60 minutes');assert.equal(await page.locator('#xa-importance-table tbody tr').count(),8);await page.locator('#xa-feature-search').fill('');
    await page.locator('#xa-tab-decisions').click();assert.equal(await page.locator('#xa-payment-table tbody tr').count(),25);
    const first=await page.locator('#xa-payment-table tbody tr button').first().textContent();await page.locator('#xa-payment-table tbody tr button').nth(1).click();assert.notEqual((await state()).selectedId,first);
    await page.locator('#xa-tree').selectOption('31');assert.match(await page.locator('#xa-tree-summary').textContent(),/Tree 32/);assert.ok(await page.locator('#xa-tree-path li').count()>1);await capture('analytics-decisions-desktop');
    await page.locator('#xa-next-page').click();assert.equal((await state()).page,1);
    await page.locator('#xa-outcome').selectOption('fp');const fp=await state();assert.equal(fp.filteredCount,initial.metrics.fp);assert.equal(fp.metrics.fp,fp.filteredCount);assert.equal(fp.metrics.tp,0);
    const exported=await page.evaluate(()=>{const d=document.getElementById('xgboost-analytics').demo;return {json:JSON.parse(d.exportJSON()),csv:d.exportCSV()};});assert.equal(exported.json.records.length,fp.filteredCount);assert.equal(exported.json.exportedPopulation.count,fp.filteredCount);assert.equal(exported.json.featureDefinitions.length,27);assert.equal(exported.csv.trim().split('\r\n').length,fp.filteredCount+1);
    const downloadPromise=page.waitForEvent('download');await page.locator('#xa-export-csv').click();const download=await downloadPromise;assert.match(download.suggestedFilename(),/\.csv$/);assert.equal(await download.failure(),null);
    await page.locator('#xa-search').fill('no-such-payment');const empty=await state();assert.equal(empty.filteredCount,0);assert.equal(empty.selectedId,null);assert.equal(await page.locator('#xa-inspector').isVisible(),false);assert.equal(await page.locator('#xa-payment-empty').isVisible(),true);assert.equal(await page.locator('#xa-importance-table tbody tr').count(),27);
    await page.locator('#xa-reset-filters').click();assert.equal((await state()).filteredCount,initial.count);
    await page.locator('#xa-decision').selectOption('LEARNING');assert.equal((await state()).filteredCount,128);await page.locator('#xa-tab-overview').click();assert.match(await page.locator('#xa-metric-note').textContent(),/unavailable/);assert.equal((await state()).metrics.eligible,0);await page.locator('#xa-reset-filters').click();
    const unchanged=await page.evaluate(()=>{const d=document.getElementById('xgboost-analytics').demo,before=d.report;d.setFilters({query:'P'});d.selectPayment(d.report.records[5].event.id);d.setFilters({query:''});return d.report===before;});assert.equal(unchanged,true,'Filtering and selection must keep the completed report intact.');
    await page.evaluate(async()=>{const d=document.getElementById('xgboost-analytics').demo,before=d.report;const pending=d.run({size:'large',seed:918});d.cancel();await pending;if(d.report!==before)throw Error('Cancelled analysis replaced the completed report.');});assert.match(await page.locator('#xa-status').textContent(),/cancelled/);
    await page.evaluate(async()=>{const d=document.getElementById('xgboost-analytics').demo;const earlier=d.run({size:'large',seed:919});const latest=d.run({size:'small',scenario:'benign',seed:920});await Promise.all([earlier,latest]);});await idle();let current=await state();assert.equal(current.configuration.seed,920);assert.equal(current.configuration.scenario,'benign');assert.equal(current.metrics.tp+current.metrics.fn,0);
    await page.evaluate(()=>document.getElementById('xgboost-analytics').demo.run({size:'small',scenario:'relay',seed:921,warmup:10000}));await idle();current=await state();assert.equal(current.metrics.eligible,0);assert.equal(current.metrics.warmup,current.count);assert.match(await page.locator('#xa-metric-note').textContent(),/unavailable/);
    await page.evaluate(async()=>{const d=document.getElementById('xgboost-analytics').demo,original=FraudScenarios.build;FraudScenarios.build=(...args)=>{const data=original(...args);data.truth={};data.events.push({id:'ending-deposit',kind:'deposit',u:-1,v:0,amount:1,t:data.events.at(-1).t+1});return data;};try{await d.run({size:'small',scenario:'relay',seed:922});}finally{FraudScenarios.build=original;}});await idle();current=await state();assert.equal(current.metrics.labeled,0);assert.ok(current.metrics.eligible>0);assert.equal(current.metrics.unknown,current.count);assert.equal(current.error,null);assert.match(await page.locator('#xa-metric-note').textContent(),/unavailable/);
    await page.evaluate(()=>document.getElementById('xgboost-analytics').demo.run());await idle();
    await page.setViewportSize({width:390,height:844});await page.locator('#xa-tab-overview').click();await noOverflow();await capture('analytics-overview-mobile');await page.locator('#xa-tab-features').click();await noOverflow();await capture('analytics-features-mobile');await page.locator('#xa-tab-decisions').click();await noOverflow();await capture('analytics-decisions-mobile');
    await page.locator('#xa-tab-decisions').focus();await page.keyboard.press('ArrowLeft');assert.equal((await state()).tab,'features');assert.equal(await page.locator('#xa-tab-features').getAttribute('aria-selected'),'true');
    assert.deepEqual(errors,[],'The browser must not report JavaScript errors.');assert.deepEqual(external,[],'Offline pages must not request external resources.');
    console.log(JSON.stringify({test:'xgboost-analytics browser',passed:true,payments:initial.count,features:27,viewports:[1440,390],checks:['offline','completed-run','feature definitions','tree paths','filters','pagination','exports','empty population','warm-up','unknown outcomes','ending nonpayment','cancellation','stale publication','keyboard navigation','responsive layout']},null,2));
  }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
