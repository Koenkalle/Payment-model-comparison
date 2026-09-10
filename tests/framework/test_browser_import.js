/* Offline file-picker integration for both dataset consumers. */
'use strict';
const assert=require('node:assert/strict');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const ROOT=path.resolve(__dirname,'../..');
const document={schema:'payment-events/v1',name:'Imported fixture',units:{time:'minutes',currency:'EUR'},accounts:[{id:0,external_id:'alice',name:'Alice'},{id:1,external_id:'bob',name:'Bob'}],events:Array.from({length:150},(_,i)=>({id:'real-'+i,kind:'payment',t:i,u:i%2,v:1-i%2,amount:10+i%7,label:1})),truth:{'real-140':true,'real-141':false}};
async function main(){
  const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
  try{
    const context=await browser.newContext({offline:true,viewport:{width:390,height:844}});
    for(const [file,root,prefix] of [['index.html','fraud-memory-demo','fd'],['xgboost-analytics.html','xgboost-analytics','xa']]){
      const page=await context.newPage(),errors=[];page.on('pageerror',error=>errors.push(error.message));
      await page.goto(pathToFileURL(path.join(ROOT,file)).href);
      const idle=()=>page.evaluate(root=>document.getElementById(root).demo.whenIdle(),root);
      await idle();
      await page.locator('#'+prefix+'-dataset-file').setInputFiles({name:'payments.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(document))});
      await page.waitForFunction(prefix=>document.getElementById(prefix+'-dataset-status').textContent.includes('Imported fixture'),prefix);await idle();
      const state=await page.evaluate(root=>document.getElementById(root).demo.getSnapshot(),root);
      if(prefix==='fd'){assert.equal(state.accounts,2);assert.equal(state.events,150);}
      else {assert.equal(state.count,150);assert.equal(state.finalPosition,150);assert.equal(state.metrics.labeled,2);assert.equal(state.error,null);}
      assert.equal(await page.locator('#'+prefix+'-scenario').isDisabled(),true);
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Imported controls must fit mobile viewport.');
      const isolated=await page.evaluate(doc=>FraudDatasets.load('payment_json',doc).events.every(event=>!('label' in event)),document);assert.equal(isolated,true);
      await page.locator('#'+prefix+'-dataset-file').setInputFiles({name:'invalid.json',mimeType:'application/json',buffer:Buffer.from('{}')});
      await page.waitForFunction(prefix=>document.getElementById(prefix+'-dataset-status').textContent.includes('not loaded'),prefix);
      assert.equal(await page.locator('#'+prefix+'-scenario').isDisabled(),true);
      await page.locator('#'+prefix+'-dataset-reset').click();await idle();assert.equal(await page.locator('#'+prefix+'-scenario').isDisabled(),false);
      assert.deepEqual(errors,[]);await page.close();
    }
    console.log('Both offline browser tools: event import, label isolation, invalid input recovery, reset and mobile layout passed.');
  }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
