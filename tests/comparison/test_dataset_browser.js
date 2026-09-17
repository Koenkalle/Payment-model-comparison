/* Browser picker -> real Python adapters -> all browser comparison models. Build first. */
'use strict';
const assert=require('node:assert/strict'),path=require('node:path');
const {spawn}=require('node:child_process');
const {pathToFileURL}=require('node:url');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const ROOT=path.resolve(__dirname,'../..');
const CSV='TRANSACTION_ID,TX_TIME_SECONDS,CUSTOMER_ID,TERMINAL_ID,TX_AMOUNT,TX_FRAUD,TX_FRAUD_SCENARIO\nlate,120,1,1,12,1,99\nearly,60,1,2,10,0,0\nunknown,120,2,2,8,,\n';

async function start(){
  // Real dataset HTTP routes, without making this data/UI test depend on Torch.
  const code="from types import SimpleNamespace\nfrom framework.comparison_service import make_server\ns=make_server(SimpleNamespace(models=lambda: {'version':1,'models':[]}),port=0)\nprint(s.server_port,flush=True)\ns.serve_forever()";
  const server=spawn(process.env.PYTHON_BINARY||'python',['-c',code],{cwd:ROOT,stdio:['ignore','pipe','pipe']});
  let errors='';server.stderr.on('data',chunk=>errors+=chunk);
  const port=await new Promise((resolve,reject)=>{
    const timeout=setTimeout(()=>{server.kill();reject(Error('Dataset server startup timeout: '+errors));},30000);
    server.stdout.once('data',chunk=>{clearTimeout(timeout);resolve(Number(String(chunk).trim()));});
    server.once('error',error=>{clearTimeout(timeout);reject(error);});
    server.once('exit',code=>{clearTimeout(timeout);reject(Error('Dataset server exited '+code+': '+errors));});
  });
  return {server,url:'http://127.0.0.1:'+port+'/index.html'};
}

async function main(){
  const {server,url}=await start();let browser;
  try{
    browser=await chromium.launch({headless:true,args:['--no-sandbox']});
    const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
    page.on('pageerror',error=>errors.push(error.message));
    const idle=()=>page.evaluate(()=>document.getElementById('fraud-memory-demo').demo.whenIdle());
    const snapshot=()=>page.evaluate(()=>document.getElementById('fraud-memory-demo').demo.getSnapshot());
    await page.goto(url);await idle();
    assert.equal(await page.locator('#fd-benchmark-source').isEnabled(),true);
    await page.locator('#fd-benchmark-load').click();await idle();
    assert.equal((await snapshot()).events,48);
    assert.match(await page.locator('#fd-dataset-status').textContent(),/48 labeled payments/);
    await page.getByText('Threshold and replay settings',{exact:true}).click();
    await page.locator('#fd-warmup').selectOption('8');await idle();
    assert.doesNotMatch(await page.locator('#fd-dataset-status').textContent(),/covers this dataset/);
    await page.locator('#fd-run').click();await idle();
    const full=await snapshot();assert.equal(full.count,48);assert.equal(full.comparison.length,9);
    assert(full.comparison.every(row=>JSON.stringify(row.payments)===JSON.stringify(full.comparison[0].payments)));

    await page.locator('#fd-benchmark-source').selectOption('csv:handbook');
    await page.locator('#fd-benchmark-file').setInputFiles({name:'transactions.csv',mimeType:'text/csv',buffer:Buffer.from(CSV)});
    await page.locator('#fd-benchmark-load').click();await idle();
    assert.match(await page.locator('#fd-dataset-status').textContent(),/positive EUR conversion/);
    assert.equal((await snapshot()).events,48,'A failed import keeps the selected dataset.');
    await page.locator('#fd-benchmark-factor').fill('2');
    await page.getByText('Time interval (optional)',{exact:true}).click();
    await page.locator('#fd-benchmark-start').fill('120');
    await page.locator('#fd-benchmark-load').click();await idle();
    assert.equal((await snapshot()).events,2);
    assert.match(await page.locator('#fd-dataset-status').textContent(),/1 labeled payments; 1 unknown/);
    await page.locator('#fd-benchmark-source').selectOption('csv:ulb');
    assert.equal(await page.locator('#fd-benchmark-load').isDisabled(),true);
    assert.match(await page.locator('#fd-benchmark-help').textContent(),/no account or merchant/);
    assert.equal((await snapshot()).events,2);

    await page.locator('#fd-benchmark-source').selectOption('csv:paysim');
    const paysim='step,type,amount,nameOrig,nameDest,oldbalanceOrg,oldbalanceDest,isFraud\n1,TRANSFER,10,C1,C2,100,0,1\n2,PAYMENT,5,C2,M1,10,0,0\n';
    await page.locator('#fd-benchmark-file').setInputFiles({name:'paysim.csv',mimeType:'text/csv',buffer:Buffer.from(paysim)});
    await page.locator('#fd-benchmark-factor').fill('1');
    await page.locator('#fd-benchmark-load').click();await idle();
    assert.equal((await snapshot()).accounts,3);
    assert.match(await page.locator('#fd-dataset-status').textContent(),/paysim.csv/);
    await page.locator('#fd-benchmark-file').setInputFiles({name:'bad.csv',mimeType:'text/csv',buffer:Buffer.from('wrong,header\n1,2\n')});
    await page.locator('#fd-benchmark-load').click();await idle();
    assert.match(await page.locator('#fd-dataset-status').textContent(),/Missing CSV columns/);
    assert.equal((await snapshot()).accounts,3);

    let release,arrived;
    const gate=new Promise(resolve=>release=resolve),requested=new Promise(resolve=>arrived=resolve);
    await page.route('**/api/datasets/load',async route=>{arrived();await gate;await route.continue().catch(()=>{});});
    await page.locator('#fd-benchmark-source').selectOption('handbook-demo');
    await page.locator('#fd-benchmark-load').click();await requested;
    await page.locator('#fd-dataset-reset').click();release();await idle();
    assert((await snapshot()).events>450,'Reset prevents an outdated dataset response from taking over.');
    await page.unroute('**/api/datasets/load');

    await page.setViewportSize({width:390,height:844});
    await page.locator('#fd-benchmark-source').selectOption('csv:handbook');
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Dataset controls fit a mobile webview.');
    if(process.env.DATASET_SCREENSHOT)await page.screenshot({path:process.env.DATASET_SCREENSHOT,fullPage:false});
    assert.deepEqual(errors,[]);
    await page.goto(pathToFileURL(path.join(ROOT,'index.html')).href);await idle();
    assert.equal(await page.locator('#fd-benchmark-load').isDisabled(),true);
    assert.match(await page.locator('#fd-benchmark-help').textContent(),/python serve.py/);
    assert((await snapshot()).events>450,'Offline synthetic comparison remains usable.');
    console.log('PASS benchmark webview: demo, CSVs, currency, intervals, unknown labels, unsupported ULB, failures, cancellation, mobile and offline fallback.');
  }finally{await browser?.close();server.kill();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
