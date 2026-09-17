/* Cursor clearance in a dense feature grid, using the real shell and source UI.
 * PLAYWRIGHT_MODULE=/path/to/playwright node tests/shared/test_info_placement_browser.js
 * Screenshots and measured cursor/popup geometry are written to /tmp by default.
 */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const ROOT=path.resolve(__dirname,'../..');
const read=file=>fs.readFileSync(path.join(ROOT,file),'utf8');
const screenshotDirectory=process.env.INFO_PLACEMENT_SCREENSHOTS||'/tmp';
const almost=(actual,expected,label,tolerance=2)=>assert(Math.abs(actual-expected)<=tolerance,label+': expected '+expected+', received '+actual);
function markup(){
  const names=['Sender PageRank','Recipient PageRank','Contact PPR','PPR error bound','Prior contacts','Component size','Outgoing value','Incoming value','Recent activity','Activity gap','Amount category','Prior payment count'];
  return `<style>
    body{margin:0;font:14px/1.45 system-ui;color:#27372e;background:#f7faf8}
    .placement-fixture{padding:32px;max-width:1150px}.placement-fixture h1{font-size:24px;margin:0 0 8px}.placement-fixture p{margin:0 0 22px;color:#64766b}
    .feature-grid{display:grid;grid-template-columns:repeat(4,168px);gap:6px;width:max-content;max-width:100%}
    .feature-cell{display:flex;align-items:center;justify-content:space-between;gap:4px;box-sizing:border-box;width:168px;height:48px;padding:9px;border:1px solid #d6e3db;border-radius:5px;background:white;font-size:11px}
    .edge-host{position:fixed;display:flex;align-items:center;justify-content:space-between;box-sizing:border-box;width:164px;min-height:40px;padding:9px;border:1px solid #ccdcd2;border-radius:5px;background:white;font-size:12px}
    #right-host{right:16px;top:170px}#bottom-host{left:36px;bottom:12px}#left-host{left:3px;top:440px}#top-host{left:610px;top:5px}#keyboard-host{left:38px;top:360px}
    #outside{margin-bottom:20px}.fixture-note{margin-top:30px!important;width:650px;max-width:100%}
    @media(max-width:600px){.placement-fixture{padding:22px}.feature-grid{grid-template-columns:repeat(2,minmax(0,1fr));width:100%}.feature-cell{width:100%;font-size:10px}#top-host,#left-host,#right-host,#xgboost-analytics{display:none}#keyboard-host{left:22px;top:420px}#bottom-host{left:22px}.fixture-note{font-size:12px}}
  </style><main class="placement-fixture"><button id="outside">Outside help</button><h1>Dataset feature inputs</h1><p>Inspect a feature without covering the next nearby value.</p><div class="feature-grid">${names.map((name,index)=>'<div id="feature-'+index+'" class="feature-cell"><span>'+name+'</span></div>').join('')}</div><p class="fixture-note">This dense fixture provides adjacent rows and columns, screen-edge targets, and a long definition that can be scrolled inside the information window.</p></main>
  <div id="right-host" class="edge-host">Right-edge feature</div><div id="bottom-host" class="edge-host">Bottom-edge feature</div><div id="left-host" class="edge-host">Left-edge feature</div><div id="top-host" class="edge-host">Top-edge feature</div><div id="keyboard-host" class="edge-host">Long feature definition</div><section id="xgboost-analytics" style="position:fixed;left:850px;top:360px;width:250px;padding:0"><label id="xgb-host">Actual XGBoost control styling</label></section><div style="height:1000px" aria-hidden="true"></div>`;
}
function html(){
  const scripts=['shared/vendor/floating-ui.core.min.js','shared/vendor/floating-ui.dom.min.js','shared/ui/info-windows.js'].map(file=>'<script>'+read(file)+'</script>').join('');
  const analyticsStyles=read('tools/xgboost-analytics/template.html').match(/<style>[\s\S]*?<\/style>/)[0];
  return read('shared/ui/standalone-shell.html').replace('<!-- FRAUD_DEMO_FRAGMENT -->',()=>analyticsStyles+markup()).replace('<!-- SHARED_UI_SCRIPTS -->',()=>scripts);
}
async function configure(page){
  await page.setContent(html());
  await page.evaluate(()=>{
    for(const [index,host]of [...document.querySelectorAll('.feature-cell,.edge-host,#xgb-host')].entries()){
      const title=host.textContent;
      PaymentInfo.attach(host,{title,description:'Specific definition for '+title.toLowerCase()+'. This value summarizes earlier observable payment relationships.',
        sections:host.id==='keyboard-host'?Array.from({length:18},(_,section)=>({title:'Interpretation '+(section+1),text:('Read the recorded method and history window before interpreting this feature. ').repeat(5)})):[{title:'How to read it',text:'The feature uses preceding history. Higher values describe a different observed pattern; they do not alone determine the model decision.'}],
        facts:[['Feature identifier','feature_'+index],['History','Strictly earlier timestamps'],['Transformation','Defined by this feature’s saved recipe']]});
    }
  });
}
async function main(){
  const browser=await chromium.launch({headless:true,args:['--no-sandbox'],...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
  const context=await browser.newContext({viewport:{width:1280,height:920},offline:true}),page=await context.newPage(),errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  const popup=page.locator('.payment-info-window');
  const geometry=async()=>popup.boundingBox();
  const waitOpen=async title=>{await popup.waitFor({state:'visible'});await page.waitForFunction(expected=>document.querySelector('.payment-info-title')?.textContent===expected,title);};
  const close=async()=>{await page.keyboard.press('Escape');await page.mouse.move(900,70);await popup.waitFor({state:'detached'});};
  const fit=async(targetPage=page)=>{
    const box=await targetPage.locator('.payment-info-window').boundingBox(),viewport=targetPage.viewportSize();
    assert(box&&box.x>=7&&box.y>=7&&box.x+box.width<=viewport.width-7&&box.y+box.height<=viewport.height-7,'Popup stays within the viewport: '+JSON.stringify(box));return box;
  };
  const point=async(selector,position={x:12,y:14})=>{const box=await page.locator(selector).boundingBox();return {x:box.x+position.x,y:box.y+position.y};};
  try{
    await configure(page);
    const icon=await page.locator('#feature-0 .payment-info-trigger').boundingBox();
    almost(icon.width,16,'Compact desktop trigger width',.5);almost(icon.height,16,'Compact desktop trigger height',.5);
    const ring=await page.locator('#feature-0 .payment-info-trigger').evaluate(button=>({width:parseFloat(getComputedStyle(button,'::before').width),height:parseFloat(getComputedStyle(button,'::before').height)}));
    almost(ring.width,14,'Visible information ring width',.5);almost(ring.height,14,'Visible information ring height',.5);
    const analyticsIcon=await page.locator('#xgb-host .payment-info-trigger').boundingBox();
    almost(analyticsIcon.width,16,'XGBoost-specific button styles preserve compact trigger width',.5);almost(analyticsIcon.height,16,'XGBoost-specific button styles preserve compact trigger height',.5);

    // Move inside the same host during the delay: placement must capture the
    // latest pointer position, then stop following that pointer once open.
    const host=await page.locator('#feature-0').boundingBox(),first={x:host.x+12,y:host.y+16},latest={x:host.x+host.width-7,y:host.y+22};
    await page.mouse.move(first.x,first.y);await page.waitForTimeout(120);
    await page.mouse.move(latest.x,latest.y);await page.waitForTimeout(120);
    assert.equal(await popup.count(),0,'The 350ms hover delay is retained while the pointer moves within a host.');
    await waitOpen('Sender PageRank');const original=await fit();
    almost(original.x-latest.x,32,'Desktop popup clearance to the latest pointer');
    almost(original.y-latest.y,12,'Desktop popup vertical offset');
    const horizontal=await point('#feature-1',{x:5,y:22}),vertical=await point('#feature-4',{x:host.width-7,y:15});
    assert(horizontal.x<original.x,'The next horizontal feature has an unobscured entry point.');
    assert(vertical.x<original.x,'The next vertical feature is outside the popup.');

    fs.mkdirSync(screenshotDirectory,{recursive:true});
    const measured={cursor:latest,popup:original,rightGap:original.x-latest.x,downGap:original.y-latest.y,desktopTrigger:icon,analyticsTrigger:analyticsIcon};
    await page.evaluate(({cursor,rightGap,downGap})=>{
      const cursorMark=document.createElement('div');cursorMark.id='cursor-evidence';cursorMark.style.cssText='position:fixed;z-index:2147483500;pointer-events:none;width:12px;height:12px;border:2px solid #ce2c3b;border-radius:50%;box-sizing:border-box;left:'+(cursor.x-6)+'px;top:'+(cursor.y-6)+'px;';document.body.append(cursorMark);
      const caption=document.createElement('div');caption.id='geometry-evidence';caption.textContent='Mouse ('+cursor.x+', '+cursor.y+') · popup starts '+rightGap+'px right and '+downGap+'px below';caption.style.cssText='position:fixed;left:250px;bottom:22px;padding:10px 14px;background:#fff;border:1px solid #cadbd0;border-radius:5px;font:12px system-ui;pointer-events:none';document.body.append(caption);
    },measured);
    await page.screenshot({path:path.join(screenshotDirectory,'payment-info-placement-desktop.png')});
    fs.writeFileSync(path.join(screenshotDirectory,'payment-info-placement-geometry.json'),JSON.stringify(measured,null,2)+'\n');
    await page.evaluate(()=>{document.getElementById('cursor-evidence').remove();document.getElementById('geometry-evidence').remove();});

    await page.mouse.move(first.x,first.y);await page.waitForTimeout(90);const stationary=await geometry();
    almost(stationary.x,original.x,'Open popup freezes its horizontal position');almost(stationary.y,original.y,'Open popup freezes its vertical position');
    await page.mouse.move(latest.x,latest.y);
    await page.mouse.move(original.x+18,original.y+35,{steps:6});await page.waitForTimeout(300);
    assert(await popup.isVisible(),'The pointer can enter the detached popup to read it.');
    assert.equal(await popup.locator('.payment-info-title').textContent(),'Sender PageRank','Crossing a neighboring host on the way into the popup does not replace the explanation being read.');
    const entered=await geometry();almost(entered.x,original.x,'Entering the popup does not chase the pointer');almost(entered.y,original.y,'Entering preserves vertical placement');

    await page.mouse.move(horizontal.x,horizontal.y);await page.waitForTimeout(100);await popup.focus();await page.waitForTimeout(400);
    assert.equal(await popup.locator('.payment-info-title').textContent(),'Sender PageRank','Focusing the readable popup cancels a neighboring feature’s pending hover.');
    await page.mouse.move(original.x+18,original.y+35);

    // The pointer can switch directly to neighboring features through the clear
    // strip. No click, forced event, or artificial DOM dispatch is used.
    await page.mouse.move(horizontal.x,horizontal.y);await waitOpen('Recipient PageRank');const nextHorizontal=await fit();
    almost(nextHorizontal.x-horizontal.x,32,'Neighbor feature gets its own cursor clearance');
    await page.mouse.move(vertical.x,vertical.y);await waitOpen('Prior contacts');const nextVertical=await fit();
    almost(nextVertical.x-vertical.x,32,'Vertical neighbor gets its own cursor clearance');
    await close();

    const right=await page.locator('#right-host').boundingBox(),rightPointer={x:right.x+right.width-10,y:right.y+18};
    await page.mouse.move(rightPointer.x,rightPointer.y);await waitOpen('Right-edge feature');const leftPlacement=await fit();
    almost(rightPointer.x-(leftPlacement.x+leftPlacement.width),32,'Insufficient right space prefers a clear left placement');await close();
    for(const id of ['left-host','top-host','bottom-host']){
      const cursor=await point('#'+id);await page.mouse.move(cursor.x,cursor.y);await waitOpen({'left-host':'Left-edge feature','top-host':'Top-edge feature','bottom-host':'Bottom-edge feature'}[id]);await fit();await close();
    }

    // A keyboard activation on the currently hovered host must change from
    // cursor anchoring to element anchoring even though its content is open.
    await page.mouse.move(first.x,first.y);await waitOpen('Sender PageRank');
    const firstTrigger=page.locator('#feature-0 .payment-info-trigger'),firstTriggerBox=await firstTrigger.boundingBox();
    assert((await geometry()).x<firstTriggerBox.x+firstTriggerBox.width,'The initial hover origin differs visibly from the keyboard origin.');
    await firstTrigger.focus();
    await page.waitForFunction(right=>document.querySelector('.payment-info-window').getBoundingClientRect().left>=right,firstTriggerBox.x+firstTriggerBox.width);
    await close();await page.locator('#outside').focus();

    const keyboard=page.locator('#keyboard-host .payment-info-trigger'),keyboardBox=await keyboard.boundingBox();
    await keyboard.focus();await waitOpen('Long feature definition');const keyboardPopup=await fit();
    assert(keyboardPopup.x>=keyboardBox.x+keyboardBox.width,'Keyboard help opens to the right of its activating element when space permits.');
    await page.keyboard.press('Enter');assert(await popup.evaluate(node=>node===document.activeElement),'Keyboard activation focuses the readable popup.');
    await page.keyboard.press('PageDown');await page.waitForFunction(()=>document.querySelector('.payment-info-window')?.scrollTop>0);
    const scrolled=await geometry();almost(scrolled.x,keyboardPopup.x,'Reading scroll does not move the popup horizontally');
    await page.keyboard.press('Escape');await popup.waitFor({state:'detached'});assert(await keyboard.evaluate(node=>node===document.activeElement),'Closing restores keyboard focus.');

    await page.locator('#outside').focus();await page.mouse.move(first.x,first.y);await waitOpen('Sender PageRank');
    await page.evaluate(()=>window.scrollTo(0,600));await popup.waitFor({state:'detached'});
    await page.evaluate(()=>window.scrollTo(0,0));await page.mouse.move(900,70);
    await page.mouse.move(first.x,first.y);await page.waitForTimeout(100);await page.evaluate(()=>window.scrollTo(0,600));await page.waitForTimeout(400);
    assert.equal(await popup.count(),0,'Scrolling a pending hover target away prevents stale help from opening.');
    await page.evaluate(()=>window.scrollTo(0,0));await page.mouse.move(900,70);

    // Exercise the built-in viewport fallback without the optional positioning
    // vendor as well; narrow displays need a usable, scrollable full definition.
    await page.setViewportSize({width:390,height:844});await page.locator('#outside').focus();await page.mouse.move(350,70);await keyboard.focus();await waitOpen('Long feature definition');await fit();await close();
    await page.evaluate(()=>{window.FloatingUIDOM=undefined;});
    const narrowPoint=await point('#feature-0',{x:16,y:18});await page.mouse.move(narrowPoint.x,narrowPoint.y);await waitOpen('Sender PageRank');await fit();await close();

    // A desktop cursor can leave the viewport after resizing. Pinned help must
    // reanchor to its responsive field; ordinary hover should disappear.
    const resized=await context.newPage();resized.on('pageerror',error=>errors.push(error.message));
    await resized.setViewportSize({width:1200,height:800});await configure(resized);
    await resized.evaluate(()=>{
      const host=document.createElement('div');host.id='resize-host';host.textContent='Responsive feature';
      host.style.cssText='position:fixed;left:40px;top:40px;width:calc(100vw - 80px);height:150px;z-index:1;background:white;border:1px solid #d6e3db';
      document.body.append(host);PaymentInfo.attach(host,{title:'Responsive feature',description:'A definition whose original mouse position may leave the resized viewport.'});
    });
    const resizedPopup=resized.locator('.payment-info-window');
    await resized.mouse.move(850,90);await resizedPopup.waitFor({state:'visible'});const beforeResize=await fit(resized);
    await resized.locator('#resize-host .payment-info-trigger').click();
    almost((await resizedPopup.boundingBox()).x,beforeResize.x,'Pinning preserves the frozen mouse origin before resize');
    await resized.setViewportSize({width:450,height:800});
    await resized.waitForFunction(()=>{const rect=document.querySelector('.payment-info-window')?.getBoundingClientRect();return rect&&rect.left>=7&&rect.right<=innerWidth-7;});
    await fit(resized);await resized.locator('.payment-info-close').click();
    await resized.setViewportSize({width:1200,height:800});await resized.mouse.move(0,0);await resized.mouse.move(850,90);await resizedPopup.waitFor({state:'visible'});
    await resized.setViewportSize({width:450,height:800});await resizedPopup.waitFor({state:'detached'});
    await resized.close();

    const touchContext=await browser.newContext({viewport:{width:1100,height:900},hasTouch:true,offline:true}),touch=await touchContext.newPage();
    touch.on('pageerror',error=>errors.push(error.message));await configure(touch);
    const touchButton=touch.locator('#feature-0 .payment-info-trigger'),touchButtonBox=await touchButton.boundingBox();
    const touchHost=await touch.locator('#feature-0').boundingBox();
    await touch.mouse.move(touchHost.x+12,touchHost.y+16);await touch.locator('.payment-info-window').waitFor({state:'visible'});
    await touchButton.tap();await touch.locator('.payment-info-window').waitFor({state:'visible'});const touchPopup=await fit(touch);
    assert(touchPopup.x>=touchButtonBox.x+touchButtonBox.width,'Touch help anchors to the activating element’s right when space permits.');
    await touch.locator('.payment-info-close').tap();assert.equal(await touch.locator('.payment-info-window').count(),0);
    await touch.setViewportSize({width:390,height:844});await touchButton.tap();await touch.locator('.payment-info-window').waitFor({state:'visible'});await fit(touch);
    await touch.locator('.payment-info-close').tap();await touchContext.close();
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({test:'information window pointer placement',passed:true,geometry:measured,screenshot:path.join(screenshotDirectory,'payment-info-placement-desktop.png'),checks:['compact icons','latest delayed pointer','frozen open position','horizontal and vertical neighbors','pending hover cancellation while reading','right/left viewport edges','keyboard scrolling','mobile fallback','pinned and hover resize','touch placement']}));
  }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
