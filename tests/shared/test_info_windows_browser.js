/* Run with PLAYWRIGHT_MODULE=/path/to/playwright node tests/shared/test_info_windows_browser.js. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const ROOT = path.resolve(__dirname, '../..');

async function main() {
  const shell = await fs.readFile(path.join(ROOT, 'shared/ui/standalone-shell.html'), 'utf8');
  const names = ['shared/ui/info-windows.js', 'shared/vendor/floating-ui.core.min.js', 'shared/vendor/floating-ui.dom.min.js'];
  const scripts = (await Promise.all(names.map(name => fs.readFile(path.join(ROOT, name), 'utf8'))))
    .map(source => '<script>' + source + '</script>').join('');
  const markup = `<main style="padding:40px;max-width:700px">
    <button id="outside">Outside</button>
    <p id="legacy" data-tooltip="A brief tooltip">Legacy help</p>
    <label id="parameter" for="count"><span id="parameter-heading">Tree count</span><input id="count" type="number" value="100"></label>
    <details id="feature"><summary id="feature-heading"><label id="feature-label" for="enabled"><input id="enabled" type="checkbox">Graph feature</label></summary><p>Expanded feature details</p></details>
    <p id="plain" title="Old native title" data-tooltip="Old short tooltip">Plain feature</p>
    <svg width="200" height="40"><text id="svg-feature" x="0" y="20"><title id="svg-feature-title" data-original="kept">Detailed chart feature name</title>Chart feature</text></svg>
    <div id="dynamic"></div>
  </main>`;
  const browser = await chromium.launch({headless: true, args: ['--no-sandbox']});
  try {
    const context = await browser.newContext({viewport: {width: 1000, height: 760}});
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.setContent(shell.replace('<!-- FRAUD_DEMO_FRAGMENT -->', markup).replace('<!-- SHARED_UI_SCRIPTS -->', scripts));
    await page.evaluate(() => {
      const $ = id => document.getElementById(id);
      PaymentInfo.attach($('parameter'), PaymentInfo.parameter({name: 'tree_count', label: 'Tree count', type: 'integer',
        default: 100, min: 1, max: 1000, description: 'How many trees are fitted.',
        info: {sections: [{title: 'Choosing a value', text: 'More trees increase work and can improve fit.'}]}}),
      {buttonParent: $('parameter-heading'), hoverDelay: 20});
      PaymentInfo.attach($('feature-label'), PaymentInfo.feature({id: 'graph_ppr', label: 'Contact PPR',
        description: 'Local proximity <script>window.injection=true</script>', transform: 'identity', unit: 'probability',
        method: 'Local push', approximation: 'A lower bound with bounded work.', orientation: 'undirected contacts',
        readout: 'Larger values mean more proximity.', window: 'strictly earlier payments', requirements: ['identities'],
        parameters: {edge_budget: 512}}), {hoverDelay: 20});
      window.detachPlain = PaymentInfo.attach($('plain'), {title: 'Plain feature', description: 'Original extended help.'}, {button: false, hoverDelay: 20});
      window.originalSvgTitle = $('svg-feature').querySelector('title');
      window.originalSvgContent = $('svg-feature').textContent;
      window.detachSvg = PaymentInfo.attach($('svg-feature'), {title: 'Chart feature', description: 'Graphical feature information.'}, {hoverDelay: 20});
    });
    const popup = page.locator('.payment-info-window');
    const waitOpen = () => popup.waitFor({state: 'visible'});
    const waitClosed = () => popup.waitFor({state: 'detached'});
    const dismiss = async () => { await page.keyboard.press('Escape'); await waitClosed(); };
    const fit = async () => {
      const box = await popup.boundingBox(), viewport = page.viewportSize();
      assert(box.x >= 7 && box.y >= 7 && box.x + box.width <= viewport.width - 7 && box.y + box.height <= viewport.height - 7,
        'Popup remains within viewport: ' + JSON.stringify(box));
    };
    assert.equal(await page.locator('#parameter-heading').textContent(), 'Tree count', 'Info icon does not alter feature/parameter text.');
    assert.equal(await page.locator('#plain').getAttribute('title'), null);
    assert.equal(await page.locator('#plain').getAttribute('tabindex'), '0');
    assert.equal(await page.locator('#svg-feature').getAttribute('tabindex'), '0');
    assert.equal(await page.locator('#svg-feature > title').count(), 0, 'SVG marks suppress their native title tooltip.');
    assert.equal(await page.locator('#svg-feature').getAttribute('aria-label'), 'Detailed chart feature name');
    assert.equal(await page.locator('#svg-feature').textContent(), await page.evaluate(() => window.originalSvgContent), 'SVG text remains intact.');
    const metadata = await page.evaluate(() => ({
      feature: PaymentInfo.feature({id: 'tiny', statistics: {count: 10, missing: 0, min: 0, max: 2e-9, mean: 1e-9, std: 3e-10}}),
      parameter: PaymentInfo.parameter({name: 'threshold', default: null, default_label: 'Selected on validation data',
        info: {facts: [['Parameter', 'decision_threshold']]}}),
    }));
    assert.equal(new Map(metadata.feature.facts).get('Mean'), '1.000e-9', 'Tiny statistics remain distinguishable from zero.');
    assert.equal(new Map(metadata.feature.facts).get('Missing values'), '0');
    assert.equal(new Map(metadata.parameter.facts).get('Default'), 'Selected on validation data');
    assert.equal(new Map(metadata.parameter.facts).get('Parameter'), 'decision_threshold', 'Metadata can override standard facts cleanly.');

    await page.locator('#parameter').hover(); await waitOpen();
    assert.match(await popup.textContent(), /How many trees.*Choosing a value.*100.*1000/s);
    await fit();
    await popup.hover(); await page.waitForTimeout(300);
    assert(await popup.isVisible(), 'Pointer can enter the window to read it.');
    await page.locator('#outside').hover(); await waitClosed();
    await page.locator('#plain').hover(); await waitOpen();
    await page.waitForTimeout(800);
    assert.equal(await page.locator('.tooltip').count(), 0, 'Extended help suppresses legacy help on its host.');
    await dismiss();
    await page.locator('#legacy').hover();
    await page.locator('.tooltip').waitFor({state: 'visible'});
    assert.equal(await page.locator('.tooltip').textContent(), 'A brief tooltip', 'Legacy tooltip still works elsewhere.');
    await page.locator('#outside').hover();

    // Clicks on an info button are distinct from checkbox and summary actions.
    await page.locator('#feature-label .payment-info-trigger').click(); await waitOpen();
    assert.equal(await page.locator('#enabled').isChecked(), false);
    assert.equal(await page.locator('#feature').getAttribute('open'), null);
    assert.match(await popup.textContent(), /Local push.*Reading the value.*Approximation.*Graph orientation.*edge_budget/s);
    assert.equal(await page.evaluate(() => window.injection), undefined);
    assert.equal(await popup.locator('script').count(), 0);
    await page.locator('#outside').hover(); await page.waitForTimeout(300);
    assert(await popup.isVisible(), 'Explicit click pins the window.');
    await page.locator('#outside').click(); await waitClosed();
    await page.locator('#enabled').check();
    assert.equal(await page.locator('#enabled').isChecked(), true, 'Normal checkbox remains usable.');
    await dismiss();
    await page.locator('#feature-heading').click({position: {x: 4, y: 4}});
    assert.notEqual(await page.locator('#feature').getAttribute('open'), null, 'Normal summary remains usable.');

    // Focus opens help without requiring a pointer; explicit keyboard activation
    // enters the dialog so long descriptions can be scrolled with the keyboard.
    await page.locator('#outside').focus();
    await page.keyboard.press('Tab');
    await waitOpen();
    assert.equal(await page.locator('#parameter .payment-info-trigger').evaluate(node => node === document.activeElement), true);
    await page.keyboard.press('Enter');
    assert.equal(await popup.evaluate(node => node === document.activeElement), true);
    await page.keyboard.press('Escape'); await waitClosed();
    assert.equal(await page.locator('#parameter .payment-info-trigger').evaluate(node => node === document.activeElement), true);
    await page.waitForTimeout(100);
    assert.equal(await popup.count(), 0, 'Focus restoration does not reopen dismissed help.');
    await page.locator('#svg-feature').focus(); await waitOpen();
    assert.match(await popup.textContent(), /Graphical feature information/); await dismiss();
    await page.evaluate(() => window.detachSvg());
    assert(await page.evaluate(() => document.querySelector('#svg-feature > title') === window.originalSvgTitle), 'Detach restores the original SVG title node.');
    assert.equal(await page.locator('#svg-feature > title').getAttribute('data-original'), 'kept');
    assert.equal(await page.locator('#svg-feature').getAttribute('aria-label'), null);
    assert.equal(await page.locator('#svg-feature > desc').count(), 0);
    const referencedName = await page.evaluate(() => {
      const host = document.getElementById('svg-feature');
      host.setAttribute('aria-labelledby', 'svg-feature-title');
      host.setAttribute('aria-label', 'Existing chart label');
      const detach = PaymentInfo.attach(host, {title: 'Chart feature'});
      const attached = {label: host.getAttribute('aria-label'), reference: document.getElementById('svg-feature-title').textContent,
        titleCount: host.querySelectorAll(':scope > title').length};
      detach();
      return {...attached, restoredLabel: host.getAttribute('aria-label'), restoredReference: host.getAttribute('aria-labelledby')};
    });
    assert.deepEqual(referencedName, {label: 'Existing chart label', reference: 'Detailed chart feature name', titleCount: 0,
      restoredLabel: 'Existing chart label', restoredReference: 'svg-feature-title'});

    // Updating attachments and replacing DOM cannot accumulate icons or windows.
    await page.evaluate(() => {
      PaymentInfo.attach(document.getElementById('plain'), {title: 'Updated feature', description: 'Updated details.'}, {button: false});
      const host = document.createElement('span'); host.id = 'temporary'; host.textContent = 'Temporary';
      document.getElementById('dynamic').appendChild(host);
      for (let i = 0; i < 20; i++) PaymentInfo.attach(host, {title: 'Temporary ' + i, description: 'Current metadata.'}, {hoverDelay: 20});
    });
    assert.equal(await page.locator('#temporary .payment-info-trigger').count(), 1);
    await page.evaluate(() => {
      const host = document.getElementById('temporary');
      host.textContent = 'Renamed visible title';
      const heading = document.createElement('span'); host.appendChild(heading);
      for (let i = 0; i < 3; i++) PaymentInfo.attach(host, {title: 'Renamed metadata', description: 'Current metadata.'}, {buttonParent: heading});
      heading.textContent = 'Updated heading';
      PaymentInfo.attach(host, {title: 'Restored metadata', description: 'Current metadata.'}, {buttonParent: heading});
    });
    assert.equal(await page.locator('#temporary .payment-info-trigger').count(), 1, 'Rerendered headings restore one existing icon.');
    assert.equal(await page.locator('#temporary .payment-info-trigger').getAttribute('aria-label'), 'More about Restored metadata');
    assert.equal(await page.locator('#temporary > span > .payment-info-trigger').count(), 1, 'Existing icon moves into the current heading.');
    await page.locator('#temporary .payment-info-trigger').click(); await waitOpen();
    assert.equal(await popup.count(), 1);
    assert.match(await popup.textContent(), /Restored metadata/);
    await page.evaluate(() => document.getElementById('dynamic').replaceChildren()); await waitClosed();
    await page.evaluate(() => window.detachPlain());
    assert.equal(await page.locator('#plain').getAttribute('title'), 'Old native title');
    assert.equal(await page.locator('#plain').getAttribute('data-tooltip'), 'Old short tooltip');
    assert.equal(await page.locator('#plain').getAttribute('tabindex'), null);

    await page.setViewportSize({width: 320, height: 480});
    await page.evaluate(() => PaymentInfo.attach(document.getElementById('parameter'), {
      title: 'A long parameter description', description: 'Details for this parameter.',
      sections: Array.from({length: 20}, (_, index) => ({title: 'Section ' + index, text: 'Useful detailed information '.repeat(12)})),
    }));
    await page.locator('#parameter .payment-info-trigger').click(); await waitOpen(); await fit();
    assert(await popup.evaluate(node => node.scrollHeight > node.clientHeight), 'Long help scrolls.');
    await popup.evaluate(node => { node.scrollTop = 100; });
    assert(await popup.evaluate(node => node.scrollTop > 0));
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Help does not widen mobile layout.');
    await dismiss();
    assert.deepEqual(errors, []);
    await context.close();

    const touch = await browser.newContext({viewport: {width: 390, height: 740}, hasTouch: true, isMobile: true});
    const phone = await touch.newPage();
    await phone.setContent(shell.replace('<!-- FRAUD_DEMO_FRAGMENT -->', markup).replace('<!-- SHARED_UI_SCRIPTS -->', scripts));
    await phone.evaluate(() => PaymentInfo.attach(document.getElementById('feature-label'), {title: 'Touch feature', description: 'Tap opens this information.'}));
    await phone.locator('#feature-label .payment-info-trigger').tap();
    await phone.locator('.payment-info-window').waitFor({state: 'visible'});
    assert.equal(await phone.locator('#enabled').isChecked(), false);
    assert.equal(await phone.locator('#feature').getAttribute('open'), null);
    await phone.locator('.payment-info-close').tap();
    await phone.locator('.payment-info-window').waitFor({state: 'detached'});
    await touch.close();
    console.log('Shared info windows browser checks passed.');
  } finally { await browser.close(); }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
