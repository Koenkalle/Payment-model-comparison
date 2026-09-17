/* Source-level regression: metadata formats must reach the popup introduction. */
'use strict';
const assert = require('node:assert/strict');
const path = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const ROOT = path.resolve(__dirname, '../..');

async function main() {
  const browser = await chromium.launch({headless: true, args: ['--no-sandbox']});
  try {
    const page = await browser.newPage({viewport: {width: 1100, height: 800}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.setContent('<main style="padding:80px"><div id="features"></div></main>');
    for (const file of ['datasets/payment_features.js', 'shared/ui/info-windows.js', 'tools/pipeline/client.js']) {
      await page.addScriptTag({path: path.join(ROOT, file)});
    }
    const inspect = async (names, definitions, expected) => {
      await page.mouse.move(1, 1);
      await page.keyboard.press('Escape');
      await page.evaluate(({names, definitions}) => PaymentPipelineUI.featureList(document.getElementById('features'), names, definitions), {names, definitions});
      await page.locator('#features [data-feature-id]').first().hover();
      await page.locator('.payment-info-window').waitFor({state: 'visible'});
      const intro = await page.locator('.payment-info-description').textContent();
      if (expected) assert.equal(intro, expected);
      assert(!/^(Numeric input supplied|A stored numeric model input|An original numeric column)/.test(intro), intro);
      return intro;
    };
    const named = 'Payments this recipient received during the provider’s recorded six-hour window.';
    await inspect(['recipient_in_count'], [{name: 'recipient_in_count', label: 'Provider recipient count', description: named,
      transform: 'sqrt(value)', window: 'six hours'}], named);
    assert((await page.locator('.payment-info-window').textContent()).includes('six hours'), 'Provider history is preserved.');
    await inspect(['recipient_in_count'], [{id: 'recipient_in_count', description: named,
      info: {description: 'Numeric input supplied to the model from this dataset.'}}], named);
    const mapped = 'Vendor signal q: the source’s recorded balance adjustment in account currency.';
    await inspect(['vendor_signal_q'], {vendor_signal_q: {label: 'Balance adjustment', description: mapped}}, mapped);
    const custom = 'Provider-specific interpretation of the sender count for this saved snapshot.';
    await inspect(['sender_out_count'], [{id: 'sender_out_count', description: 'Saved sender count.', info: {description: custom}}], custom);
    const known = await inspect(['sender_out_count'], []);
    assert.match(known, /sender/i);
    assert.match(known, /sent|outgoing/i, 'A known feature recovers its actual direction instead of generic model-input text.');
    assert(!/settled|strictly earlier|60 minutes|log1p/.test(known), 'Missing saved definitions do not borrow a checkpoint’s history or calculation.');
    assert.match(known, /saved definition/i, 'Known-name recovery states that the saved definition is needed.');
    const unknown = await inspect(['unrecognized_vendor_signal'], []);
    assert(unknown.includes('unrecognized_vendor_signal') || unknown.toLowerCase().includes('unrecognized vendor signal'), 'An unknown column is identified explicitly.');
    assert.match(await page.locator('.payment-info-window').textContent(), /saved definition|source schema/i);
    assert.deepEqual(errors, []);
    console.log('PASS feature introductions: ID/name arrays, keyed definitions, provider semantics, known-feature recovery and explicit unknown-column help.');
  } finally {
    await browser.close();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
