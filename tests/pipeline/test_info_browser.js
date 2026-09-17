/* Build first. Real service integration for model parameter and feature help. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const {spawn} = require('node:child_process');
const {once} = require('node:events');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const ROOT = path.resolve(__dirname, '../..');

async function start(directory) {
  const code = [
    'import sys', 'from types import SimpleNamespace',
    'from framework.comparison_service import make_server',
    'from framework.pipeline_service import PipelineService',
    'pipeline = PipelineService(sys.argv[1])',
    // An unavailable model must still expose its complete help catalog.
    "pipeline.training._dependencies['torch'] = 'Unavailable for the help integration test'",
    "server = make_server(SimpleNamespace(models=lambda: {'models': []}), port=0, pipeline_service=pipeline)",
    'print(server.server_port, flush=True)', 'server.serve_forever()',
  ].join('\n');
  const child = spawn(process.env.PYTHON_BINARY || 'python', ['-c', code, directory], {
    cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'],
  });
  let logs = '';
  child.stderr.on('data', chunk => { logs += chunk; });
  const port = await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => { child.kill(); reject(Error('Info test server timeout: ' + logs)); }, 30000);
    child.stdout.once('data', chunk => { clearTimeout(timeout); resolve(Number(String(chunk).trim())); });
    child.once('error', error => { clearTimeout(timeout); reject(error); });
    child.once('exit', code => { clearTimeout(timeout); reject(Error('Info test server exited ' + code + ': ' + logs)); });
  });
  return {child, base: 'http://127.0.0.1:' + port, logs: () => logs};
}

async function main() {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'payment-info-browser-'));
  let server, browser;
  try {
    server = await start(directory);
    browser = await chromium.launch({headless: true, args: ['--no-sandbox']});
    const context = await browser.newContext({viewport: {width: 1440, height: 1050}, hasTouch: true});
    const page = await context.newPage(), errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const idle = async (id = 'data-lab') => {
      await page.waitForFunction(id => !!document.getElementById(id)?.demo, id);
      await page.evaluate(id => document.getElementById(id).demo.whenIdle(), id);
    };
    const snapshot = (id = 'data-lab') => page.evaluate(id => document.getElementById(id).demo.getSnapshot(), id);
    const popup = page.locator('.payment-info-window');
    const dismiss = async () => {
      await page.mouse.move(1, 1);
      await page.keyboard.press('Escape');
      await popup.waitFor({state: 'hidden'});
    };
    const inspect = async (host, description, mode = 'hover') => {
      await dismiss();
      if (mode === 'focus') await host.focus();
      else if (mode === 'tap') await host.tap();
      else await host.hover();
      await popup.waitFor({state: 'visible'});
      assert.equal(await popup.getAttribute('role'), 'dialog');
      if (description) {
        await page.waitForFunction(text => document.querySelector('.payment-info-window')?.textContent.includes(text), description);
      }
      const text = await popup.textContent();
      assert(text.length > 100, 'Extended help contains more than a short native title.');
      return text;
    };
    const popupFacts = () => popup.locator('.payment-info-facts').evaluate(node =>
      Object.fromEntries([...node.querySelectorAll('dt')].map(term => [term.textContent, term.nextElementSibling.textContent])));
    const featureRow = id => page.locator('#dl-feature-list .dl-feature[data-feature-id=' + JSON.stringify(id) + ']');
    const input = id => page.locator('#dl-feature-list input[data-feature-id=' + JSON.stringify(id) + ']');
    const description = value => value.info?.description || value.description;
    const csv = 'TRANSACTION_ID,TX_TIME_SECONDS,CUSTOMER_ID,TERMINAL_ID,TX_AMOUNT,TX_FRAUD,TX_FRAUD_SCENARIO\n' +
      Array.from({length: 80}, (_, i) => ['i' + i, i * 60, i % 5, (i * 3) % 7, 10 + i, i % 4 === 0 ? 1 : 0, 0].join(',')).join('\n');
    const imported = await page.request.post(server.base + '/api/pipeline/datasets/import', {
      data: {source: 'handbook', csv, amount_to_eur: 1, name: 'Extended help source'},
    });
    assert.equal(imported.status(), 201, await imported.text());
    const source = (await imported.json()).dataset;
    await page.goto(server.base + '/data-lab.html?dataset=' + source.id);
    await idle();
    let state = await snapshot();
    assert.equal(state.features.error, null);
    const catalog = state.features.catalog;
    assert(catalog.some(feature => feature.kind === 'source'));
    assert(catalog.some(feature => feature.kind === 'derived' && feature.group !== 'graph'));
    assert(catalog.some(feature => feature.group === 'graph'));
    const beforeSelection = state.features.selected;
    for (const feature of catalog) {
      const summary = featureRow(feature.id).locator('summary');
      assert.equal(await summary.getAttribute('data-payment-info') !== null, true, feature.id + ' is a help host.');
      assert.equal(await summary.locator('.payment-info-trigger').count(), 1);
      const text = await inspect(summary, description(feature));
      assert(text.includes(feature.id));
      if (feature.group === 'graph') {
        assert(text.includes(feature.approximation), feature.id + ' explains its approximation.');
      }
    }
    assert.deepEqual((await snapshot()).features.selected, beforeSelection, 'Hovering features never enables them.');
    const plain = catalog.find(feature => feature.id === 'log_amount');
    const graph = catalog.find(feature => feature.id === 'graph_ppr_sender_to_recipient');
    const plainSummary = featureRow(plain.id).locator('summary');
    await inspect(plainSummary.locator('.payment-info-trigger'), description(plain), 'tap');
    assert.equal(await plainSummary.locator('..').getAttribute('open'), null, 'Tapping help does not expand the feature statistics.');
    await dismiss();
    await input(plain.id).check();
    assert((await snapshot()).features.selected.includes(plain.id), 'Feature enable/disable still works independently.');
    await input(plain.id).uncheck();
    assert.deepEqual((await snapshot()).features.selected, beforeSelection);

    const selected = ['amount', plain.id, graph.id];
    const savedResponse = await page.request.post(server.base + '/api/pipeline/datasets/' + source.id + '/features', {
      data: {name: 'Stored help features', features: selected},
    });
    assert.equal(savedResponse.status(), 201, await savedResponse.text());
    const saved = (await savedResponse.json()).dataset;
    await page.goto(server.base + '/data-lab.html?dataset=' + saved.id);
    await idle();
    for (const id of selected) {
      // The first three columns are row ID, time and outcome; saved feature
      // order prevents accidentally matching amount inside log_amount.
      const target = page.locator('#dl-sample-head th').nth(3 + selected.indexOf(id));
      const feature = catalog.find(item => item.id === id);
      assert((await inspect(target, description(feature))).includes(id), 'Stored preview header has feature help.');
    }
    await dismiss();
    await page.locator('#dl-train').click();
    await idle('model-trainer');
    const datasetFeatureDetails = page.locator('#mt-dataset-features').locator('xpath=ancestor::details[1]');
    if (!(await datasetFeatureDetails.evaluate(node => node.open))) await datasetFeatureDetails.locator('summary').click();
    for (const id of selected) {
      const chip = page.locator('#mt-dataset-features [data-feature-id=' + JSON.stringify(id) + ']');
      assert.equal(await chip.count(), 1);
      await inspect(chip, description(catalog.find(item => item.id === id)));
    }

    const modelsResponse = await page.request.get(server.base + '/api/pipeline/models?dataset_id=' + saved.id);
    assert.equal(modelsResponse.status(), 200);
    const models = (await modelsResponse.json()).models;
    assert(models.some(model => !model.available), 'Unavailable model metadata remains inspectable.');
    let parameterCount = 0;
    for (const model of models) {
      await dismiss();
      await page.locator('#mt-model').selectOption(model.id);
      await idle('model-trainer');
      assert.equal(await page.locator('#mt-parameters [data-parameter-type]').count(), model.parameters.length);
      assert.equal(await page.locator('#mt-parameters .payment-info-trigger').count(), model.parameters.length);
      const originalValues = await page.evaluate(() => PaymentPipelineUI.values(document.getElementById('mt-parameters')));
      for (const field of model.parameters) {
        const control = page.locator('#mt-param-' + field.name);
        const text = await inspect(control, field.info.description);
        for (const section of field.info.sections) assert(text.includes(section.text));
        const facts = await popupFacts();
        assert.equal(facts.Parameter, field.name);
        assert.equal(facts.Default, field.default_label || (field.default === null ? 'Automatic / not explicitly set' : String(field.default)));
        if (field.min !== undefined) assert.equal(facts.Minimum, String(field.min));
        if (field.max !== undefined) assert.equal(facts.Maximum, String(field.max));
        if (field.options) assert.equal(facts.Options, field.options.map(value => value === null ? 'None' : String(value)).join(', '));
        parameterCount++;
      }
      assert.deepEqual(await page.evaluate(() => PaymentPipelineUI.values(document.getElementById('mt-parameters'))), originalValues,
        model.id + ' parameters remain unchanged after help inspection.');
      if (!model.available) assert(await page.locator('#mt-train').isDisabled());
    }
    assert(parameterCount >= 31);
    await dismiss();
    await page.locator('#mt-model').selectOption('logistic_regression');
    const logistic = models.find(model => model.id === 'logistic_regression');
    await inspect(page.locator('#mt-param-C'), logistic.parameters.find(field => field.name === 'C').info.description, 'focus');
    await dismiss();
    await page.locator('#mt-train').click();
    await idle('model-trainer');
    state = await snapshot('model-trainer');
    assert.equal(state.error, null);
    assert.equal(state.runs.length, 1);
    await page.locator('#mt-runs details summary').click();
    for (const field of logistic.parameters) {
      const savedParameter = page.locator('#mt-runs .pl-saved-parameters [data-parameter-name=' + JSON.stringify(field.name) + ']');
      assert.equal(await savedParameter.count(), 1);
      await inspect(savedParameter, field.info.description);
      assert.equal((await popupFacts())['Saved value'], field.default === null ? field.default_label : String(field.default));
    }

    // Missing graph identities disable selection but preserve explanations.
    const columns = ['Time', ...Array.from({length: 28}, (_, i) => 'V' + (i + 1)), 'Amount', 'Class'];
    const numericCSV = columns.join(',') + '\n' + Array.from({length: 20}, (_, i) =>
      [i, ...Array.from({length: 28}, (_, j) => i % 2 + j), i + 1, i % 2].join(',')).join('\n');
    const numericResponse = await page.request.post(server.base + '/api/pipeline/datasets/import', {
      data: {source: 'ulb', csv: numericCSV, name: 'Help for unavailable identities'},
    });
    assert.equal(numericResponse.status(), 201, await numericResponse.text());
    await page.goto(server.base + '/data-lab.html?dataset=' + (await numericResponse.json()).dataset.id);
    await idle();
    const numericFeatures = (await snapshot()).features.catalog;
    assert(numericFeatures.some(feature => feature.available === false));
    for (const feature of numericFeatures) {
      const summary = featureRow(feature.id).locator('summary');
      assert.notEqual(await summary.getAttribute('data-payment-info'), null);
      const text = await inspect(summary, description(feature));
      if (feature.available === false) {
        assert(await input(feature.id).isDisabled());
        assert(text.includes(feature.unavailable_reason), 'Unavailable help states the missing data requirement.');
      }
    }
    await dismiss();
    await page.setViewportSize({width: 390, height: 844});
    const unavailable = numericFeatures.find(feature => feature.group === 'graph');
    await inspect(featureRow(unavailable.id).locator('summary .payment-info-trigger'), description(unavailable), 'tap');
    const bounds = await popup.boundingBox();
    assert(bounds.x >= 0 && bounds.x + bounds.width <= 391, 'Touch help fits the mobile viewport.');
    if (process.env.PIPELINE_SCREENSHOTS) {
      await fs.mkdir(process.env.PIPELINE_SCREENSHOTS, {recursive: true});
      await page.screenshot({path: path.join(process.env.PIPELINE_SCREENSHOTS, 'pipeline-info-mobile.png')});
    }
    assert.deepEqual(errors, []);
    console.log('PASS extended help: ' + parameterCount + ' model parameters, ' + catalog.length + ' payment/source/graph features, ' +
      numericFeatures.length + ' numeric/unavailable features, preserved values and toggles, stored preview/trainer/run help, keyboard and mobile touch.');
  } catch (error) {
    if (server) console.error(server.logs().slice(-8000));
    throw error;
  } finally {
    await browser?.close();
    if (server?.child.exitCode === null) {
      const exited = once(server.child, 'exit'); server.child.kill(); await exited;
    }
    await fs.rm(directory, {recursive: true, force: true});
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
