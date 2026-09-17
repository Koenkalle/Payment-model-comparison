/* Build first. Real service: shared feature selection, persistence and training. */
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
    "server = make_server(SimpleNamespace(models=lambda: {'models': []}), port=0, pipeline_service=pipeline)",
    'print(server.server_port, flush=True)', 'server.serve_forever()',
  ].join('\n');
  const child = spawn(process.env.PYTHON_BINARY || 'python', ['-c', code, directory], {
    cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'],
  });
  let logs = '';
  child.stderr.on('data', chunk => { logs += chunk; });
  const port = await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => { child.kill(); reject(Error('Feature test server timeout: ' + logs)); }, 30000);
    child.stdout.once('data', chunk => { clearTimeout(timeout); resolve(Number(String(chunk).trim())); });
    child.once('error', error => { clearTimeout(timeout); reject(error); });
    child.once('exit', code => { clearTimeout(timeout); reject(Error('Feature test server exited ' + code + ': ' + logs)); });
  });
  return {child, base: 'http://127.0.0.1:' + port, logs: () => logs};
}

async function main() {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'payment-feature-browser-'));
  let server, browser;
  try {
    server = await start(directory);
    browser = await chromium.launch({headless: true, args: ['--no-sandbox']});
    const page = await browser.newPage({viewport: {width: 1440, height: 1050}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const idle = async (id = 'data-lab') => {
      await page.waitForFunction(id => !!document.getElementById(id)?.demo, id);
      await page.evaluate(id => document.getElementById(id).demo.whenIdle(), id);
    };
    const snapshot = (id = 'data-lab') => page.evaluate(id => document.getElementById(id).demo.getSnapshot(), id);
    const action = async selector => { await page.locator(selector).click(); await idle(); };
    const generated = await page.request.post(server.base + '/api/pipeline/datasets/generate', {
      data: {generator: 'handbook_generator', name: 'Feature browser source', parameters: {
        transactions: 200, customer_count: 12, terminal_count: 6, fraud_rate: 0.3, seed: 41,
      }},
    });
    assert.equal(generated.status(), 201, await generated.text());
    const source = (await generated.json()).dataset;
    await page.goto(server.base + '/data-lab.html?dataset=' + source.id);
    await idle();
    let state = await snapshot();
    assert.equal(state.error, null);
    assert.equal(state.features.error, null);
    const originals = state.features.catalog.filter(feature => feature.kind === 'source');
    const derived = state.features.catalog.filter(feature => feature.kind === 'derived' && feature.available !== false);
    assert(originals.length > 0 && derived.length >= 27, 'Source columns and mined payment features are inspectable together.');
    assert.equal(state.features.dirty, false);
    const featureInput = id => page.locator('#dl-feature-list input[data-feature-id=' + JSON.stringify(id) + ']');
    const sourceFeature = originals[0];
    await featureInput(sourceFeature.id).uncheck();
    assert.equal((await snapshot()).features.dirty, true);
    assert.equal(await page.locator('#dl-train').getAttribute('aria-disabled'), 'true');
    assert.match(await page.locator('#dl-feature-state').textContent(), /Unsaved|No columns/);
    if (!(await snapshot()).features.selected.length) assert(await page.locator('#dl-feature-save').isDisabled());
    assert((await page.locator('#dl-sample-head').textContent()).includes(sourceFeature.id), 'Preview still shows stored values while the selection is unsaved.');
    await action('#dl-feature-reset');
    assert.equal((await snapshot()).features.dirty, false);
    assert.equal(await page.locator('#dl-train').getAttribute('aria-disabled'), 'false');

    await action('#dl-feature-add');
    assert((await snapshot()).features.selected.includes(derived[0].id));
    await page.locator('#dl-feature-filter').selectOption('derived');
    await page.locator('#dl-feature-search').fill(derived[0].id);
    const row = page.locator('#dl-feature-list .dl-feature[data-feature-id=' + JSON.stringify(derived[0].id) + ']');
    await row.locator('summary').click();
    assert.match(await row.textContent(), /Minimum|Maximum/);
    assert.match(await row.textContent(), /Mean|Std\. deviation/);
    assert(await row.locator('[role=img]').isVisible(), 'Feature distribution is inspectable.');
    await page.locator('#dl-feature-search').fill('feature-does-not-exist');
    assert(await page.locator('#dl-feature-empty').isVisible());
    await page.locator('#dl-feature-search').fill('');
    await page.locator('#dl-feature-filter').selectOption('all');
    await action('#dl-feature-source');
    assert.deepEqual((await snapshot()).features.selected, originals.map(feature => feature.id));

    await featureInput(sourceFeature.id).uncheck();
    for (const feature of derived.slice(0, 3)) await featureInput(feature.id).check();
    const requested = (await snapshot()).features.selected;
    await page.locator('#dl-feature-name').fill('Feature browser shared inputs');
    const featureURL = server.base + '/api/pipeline/datasets/' + source.id + '/features';
    await page.route(featureURL, route => route.request().method() === 'POST'
      ? route.fulfill({status: 503, contentType: 'application/json', body: JSON.stringify({error: 'Temporary feature save outage'})})
      : route.continue(), {times: 1});
    await action('#dl-feature-save');
    assert.match((await snapshot()).features.error, /Temporary feature save outage/);
    assert.deepEqual((await snapshot()).features.selected, requested, 'A failed save preserves the editable selection.');
    assert(await page.locator('#dl-feature-save').isEnabled());
    await action('#dl-feature-save');
    state = await snapshot();
    assert.equal(state.features.error, null);
    assert.notEqual(state.selectedDatasetId, source.id, 'Feature edits create a separate immutable dataset.');
    const savedId = state.selectedDatasetId;
    assert.equal(state.detail.name, 'Feature browser shared inputs');
    assert.equal(state.features.dirty, false);
    assert.deepEqual(state.features.selected, requested);
    const columns = (await page.locator('#dl-sample-head th').allTextContents()).slice(3);
    assert.deepEqual(columns, requested, 'Preview columns exactly match the saved feature selection.');
    assert.equal((await (await page.request.get(server.base + '/api/pipeline/datasets/' + source.id)).json()).dataset.fingerprint, source.fingerprint);
    await page.reload();
    await idle();
    assert.deepEqual((await snapshot()).features.selected, requested, 'Feature selection survives reload.');

    await page.locator('#dl-train').click();
    await idle('model-trainer');
    assert.equal((await snapshot('model-trainer')).datasetId, savedId);
    assert((await page.locator('#mt-dataset-features').textContent()).includes(derived[0].id), 'The trainer exposes the saved shared input columns.');
    await page.locator('#mt-model').selectOption('logistic_regression');
    await page.locator('#mt-train').click();
    await idle('model-trainer');
    state = await snapshot('model-trainer');
    assert.equal(state.error, null);
    assert.equal(state.runs.length, 1, JSON.stringify(state));
    assert.deepEqual(state.runs[0].feature_names, requested, 'Training consumes exactly the saved selected features.');

    await page.goto(server.base + '/data-lab.html?dataset=' + savedId);
    await idle();
    assert((await page.locator('#dl-facts').textContent()).includes('Original source'));
    // A save refresh may finish after a user has selected another dataset.
    let releaseSaveRefresh;
    const saveRefreshReleased = new Promise(resolve => { releaseSaveRefresh = resolve; });
    await page.route(server.base + '/api/pipeline/datasets', async route => { await saveRefreshReleased; await route.continue(); }, {times: 1});
    await featureInput(derived[3].id).check();
    await page.locator('#dl-feature-name').fill('Feature browser save race');
    const refreshingAfterSave = page.waitForRequest(request => request.url() === server.base + '/api/pipeline/datasets');
    await page.locator('#dl-feature-save').click();
    await refreshingAfterSave;
    await page.locator('#dl-library button').filter({hasText: source.name}).click();
    await page.waitForFunction(id => document.getElementById('data-lab').demo.getSnapshot().features.datasetId === id, source.id);
    await featureInput(derived[4].id).check();
    releaseSaveRefresh();
    await idle();
    state = await snapshot();
    assert.equal(state.selectedDatasetId, source.id, 'A late save refresh preserves the newer dataset selection.');
    assert.equal(state.features.dirty, true, 'A late save refresh preserves edits made on the newer dataset.');
    assert(state.datasets.some(dataset => dataset.name === 'Feature browser save race'), 'The background save still appears in the library.');
    await page.locator('#dl-library button').filter({hasText: 'Feature browser shared inputs'}).click();
    await idle();
    const savedURL = server.base + '/api/pipeline/datasets/' + savedId + '/features';
    await page.route(savedURL, route => route.fulfill({status: 503, contentType: 'application/json', body: JSON.stringify({error: 'Temporary feature inspection outage'})}), {times: 1});
    await action('#dl-refresh');
    assert.match((await snapshot()).features.error, /Temporary feature inspection outage/);
    assert(await page.locator('#dl-preview').isVisible(), 'A feature outage leaves the saved record preview usable.');
    await action('#dl-feature-retry');
    assert.equal((await snapshot()).features.error, null);
    const pinnedFeatures = await (await page.request.get(savedURL)).json();
    await page.route(savedURL, route => route.fulfill({status: 200, contentType: 'application/json', body: JSON.stringify({...pinnedFeatures, latest_recipe_version: 'payment-features/future'})}), {times: 1});
    await action('#dl-refresh');
    assert(await page.locator('#dl-feature-update').isVisible(), 'Pinned variants surface newly available recipe versions.');
    assert.equal(new URL(await page.locator('#dl-feature-update a').getAttribute('href'), server.base).searchParams.get('dataset'), source.id);
    assert.deepEqual((await snapshot()).features.selected, requested, 'New recipe guidance preserves the saved selection.');
    await action('#dl-refresh');
    assert(await page.locator('#dl-feature-update').isHidden());

    // Late feature responses cannot replace the current dataset or its selection.
    let release;
    const released = new Promise(resolve => { release = resolve; });
    await page.route(featureURL, async route => { await released; await route.continue(); }, {times: 1});
    const loading = page.waitForRequest(featureURL);
    await page.locator('#dl-library button').filter({hasText: source.name}).click();
    await loading;
    await page.locator('#dl-library button').filter({hasText: 'Feature browser shared inputs'}).click();
    await page.waitForFunction(id => document.getElementById('data-lab').demo.getSnapshot().features.datasetId === id, savedId);
    release();
    await idle();
    assert.equal((await snapshot()).features.datasetId, savedId);
    assert.deepEqual((await snapshot()).features.selected, requested);

    await page.setViewportSize({width: 390, height: 844});
    await page.locator('#dl-features').scrollIntoViewIfNeeded();
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'Feature view fits mobile.');
    await page.locator('#dl-feature-list summary').first().click();
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'Expanded feature statistics fit mobile.');
    if (process.env.PIPELINE_SCREENSHOTS) {
      await fs.mkdir(process.env.PIPELINE_SCREENSHOTS, {recursive: true});
      await page.screenshot({path: path.join(process.env.PIPELINE_SCREENSHOTS, 'pipeline-features-mobile.png')});
    }

    // Missing relational identity is explained per feature for fixed numeric data.
    const header = ['Time', ...Array.from({length: 28}, (_, i) => 'V' + (i + 1)), 'Amount', 'Class'];
    const csv = header.join(',') + '\n' + Array.from({length: 20}, (_, i) =>
      [i, ...Array.from({length: 28}, (_, j) => i % 2 + j), i + 1, i % 2].join(',')).join('\n');
    const imported = await page.request.post(server.base + '/api/pipeline/datasets/import', {data: {source: 'ulb', csv, name: 'Feature browser numeric'}});
    assert.equal(imported.status(), 201, await imported.text());
    const numeric = (await imported.json()).dataset;
    await page.goto(server.base + '/data-lab.html?dataset=' + numeric.id);
    await idle();
    state = await snapshot();
    const unavailable = state.features.catalog.find(feature => feature.available === false);
    assert(unavailable, 'Numeric-only data exposes unavailable history features with a reason.');
    assert(await featureInput(unavailable.id).isDisabled());
    assert(unavailable.unavailable_reason);
    assert((await page.locator('#dl-feature-list').textContent()).includes(unavailable.unavailable_reason));
    assert.deepEqual(errors, []);
    console.log('PASS feature browser: per-column enable/disable, filters, definitions/statistics, save/discard, immutable persisted previews, model input handoff and fit, retry, dataset races, mobile and unavailable features.');
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
