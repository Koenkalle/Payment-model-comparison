/* Build first. Real service: graph recipes, diagnostics and shared saved inputs. */
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
    const timeout = setTimeout(() => { child.kill(); reject(Error('Graph feature server timeout: ' + logs)); }, 30000);
    child.stdout.once('data', chunk => { clearTimeout(timeout); resolve(Number(String(chunk).trim())); });
    child.once('error', error => { clearTimeout(timeout); reject(error); });
    child.once('exit', code => { clearTimeout(timeout); reject(Error('Graph feature server exited ' + code + ': ' + logs)); });
  });
  return {child, base: 'http://127.0.0.1:' + port, logs: () => logs};
}

async function main() {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'payment-graph-feature-browser-'));
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
    const featureInput = id => page.locator('#dl-feature-list input[data-feature-id=' + JSON.stringify(id) + ']');
    const featureRow = id => page.locator('#dl-feature-list .dl-feature[data-feature-id=' + JSON.stringify(id) + ']');
    const csv = 'TRANSACTION_ID,TX_TIME_SECONDS,CUSTOMER_ID,TERMINAL_ID,TX_AMOUNT,TX_FRAUD,TX_FRAUD_SCENARIO\n' +
      Array.from({length: 80}, (_, i) => ['g' + i, i * 60, i % 5, (i * 3) % 7, 10 + i, i % 4 === 0 ? 1 : 0, 0].join(',')).join('\n');
    const imported = await page.request.post(server.base + '/api/pipeline/datasets/import', {
      data: {source: 'handbook', csv, amount_to_eur: 1, name: 'Graph feature source'},
    });
    assert.equal(imported.status(), 201, await imported.text());
    const source = (await imported.json()).dataset;
    await page.goto(server.base + '/data-lab.html?dataset=' + source.id);
    await idle();
    let state = await snapshot();
    assert.equal(state.error, null);
    assert.equal(state.features.error, null);
    const graph = state.features.catalog.filter(feature => feature.group === 'graph');
    assert(graph.length >= 13, 'PageRank, PPR, approximation diagnostics and component recipes are discoverable.');
    assert(graph.every(feature => feature.available !== false));
    const originalIds = state.features.selected;
    assert.deepEqual(originalIds, ['amount']);
    await action('#dl-feature-add');
    state = await snapshot();
    assert(state.features.selected.length > originalIds.length);
    assert(graph.every(feature => !state.features.selected.includes(feature.id)), 'Add payment features leaves graph features individually opt-in.');
    await action('#dl-feature-source');
    await action('#dl-feature-add-graph');
    assert.deepEqual(new Set((await snapshot()).features.selected), new Set([...originalIds, ...graph.map(feature => feature.id)]));
    assert(await page.locator('#dl-feature-add-graph').isDisabled());
    assert(await page.locator('#dl-feature-add').isEnabled());
    await action('#dl-feature-all');
    assert.equal((await snapshot()).features.selected.length, state.features.catalog.filter(feature => feature.available !== false).length);
    await action('#dl-feature-source');
    await action('#dl-feature-add-graph');
    await page.locator('#dl-feature-filter').selectOption('graph');
    assert.equal(await page.locator('#dl-feature-list > .dl-feature').count(), graph.length);
    assert(!await featureInput('amount').count(), 'Graph filter hides source columns.');

    const chosen = ['graph_sender_pagerank', 'graph_recipient_pagerank', 'graph_ppr_sender_to_recipient',
      'graph_ppr_forward_error_bound', 'graph_pagerank_edge_lag'];
    for (const feature of graph) await featureInput(feature.id).setChecked(chosen.includes(feature.id));
    assert.deepEqual(new Set((await snapshot()).features.selected), new Set(['amount', ...chosen]), 'Every graph feature can be disabled individually.');
    assert(await page.locator('#dl-feature-add-graph').isEnabled());
    const ppr = graph.find(feature => feature.id === 'graph_ppr_sender_to_recipient');
    assert(ppr.method && ppr.parameters && ppr.approximation && ppr.orientation && ppr.readout);
    await page.locator('#dl-feature-search').fill(ppr.id);
    await featureRow(ppr.id).locator('summary').click();
    await featureRow(ppr.id).locator('.dl-feature-details').waitFor();
    const content = await featureRow(ppr.id).textContent();
    for (const label of ['Method', 'Graph orientation', 'Readout', 'Parameters', 'Approximation', 'History window', 'Minimum', 'Maximum', 'Mean']) assert(content.includes(label), label + ' is inspectable.');
    for (const value of [ppr.method, ppr.orientation, ppr.readout, ppr.approximation, JSON.stringify(ppr.parameters)]) assert(content.includes(value), 'The UI displays recipe metadata from the dataset.');
    assert(await featureRow(ppr.id).locator('[role=img]').isVisible());
    await page.locator('#dl-feature-search').fill('');
    const boundId = 'graph_ppr_forward_error_bound';
    await featureRow(boundId).locator('summary').click();
    await featureRow(boundId).locator('.dl-feature-details').waitFor();
    assert.match(await featureRow(boundId).textContent(), /bound|residual/i);
    await page.locator('#dl-feature-search').fill(ppr.method);
    assert(await featureRow(ppr.id).count(), 'Search also finds a feature by its method.');
    await page.locator('#dl-feature-search').fill('');
    await page.locator('#dl-feature-graph-note').locator('..').locator('summary').click();
    assert.match(await page.locator('#dl-feature-graph-note').textContent(), /strictly earlier timestamps/);
    assert.match(await page.locator('#dl-feature-graph-note').textContent(), /current payment.*same timestamp.*future identities/);
    const selected = (await snapshot()).features.selected;
    await page.locator('#dl-feature-name').fill('Stored graph model inputs');
    await action('#dl-feature-save');
    state = await snapshot();
    assert.equal(state.features.error, null);
    assert.equal(state.features.dirty, false);
    const savedId = state.selectedDatasetId;
    assert.notEqual(savedId, source.id);
    assert.deepEqual((await page.locator('#dl-sample-head th').allTextContents()).slice(3), selected);
    await page.reload();
    await idle();
    assert.deepEqual((await snapshot()).features.selected, selected, 'Graph selection persists with the stored dataset.');
    await page.locator('#dl-train').click();
    await idle('model-trainer');
    assert.equal((await snapshot('model-trainer')).datasetId, savedId);
    const trainerInputs = await page.locator('#mt-dataset-features').textContent();
    for (const id of chosen) assert(trainerInputs.includes(id), id + ' is available to the trainer.');
    await page.locator('#mt-model').selectOption('logistic_regression');
    await page.locator('#mt-train').click();
    await idle('model-trainer');
    state = await snapshot('model-trainer');
    assert.equal(state.error, null);
    assert.equal(state.runs.length, 1);
    assert.deepEqual(state.runs[0].feature_names, selected, 'A model trains on the stored graph feature selection.');

    await page.goto(server.base + '/data-lab.html?dataset=' + savedId);
    await idle();
    await page.setViewportSize({width: 390, height: 844});
    await page.locator('#dl-feature-filter').selectOption('graph');
    await page.locator('#dl-feature-search').fill(ppr.id);
    await featureRow(ppr.id).locator('summary').click();
    await featureRow(ppr.id).locator('.dl-feature-details').waitFor();
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'Graph parameters and statistics fit mobile.');
    if (process.env.PIPELINE_SCREENSHOTS) {
      await fs.mkdir(process.env.PIPELINE_SCREENSHOTS, {recursive: true});
      await page.locator('#dl-features').scrollIntoViewIfNeeded();
      await page.screenshot({path: path.join(process.env.PIPELINE_SCREENSHOTS, 'pipeline-graph-features-mobile.png')});
    }

    const header = ['Time', ...Array.from({length: 28}, (_, i) => 'V' + (i + 1)), 'Amount', 'Class'];
    const numericCSV = header.join(',') + '\n' + Array.from({length: 20}, (_, i) =>
      [i, ...Array.from({length: 28}, (_, j) => i % 2 + j), i + 1, i % 2].join(',')).join('\n');
    const numericResponse = await page.request.post(server.base + '/api/pipeline/datasets/import', {
      data: {source: 'ulb', csv: numericCSV, name: 'Numeric source without graph identities'},
    });
    assert.equal(numericResponse.status(), 201, await numericResponse.text());
    await page.goto(server.base + '/data-lab.html?dataset=' + (await numericResponse.json()).dataset.id);
    await idle();
    await page.locator('#dl-feature-filter').selectOption('graph');
    const unavailableGraph = (await snapshot()).features.catalog.filter(feature => feature.group === 'graph');
    assert.equal(unavailableGraph.length, graph.length);
    for (const feature of unavailableGraph) {
      assert.equal(feature.available, false);
      assert(await featureInput(feature.id).isDisabled());
      assert(feature.unavailable_reason);
      assert((await featureRow(feature.id).textContent()).includes(feature.unavailable_reason));
    }
    assert(await page.locator('#dl-feature-add-graph').isDisabled());
    assert.deepEqual(errors, []);
    console.log('PASS graph feature browser: graph/payment controls, individual selection, method/parameters/approximation/statistics, persisted values, trainer fit, mobile and unavailable identities.');
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
