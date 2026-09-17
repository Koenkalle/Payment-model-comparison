/* Build first. Exercise the graph explorer against real stored datasets. */
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
    'import sys',
    'from types import SimpleNamespace',
    'from framework.comparison_service import make_server',
    'from framework.pipeline_service import PipelineService',
    'pipeline = PipelineService(sys.argv[1])',
    "server = make_server(SimpleNamespace(models=lambda: {'models': []}), port=0, pipeline_service=pipeline)",
    'print(server.server_port, flush=True)',
    'server.serve_forever()',
  ].join('\n');
  const child = spawn(process.env.PYTHON_BINARY || 'python', ['-c', code, directory], {
    cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'],
  });
  let logs = '';
  child.stderr.on('data', chunk => { logs += chunk; });
  const port = await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      child.kill();
      reject(Error('Graph browser server startup timeout: ' + logs));
    }, 30000);
    child.stdout.once('data', chunk => {
      clearTimeout(timeout);
      resolve(Number(String(chunk).trim()));
    });
    child.once('error', error => { clearTimeout(timeout); reject(error); });
    child.once('exit', code => {
      clearTimeout(timeout);
      reject(Error('Graph browser server exited ' + code + ': ' + logs));
    });
  });
  return {child, base: 'http://127.0.0.1:' + port, logs: () => logs};
}

async function stop(server) {
  if (server?.child.exitCode === null) {
    const exited = once(server.child, 'exit');
    server.child.kill();
    await exited;
  }
}

function assertBounded(snapshot, nodeLimit, edgeLimit) {
  const {nodes, edges} = snapshot.graph;
  assert(nodes.length <= nodeLimit, 'Rendered node count respects the viewport budget.');
  assert(edges.length <= edgeLimit, 'Rendered edge count respects the viewport budget.');
  const ids = new Set(nodes.map(node => node.id));
  assert.equal(ids.size, nodes.length, 'Repeated exploration does not duplicate nodes.');
  assert.equal(new Set(edges.map(edge => edge.id)).size, edges.length, 'Edges retain unique event identities.');
  assert(edges.every(edge => ids.has(edge.source) && ids.has(edge.target)), 'Every edge has visible endpoints.');
}

async function main() {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'payment-graph-browser-'));
  let server, browser;
  try {
    if (process.env.PIPELINE_SCREENSHOTS) await fs.mkdir(process.env.PIPELINE_SCREENSHOTS, {recursive: true});
    server = await start(directory);
    browser = await chromium.launch({headless: true, args: ['--no-sandbox']});
    const page = await browser.newPage({viewport: {width: 1440, height: 1050}});
    const errors = [], graphRequests = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => {
      if (/\/graph\/query(?:\?|$)/.test(request.url())) graphRequests.push(request.url());
    });
    const dataIdle = async () => {
      await page.waitForFunction(() => !!document.getElementById('data-lab')?.demo);
      await page.evaluate(() => document.getElementById('data-lab').demo.whenIdle());
    };
    const graphIdle = async () => {
      await page.waitForFunction(() => !!document.getElementById('data-lab')?.demo.graph);
      await page.evaluate(() => document.getElementById('data-lab').demo.graph.whenIdle());
    };
    const graphSnapshot = () => page.evaluate(() => document.getElementById('data-lab').demo.graph.getSnapshot());
    const dataSnapshot = () => page.evaluate(() => document.getElementById('data-lab').demo.getSnapshot());
    const action = async selector => { await page.locator(selector).click(); await graphIdle(); };
    const open = async () => { await action('#dl-graph-open'); };
    const apply = async () => { await action('#gx-apply'); };
    const painted = async () => {
      await page.locator('#gx-canvas canvas').scrollIntoViewIfNeeded();
      await page.waitForFunction(() => {
        const canvas = document.querySelector('#gx-canvas canvas');
        const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
        let colored = 0;
        for (let i = 0; i < pixels.length; i += 4) {
          if (pixels[i + 3] && Math.max(pixels[i], pixels[i + 1], pixels[i + 2]) - Math.min(pixels[i], pixels[i + 1], pixels[i + 2]) > 35) colored++;
          if (colored > 150) return true;
        }
        return false;
      }, undefined, {timeout: 10000});
    };

    await page.goto(server.base + '/data-lab.html');
    await dataIdle();
    await page.locator('#dl-generator').selectOption('handbook_generator');
    await page.locator('#dl-param-transactions').fill('400');
    await page.locator('#dl-param-customer_count').fill('16');
    await page.locator('#dl-param-terminal_count').fill('8');
    await page.locator('#dl-param-fraud_rate').fill('0.3');
    await page.locator('#dl-name').fill('Graph browser handbook');
    await page.locator('#dl-create').click();
    await dataIdle();
    const dataset = await dataSnapshot();
    assert.equal(dataset.error, null);
    assert.equal(dataset.detail.rows, 400);
    assert.equal(graphRequests.length, 0, 'Data inspection does not fetch graph records before the explorer opens.');
    assert(await page.locator('#dl-graph-open').isEnabled());

    // The initial index request is cancellable before any graph records arrive.
    const summaryURL = server.base + '/api/pipeline/datasets/' + dataset.selectedDatasetId + '/graph';
    let releaseSummary;
    const summaryReleased = new Promise(resolve => { releaseSummary = resolve; });
    const summaryStarted = page.waitForRequest(request => request.url() === summaryURL, {timeout: 5000});
    await page.route(summaryURL, async route => {
      await summaryReleased;
      await route.continue().catch(() => {});
    }, {times: 1});
    await page.locator('#dl-graph-open').click();
    await summaryStarted;
    assert(await page.locator('#gx-cancel').isVisible());
    await page.locator('#gx-cancel').click();
    releaseSummary();
    await graphIdle();
    assert.equal((await graphSnapshot()).summary, null, 'Canceled summary cannot populate the explorer.');
    assert.equal((await graphSnapshot()).busy, false);
    await page.locator('#dl-graph-close').click();

    await open();
    let state = await graphSnapshot();
    assert.equal(state.datasetId, dataset.selectedDatasetId);
    assert.equal(state.error, null);
    assert.equal(state.summary.counts.nodes, 24);
    assert.equal(state.summary.counts.edges, 400);
    assert(state.graph.nodes.length > 0 && state.graph.edges.length > 0);
    assertBounded(state, 500, 2000);
    const allNodes = state.graph.nodes;
    assert(await page.locator('#gx-canvas').isVisible());
    await page.locator('#gx-layout').selectOption('columns');
    await page.locator('#gx-labels').uncheck();
    await page.locator('#gx-fit').click();
    assert.deepEqual((await graphSnapshot()).graph.nodes, allNodes, 'Layout controls preserve graph records.');
    const target = await page.evaluate(() => {
      const renderer = document.getElementById('data-lab').demo.graph.renderer;
      const node = renderer.getPositions()[0], transform = renderer.transform;
      return {id: node.id, x: node.x * transform.k + transform.x, y: node.y * transform.k + transform.y};
    });
    await page.locator('#gx-canvas canvas').click({position: {x: target.x, y: target.y}});
    assert.deepEqual((await graphSnapshot()).selection, {kind: 'node', id: target.id}, 'Canvas pointer selection opens the entity inspector.');
    await page.locator('#gx-canvas canvas').press('Escape');
    assert.equal((await graphSnapshot()).selection, null);
    await page.locator('#gx-labels').check();
    await painted();
    if (process.env.PIPELINE_SCREENSHOTS) {
      await page.screenshot({path: path.join(process.env.PIPELINE_SCREENSHOTS, 'pipeline-graph-desktop.png')});
      await page.locator('#gx-canvas').screenshot({path: path.join(process.env.PIPELINE_SCREENSHOTS, 'pipeline-graph-canvas.png')});
    }

    // Each new page replaces the viewport so repeated browsing cannot grow it.
    await page.locator('#gx-node-limit').fill('6');
    await page.locator('#gx-edge-limit').fill('8');
    await apply();
    state = await graphSnapshot();
    assertBounded(state, 6, 8);
    assert(state.page.has_more);
    const firstEdgeIds = state.graph.edges.map(edge => edge.id);
    await action('#gx-next');
    state = await graphSnapshot();
    assertBounded(state, 6, 8);
    assert(state.graph.edges.length > 0);
    assert(state.graph.edges.every(edge => !firstEdgeIds.includes(edge.id)), 'Next sample advances to unseen event records.');
    await action('#gx-undo');
    assert.deepEqual((await graphSnapshot()).graph.edges.map(edge => edge.id), firstEdgeIds);

    // Both graph tables expose the same property selection as canvas clicks.
    await page.locator('#gx-nodes button').first().click();
    state = await graphSnapshot();
    assert(state.selection, 'A node can be selected without a pointing device on the canvas.');
    assert((await page.locator('#gx-inspect').textContent()).trim().length > 0);
    await action('#gx-expand');
    assertBounded(await graphSnapshot(), 6, 8);
    await action('#gx-focus');
    state = await graphSnapshot();
    assertBounded(state, 6, 8);
    assert(state.graph.nodes.length > 0);
    await action('#gx-undo');

    await page.locator('#gx-tabs-edge').click();
    await page.locator('#gx-edges button').first().click();
    state = await graphSnapshot();
    assert(state.selection, 'An event edge can be selected from the transaction table.');
    assert.match(await page.locator('#gx-inspect').textContent(), /time|amount|outcome/i);

    // Search reaches stored entities outside the currently rendered subset.
    const visible = new Set(state.graph.nodes.map(node => node.id));
    const missing = allNodes.find(node => !visible.has(node.id) && node.out_degree > 0 && node.in_degree === 0);
    assert(missing, 'The fixture contains nodes outside the small viewport.');
    await page.locator('#gx-search').fill(missing.label);
    await action('#gx-search-button');
    await page.locator('#gx-search-results button').filter({hasText: missing.label}).first().click();
    assert.equal((await graphSnapshot()).selection.id, missing.id);
    await action('#gx-focus');
    state = await graphSnapshot();
    assert(state.graph.nodes.some(node => node.id === missing.id), 'A search result can focus an entity outside the loaded slice.');
    assertBounded(state, 6, 8);

    // Direction applies to neighborhood exploration after applying the controls.
    for (const direction of ['incoming', 'outgoing']) {
      await page.locator('#gx-direction').selectOption(direction);
      await apply();
      await action('#gx-search-button');
      await page.locator('#gx-search-results button').filter({hasText: missing.label}).first().click();
      await action('#gx-focus');
      state = await graphSnapshot();
      assert(state.graph.nodes.some(node => node.id === missing.id));
      if (direction === 'incoming') {
        assert.equal(state.graph.edges.length, 0, 'A customer source has no incoming payment events.');
      } else {
        assert(state.graph.edges.length > 0);
        assert(state.graph.edges.every(edge => edge.source === missing.id), 'Outgoing exploration follows source endpoints.');
      }
    }

    // Filters are exact source record predicates, including empty matches.
    await page.locator('#gx-label').selectOption('1');
    await apply();
    state = await graphSnapshot();
    assert(state.graph.edges.length > 0);
    assert(state.graph.edges.every(edge => edge.label === 1));
    await page.locator('#gx-label').selectOption('-1');
    await apply();
    state = await graphSnapshot();
    assert.equal(state.error, null);
    assert.equal(state.graph.edges.length, 0, 'Known-only data has no unknown-outcome edges.');
    assert.match(await page.locator('#gx-status').textContent(), /0|no|empty/i);
    await action('#gx-reset');

    // The source time interval is inclusive at the start and exclusive at stop.
    state = await graphSnapshot();
    const [first, second] = state.graph.edges;
    await page.locator('#gx-start').fill(String(first.time));
    await page.locator('#gx-stop').fill(String(second.time));
    await page.locator('#gx-edge-type').selectOption(first.type);
    await apply();
    state = await graphSnapshot();
    assert.deepEqual(state.graph.edges.map(edge => edge.id), [first.id]);

    // Failed requests preserve the last usable graph and recover on retry.
    const queryURL = '**/api/pipeline/datasets/' + dataset.selectedDatasetId + '/graph/query';
    await page.route(queryURL, route => route.fulfill({
      status: 503, contentType: 'application/json', body: JSON.stringify({error: 'Graph test temporary outage'}),
    }), {times: 1});
    await apply();
    state = await graphSnapshot();
    assert.match(state.error, /temporary outage/);
    assert.deepEqual(state.graph.edges.map(edge => edge.id), [first.id]);
    assert(await page.locator('#gx-apply').isEnabled());
    await apply();
    assert.equal((await graphSnapshot()).error, null);
    assert((await graphSnapshot()).historyLength <= 12, 'Exploration history keeps a bounded number of viewport snapshots.');

    const downloadPromise = page.waitForEvent('download', {timeout: 5000});
    await page.locator('#gx-export').click();
    const download = await downloadPromise;
    const exported = JSON.parse(await fs.readFile(await download.path(), 'utf8'));
    assert.equal(exported.dataset.id, dataset.selectedDatasetId);
    assert.equal(exported.scope, 'visible subset');
    assert.deepEqual(exported.nodes.map(node => node.id), (await graphSnapshot()).graph.nodes.map(node => node.id));
    assert.deepEqual(exported.edges.map(edge => edge.id), (await graphSnapshot()).graph.edges.map(edge => edge.id));

    await action('#gx-reset');
    await page.setViewportSize({width: 390, height: 844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'Graph explorer fits a mobile viewport.');
    assert(await page.locator('#gx-canvas').isVisible());
    await page.locator('#gx-fit').click();
    await painted();
    if (process.env.PIPELINE_SCREENSHOTS) {
      await page.screenshot({path: path.join(process.env.PIPELINE_SCREENSHOTS, 'pipeline-graph-mobile.png')});
    }
    await page.locator('#dl-graph-close').click();
    assert(await page.locator('#dl-graph-panel').isHidden());
    await page.setViewportSize({width: 1440, height: 1050});

    // A second graph provider uses the same controls and node/edge contract.
    const generated = await page.request.post(server.base + '/api/pipeline/datasets/generate', {
      data: {generator: 'synthetic_payments', name: 'Graph browser scenario', parameters: {name: 'split', size: 'small', seed: 42}},
    });
    assert.equal(generated.status(), 201);
    const scenario = (await generated.json()).dataset;
    await page.locator('#dl-refresh').click();
    await dataIdle();
    // A late query from the previous dataset must not replace the new selection.
    await open();
    let releaseQuery;
    const released = new Promise(resolve => { releaseQuery = resolve; });
    const started = page.waitForRequest(request => request.url().endsWith('/' + dataset.selectedDatasetId + '/graph/query'), {timeout: 5000});
    await page.route(queryURL, async route => {
      await released;
      await route.continue().catch(() => {}); // Closing the view may abort this request.
    }, {times: 1});
    await page.locator('#gx-apply').click();
    await started;
    await page.locator('#dl-graph-close').click();
    await page.locator('#dl-library button').filter({hasText: scenario.name}).click();
    releaseQuery();
    await dataIdle();
    await open();
    state = await graphSnapshot();
    assert.equal(state.datasetId, scenario.id);
    assert.equal(state.error, null);
    assert(state.graph.edges.length > 0);
    assertBounded(state, 500, 2000);
    await page.locator('#dl-graph-close').click();

    // Numeric datasets clearly explain why graph exploration is unavailable.
    await page.locator('#dl-method').selectOption('import');
    await page.locator('#dl-source').selectOption('ulb');
    const header = ['Time', ...Array.from({length: 28}, (_, i) => 'V' + (i + 1)), 'Amount', 'Class'];
    const csv = header.join(',') + '\n' + Array.from({length: 20}, (_, i) =>
      [i, ...Array.from({length: 28}, (_, j) => i % 2 + j), i + 1, i % 2].join(',')).join('\n');
    await page.locator('#dl-file').setInputFiles({name: 'graph-browser-ulb.csv', mimeType: 'text/csv', buffer: Buffer.from(csv)});
    await page.locator('#dl-name').fill('Graph browser numeric');
    await page.locator('#dl-create').click();
    await dataIdle();
    assert.equal((await dataSnapshot()).error, null);
    assert(await page.locator('#dl-graph-open').isDisabled());
    assert.match(await page.locator('#dl-graph-reason').textContent(), /numeric|identit|graph|entit/i);
    assert(await page.locator('#dl-graph-panel').isHidden());
    assert.deepEqual(errors, []);
    console.log('PASS graph browser: lazy loading, cancellation, painted canvas, generated graph counts, bounded pagination/history, inspection, expansion, focus, undo, full-dataset search, directions, source filters, retry, export, mobile, dataset races, provider switching and numeric-only guidance.');
  } catch (error) {
    if (server) console.error(server.logs().slice(-8000));
    throw error;
  } finally {
    await browser?.close();
    await stop(server);
    await fs.rm(directory, {recursive: true, force: true});
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
