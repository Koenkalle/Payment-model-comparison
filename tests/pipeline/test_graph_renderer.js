/* Real Canvas rendering and interaction contracts, without an API server. */
'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs/promises');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const ROOT = path.resolve(__dirname, '../..');

async function main() {
  const browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
  try {
    const page = await browser.newPage({ viewport: { width: 1200, height: 800 }, deviceScaleFactor: 2 });
    // Generous timeouts allow this stress test to share CI with model training.
    page.setDefaultTimeout(60000);
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const settled = () => page.waitForFunction(() => renderer.frame === null);
    const screenshot = async name => {
      if (!process.env.PIPELINE_SCREENSHOTS) return;
      await fs.mkdir(process.env.PIPELINE_SCREENSHOTS, { recursive: true });
      await page.screenshot({ path: path.join(process.env.PIPELINE_SCREENSHOTS, `graph-renderer-${name}.png`) });
    };
    await page.setContent(`
      <style>
        :root { color-scheme: light dark; } body { margin: 0; }
        .pl-root {
          --pl-wash: light-dark(#f4f7f6,#182024); --pl-line: light-dark(#dce5e2,#364246);
          --pl-panel: light-dark(#fff,#20272b); --pl-accent: light-dark(#127568,#71dbca);
          --pl-quiet: light-dark(#697578,#a5b2b7); --foreground: light-dark(#203330,#eef5f3);
          --pl-error: light-dark(#ae3743,#ffa0aa);
        }
        #graph { width: 100%; height: 600px; }
      </style>
      <main class="pl-root"><div id="graph"></div></main>
    `);
    await page.addScriptTag({ path: path.join(ROOT, 'shared/vendor/d3.min.js') });
    await page.addScriptTag({ path: path.join(ROOT, 'shared/graph/renderer.js') });
    await page.evaluate(() => {
      window.events = [];
      window.renderer = new DatasetGraphRenderer(document.getElementById('graph'), {
        onSelect: selection => events.push(['select', selection]),
        onExpand: id => events.push(['expand', id])
      });
      window.graph = {
        nodes: [
          { id: 'a', label: 'Customer A', type: 'customer', degree: 5 },
          { id: 'b', label: 'Terminal B', type: 'terminal', degree: 3 },
          { id: 'c', label: 'Merchant C', type: 'merchant', degree: 1 }
        ],
        edges: [
          { id: 'ab1', source: 'a', target: 'b', label: 0 },
          { id: 'ab2', source: 'a', target: 'b', label: 1 },
          { id: 'ba', source: 'b', target: 'a', label: -1 },
          { id: 'aa1', source: 'a', target: 'a', label: 1 },
          { id: 'aa2', source: 'a', target: 'a', label: 0 },
          { id: 'bc', source: 'b', target: 'c', label: 0 }
        ]
      };
      renderer.setGraph(graph, { layout: 'columns' });
    });
    await settled();
    assert.equal(await page.evaluate(() => renderer.canvas.width), 2400, 'Canvas honors device pixel ratio.');
    assert.equal(await page.evaluate(() => graph.nodes.some(node => 'x' in node)), false, 'Input nodes are not mutated.');
    assert.equal(await page.evaluate(() => graph.edges.some(edge => typeof edge.source !== 'string')), false, 'Input edges are not mutated.');
    assert(await page.evaluate(() => DatasetGraphRenderer.colorForType('customer') !== DatasetGraphRenderer.colorForType('terminal')),
      'Common entity types have distinct colors.');
    const geometry = await page.evaluate(() => renderer.edges.map(edge => edge.geometry.samples));
    assert.equal(new Set(geometry.slice(0, 3).map(samples => JSON.stringify(samples[Math.floor(samples.length / 2)]))).size, 3,
      'Parallel and reverse edges have distinct curves.');
    assert.notDeepEqual(geometry[3][10], geometry[4][10], 'Self loops remain distinct.');

    const screenPoint = (id, edge = false) => page.evaluate(({ id, edge }) => {
      const rect = renderer.canvas.getBoundingClientRect();
      const item = (edge ? renderer.edges : renderer.nodes).find(value => value.id === id);
      const p = edge ? item.geometry.samples[Math.floor(item.geometry.samples.length / 2)] : item;
      return { x: rect.left + renderer.transform.x + p.x * renderer.transform.k,
        y: rect.top + renderer.transform.y + p.y * renderer.transform.k };
    }, { id, edge });
    const click = async (id, edge = false) => {
      const p = await screenPoint(id, edge);
      await page.mouse.click(p.x, p.y);
    };
    await click('a');
    assert.deepEqual(await page.evaluate(() => events.at(-1)), ['select', { kind: 'node', id: 'a' }]);
    let p = await screenPoint('a');
    await page.mouse.dblclick(p.x, p.y);
    assert.deepEqual(await page.evaluate(() => events.at(-1)), ['expand', 'a']);
    await click('aa2', true);
    assert.deepEqual(await page.evaluate(() => events.at(-1)), ['select', { kind: 'edge', id: 'aa2' }]);
    await click('ab1', true);
    assert.deepEqual(await page.evaluate(() => events.at(-1)), ['select', { kind: 'edge', id: 'ab1' }]);
    const before = await page.evaluate(() => renderer.getPositions().find(node => node.id === 'a'));
    p = await screenPoint('a');
    await page.mouse.move(p.x, p.y); await page.mouse.down();
    await page.mouse.move(p.x + 70, p.y + 40, { steps: 6 }); await page.mouse.up();
    const after = await page.evaluate(() => renderer.getPositions().find(node => node.id === 'a'));
    assert(Math.abs(after.x - before.x) > 20, 'Dragging an entity changes its coordinates.');
    const zoom = await page.evaluate(() => renderer.transform.k);
    await page.locator('canvas').press('-');
    assert((await page.evaluate(() => renderer.transform.k)) < zoom, 'Keyboard zoom works.');
    await page.locator('canvas').press('f');
    await page.evaluate(() => renderer.setLayout('radial'));
    await settled();
    assert(await page.evaluate(() => renderer.getPositions().every(node => Number.isFinite(node.x) && Number.isFinite(node.y))));

    // A representative payment graph provides a useful optional visual artifact.
    await page.evaluate(() => {
      const nodes = Array.from({ length: 100 }, (_, i) => ({
        id: String(i), label: (i < 60 ? 'Customer ' : i < 85 ? 'Terminal ' : 'Merchant ') + i,
        type: i < 60 ? 'customer' : i < 85 ? 'terminal' : 'merchant', degree: 0
      }));
      const edges = Array.from({ length: 300 }, (_, i) => ({
        id: 'payment-' + i, source: String(i < 200 ? i % 60 : 60 + i % 25),
        target: String(i < 200 ? 60 + (i * 7 + i % 3) % 25 : 85 + (i * 7 + i % 3) % 15),
        label: i % 19 === 0 ? 1 : 0
      }));
      for (const edge of edges) { nodes[+edge.source].degree++; nodes[+edge.target].degree++; }
      renderer.setGraph({ nodes, edges }, { layout: 'force', selected: null });
    });
    await settled();
    await screenshot('100-nodes');
    await page.evaluate(() => {
      const nodes = Array.from({ length: 510 }, (_, i) => ({
        id: String(i), label: 'Entity ' + i, type: ['customer', 'terminal', 'merchant'][i % 3], degree: 8
      }));
      const edges = Array.from({ length: 2100 }, (_, i) => ({
        id: String(i), source: String(i % 500), target: String((i * 17 + 1) % 500), label: i % 15 === 0 ? 1 : 0
      }));
      renderer.setGraph({ nodes, edges }, { layout: 'force', selected: null });
    });
    await settled();
    const result = await page.evaluate(() => ({
      nodes: renderer.nodes.length, edges: renderer.edges.length, ticks: renderer.ticks,
      finite: renderer.getPositions().every(node => Number.isFinite(node.x) && Number.isFinite(node.y)),
      background: renderer.theme.background
    }));
    assert.equal(result.nodes, 500); assert.equal(result.edges, 2000);
    assert(result.ticks <= 180, 'Force simulation has a bounded number of iterations.');
    assert(result.finite);
    await screenshot('light');
    await page.emulateMedia({ colorScheme: 'dark' });
    await page.waitForFunction(background => renderer.theme.background !== background, result.background);
    await settled();
    await screenshot('dark');

    // Resizing an offscreen canvas clears its bitmap. It must repaint once even
    // while animation is suspended, including for full-page screenshots.
    await page.evaluate(() => { document.getElementById('graph').style.marginTop = '1000px'; });
    await page.waitForFunction(() => !renderer.onScreen && renderer.frame === null);
    await page.evaluate(() => { document.getElementById('graph').style.width = '80%'; });
    await page.waitForFunction(() => renderer.width === 960 && renderer.frame === null);
    assert(await page.evaluate(() => {
      const data = renderer.context.getImageData(0, 0, renderer.canvas.width, renderer.canvas.height).data;
      const colors = new Set();
      for (let i = 0; i < data.length; i += 4) {
        if (data[i + 3]) colors.add(data[i] + ',' + data[i + 1] + ',' + data[i + 2]);
        if (colors.size > 20) return true;
      }
      return false;
    }), 'Offscreen resize paints graph pixels.');
    await page.evaluate(() => {
      const container = document.getElementById('graph');
      container.style.marginTop = '0'; container.style.width = '100%';
    });
    await page.setViewportSize({ width: 390, height: 800 });
    await page.waitForFunction(() => renderer.width === 390);
    assert.equal(await page.evaluate(() => renderer.canvas.width), 780);
    await page.evaluate(() => renderer.destroy());
    assert.equal(await page.locator('canvas').count(), 0);
    assert.deepEqual(errors, []);
    console.log('PASS graph renderer: immutable inputs, HiDPI, parallel edges, self loops, selection, expansion, drag, keyboard controls, layouts, 500-node/2000-edge limits, themes, offscreen pixel regression, resize, cleanup.');
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
