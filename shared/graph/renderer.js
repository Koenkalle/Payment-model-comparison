/* Bounded, reusable Canvas graph viewport. Requires the shared D3 v7 bundle. */
(function (global) {
  'use strict';

  const LIMITS = Object.freeze({ nodes: 500, edges: 2000 });
  const PALETTE = ['#258b83', '#687bd2', '#c38b32', '#a36bc1', '#428bad', '#b96e88', '#849847', '#b77844'];
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  const point = (x, y) => ({ x, y });
  const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const nodeRadius = node => 6 + Math.min(6, Math.log2(1 + Math.max(0, Number(node.degree) || 0)));

  function colorForType(type) {
    let hash = 2166136261;
    for (const character of String(type || 'entity')) hash = Math.imul(hash ^ character.charCodeAt(0), 16777619);
    hash ^= hash >>> 16;
    hash = Math.imul(hash, 0x7feb352d);
    hash ^= hash >>> 15;
    return PALETTE[(hash >>> 0) % PALETTE.length];
  }

  function distanceToSegment(p, a, b) {
    const dx = b.x - a.x, dy = b.y - a.y;
    const t = clamp(((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy || 1), 0, 1);
    return Math.hypot(p.x - a.x - t * dx, p.y - a.y - t * dy);
  }

  // Curved parallel links and cubic self loops share geometry for paint and hit tests.
  function edgeGeometry(edge) {
    const source = edge.source, target = edge.target;
    const startRadius = nodeRadius(source) + 1, endRadius = nodeRadius(target) + 2;
    let start, end, controls, samples;
    if (source.id === target.id) {
      const radius = startRadius + 22 + edge.loopOffset;
      start = point(source.x - startRadius * .72, source.y - startRadius * .7);
      end = point(source.x + endRadius * .72, source.y - endRadius * .7);
      controls = [point(source.x - radius * 1.3, source.y - radius * 1.9), point(source.x + radius * 1.3, source.y - radius * 1.9)];
      samples = Array.from({ length: 21 }, (_, index) => {
        const t = index / 20, u = 1 - t;
        return point(u ** 3 * start.x + 3 * u ** 2 * t * controls[0].x + 3 * u * t ** 2 * controls[1].x + t ** 3 * end.x,
          u ** 3 * start.y + 3 * u ** 2 * t * controls[0].y + 3 * u * t ** 2 * controls[1].y + t ** 3 * end.y);
      });
    } else {
      const dx = target.x - source.x, dy = target.y - source.y, length = Math.hypot(dx, dy) || 1;
      const control = point((source.x + target.x) / 2 - dy / length * edge.curveOffset,
        (source.y + target.y) / 2 + dx / length * edge.curveOffset);
      const startLength = distance(source, control) || 1, endLength = distance(target, control) || 1;
      start = point(source.x + (control.x - source.x) / startLength * startRadius,
        source.y + (control.y - source.y) / startLength * startRadius);
      end = point(target.x + (control.x - target.x) / endLength * endRadius,
        target.y + (control.y - target.y) / endLength * endRadius);
      controls = [control];
      const sampleCount = edge.curveOffset ? 13 : 2;
      samples = Array.from({ length: sampleCount }, (_, index) => {
        const t = index / (sampleCount - 1), u = 1 - t;
        return point(u * u * start.x + 2 * u * t * control.x + t * t * end.x,
          u * u * start.y + 2 * u * t * control.y + t * t * end.y);
      });
    }
    const lastControl = controls[controls.length - 1];
    return {
      start, end, controls, samples,
      angle: Math.atan2(end.y - lastControl.y, end.x - lastControl.x),
      bounds: {
        minX: Math.min(...samples.map(p => p.x)), maxX: Math.max(...samples.map(p => p.x)),
        minY: Math.min(...samples.map(p => p.y)), maxY: Math.max(...samples.map(p => p.y))
      }
    };
  }

  function routeEdges(edges) {
    const groups = new Map();
    for (const edge of edges) {
      const pair = [edge.source.id, edge.target.id].sort();
      const key = JSON.stringify(pair);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(edge);
    }
    for (const group of groups.values()) {
      group.sort((a, b) => a.id.localeCompare(b.id));
      group.forEach((edge, index) => {
        const direction = edge.source.id <= edge.target.id ? 1 : -1;
        edge.curveOffset = (index - (group.length - 1) / 2) * Math.min(24, 240 / Math.max(1, group.length - 1)) * direction;
        edge.loopOffset = index * Math.min(13, 150 / Math.max(1, group.length - 1));
      });
    }
  }

  /**
   * Renderer owns viewport and layout only. Dataset loading, filtering, metadata,
   * accessible tables and property inspection belong to the embedding controller.
   * Input nodes and edges are copied; D3 never mutates the caller's data.
   */
  class DatasetGraphRenderer {
    static colorForType(type) { return colorForType(type); }
    static get limits() { return LIMITS; }

    constructor(container, callbacks = {}) {
      if (!container || !global.d3) throw new Error('Graph renderer requires a container and the shared D3 library.');
      this.container = container;
      this.callbacks = callbacks;
      this.nodes = [];
      this.edges = [];
      this.positions = new Map();
      this.selection = null;
      this.layout = 'force';
      this.labels = true;
      this.transform = { x: 0, y: 0, k: 1 };
      this.width = 1;
      this.height = 600;
      this.frame = null;
      this.simulation = null;
      this.ticks = 0;
      this.destroyed = false;
      this.onScreen = true;
      this.pointers = new Map();
      this.listeners = [];
      this.neighbors = new Map();
      this.canvas = document.createElement('canvas');
      this.canvas.className = 'dg-canvas';
      this.canvas.tabIndex = 0;
      this.canvas.setAttribute('role', 'img');
      this.canvas.setAttribute('aria-label', 'Interactive directed dataset graph. Click an entity or relationship to inspect. Double-click an entity to expand. Drag entities to arrange, drag the background to pan, or scroll to zoom. Keyboard: plus and minus zoom, F fits the view, and arrow keys pan.');
      Object.assign(this.canvas.style, { display: 'block', width: '100%', height: container.getBoundingClientRect().height ? '100%' : '600px', touchAction: 'none', cursor: 'grab' });
      this.container.appendChild(this.canvas);
      this.context = this.canvas.getContext('2d');
      if (!this.context) throw new Error('Canvas rendering is unavailable in this browser.');
      this.probe = document.createElement('span');
      this.probe.setAttribute('aria-hidden', 'true');
      Object.assign(this.probe.style, { position: 'absolute', visibility: 'hidden', pointerEvents: 'none', width: '0', height: '0' });
      this.container.appendChild(this.probe);
      this._bindEvents();
      this._readTheme();
      this.resize();
      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(this.container);
      if (global.IntersectionObserver) {
        this.intersectionObserver = new IntersectionObserver(entries => {
          this.onScreen = entries[0].isIntersecting;
          if (!this.onScreen) this._cancelFrame();
          this._requestFrame();
        });
        this.intersectionObserver.observe(this.canvas);
      }
      this.themeObserver = new MutationObserver(() => { this._readTheme(); this._requestFrame(); });
      for (const element of new Set([document.documentElement, document.body, container.closest('.pl-root')].filter(Boolean))) {
        this.themeObserver.observe(element, { attributes: true, attributeFilter: ['class', 'style', 'data-theme'] });
      }
      this.media = global.matchMedia('(prefers-color-scheme: dark)');
      this.mediaListener = () => { this._readTheme(); this._requestFrame(); };
      this.media.addEventListener('change', this.mediaListener);
    }

    setGraph(graph, options = {}) {
      if (this.destroyed) return;
      this._rememberPositions();
      this.simulation?.stop();
      const previousIds = new Set(this.nodes.map(node => node.id));
      const ids = new Set();
      this.nodes = [];
      for (const raw of (graph.nodes || [])) {
        const id = String(raw.id);
        if (ids.has(id) || this.nodes.length >= LIMITS.nodes) continue;
        ids.add(id);
        const cached = this.positions.get(id);
        const index = this.nodes.length, angle = index * Math.PI * (3 - Math.sqrt(5)), radius = 19 * Math.sqrt(index);
        this.nodes.push({ ...raw, id, x: cached?.x ?? Math.cos(angle) * radius, y: cached?.y ?? Math.sin(angle) * radius });
      }
      const nodesById = new Map(this.nodes.map(node => [node.id, node]));
      const edgeIds = new Set();
      this.edges = [];
      this.neighbors = new Map(this.nodes.map(node => [node.id, new Set()]));
      for (const raw of (graph.edges || [])) {
        const id = String(raw.id), source = nodesById.get(String(raw.source)), target = nodesById.get(String(raw.target));
        if (!source || !target || edgeIds.has(id) || this.edges.length >= LIMITS.edges) continue;
        edgeIds.add(id);
        this.edges.push({ ...raw, id, source, target });
        this.neighbors.get(source.id).add(target.id);
        this.neighbors.get(target.id).add(source.id);
      }
      // Place newly explored neighbors near an existing endpoint before relaxing.
      for (const node of this.nodes) {
        if (this.positions.has(node.id)) continue;
        const anchors = [...this.neighbors.get(node.id)].filter(id => previousIds.has(id)).map(id => nodesById.get(id));
        if (anchors.length) {
          node.x += anchors.reduce((total, anchor) => total + anchor.x, 0) / anchors.length;
          node.y += anchors.reduce((total, anchor) => total + anchor.y, 0) / anchors.length;
        }
      }
      routeEdges(this.edges);
      if (Object.hasOwn(options, 'labels')) this.labels = !!options.labels;
      if (Object.hasOwn(options, 'selected')) this.selection = options.selected ? { kind: options.selected.kind, id: String(options.selected.id) } : null;
      if (this.selection && !(this.selection.kind === 'node' ? ids : edgeIds).has(String(this.selection.id))) this.selection = null;
      this.layout = ['force', 'radial', 'columns'].includes(options.layout) ? options.layout : this.layout;
      const initialView = !previousIds.size || !this.nodes.some(node => previousIds.has(node.id));
      this._applyLayout(initialView);
      this.canvas.setAttribute('aria-description', `${this.nodes.length} entities and ${this.edges.length} directed relationships currently visible.`);
      this._requestFrame();
    }

    select(selection) {
      this.selection = selection ? { kind: selection.kind, id: String(selection.id) } : null;
      this._requestFrame();
    }

    fit() {
      if (!this.nodes.length) {
        this.transform = { x: this.width / 2, y: this.height / 2, k: 1 };
      } else {
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (const node of this.nodes) {
          minX = Math.min(minX, node.x - 35); maxX = Math.max(maxX, node.x + 35);
          minY = Math.min(minY, node.y - 35); maxY = Math.max(maxY, node.y + 35);
        }
        for (const edge of this.edges) {
          const bounds = edgeGeometry(edge).bounds;
          minX = Math.min(minX, bounds.minX); maxX = Math.max(maxX, bounds.maxX);
          minY = Math.min(minY, bounds.minY); maxY = Math.max(maxY, bounds.maxY);
        }
        const k = clamp(Math.min((this.width - 70) / Math.max(90, maxX - minX), (this.height - 80) / Math.max(90, maxY - minY)), .08, 1.5);
        this.transform = { x: this.width / 2 - (minX + maxX) / 2 * k, y: this.height / 2 - (minY + maxY) / 2 * k, k };
      }
      this._requestFrame();
    }

    zoomBy(factor) { this._zoomAt(point(this.width / 2, this.height / 2), factor); }
    setLabels(value) { this.labels = !!value; this._requestFrame(); }
    setLayout(name) {
      if (!['force', 'radial', 'columns'].includes(name) || this.layout === name) return;
      this.layout = name;
      this._applyLayout(true);
    }
    getPositions() { return this.nodes.map(node => ({ id: node.id, x: node.x, y: node.y })); }

    resize() {
      if (this.destroyed) return;
      const rect = this.container.getBoundingClientRect();
      const width = Math.max(1, rect.width), height = Math.max(1, rect.height || 600);
      this.transform.x += (width - this.width) / 2;
      this.transform.y += (height - this.height) / 2;
      this.width = width; this.height = height;
      const ratio = Math.min(3, global.devicePixelRatio || 1);
      this.canvas.width = Math.round(width * ratio);
      this.canvas.height = Math.round(height * ratio);
      this.ratio = ratio;
      this._requestFrame();
    }

    destroy() {
      this.destroyed = true;
      this._cancelFrame();
      this.simulation?.stop();
      this.resizeObserver?.disconnect();
      this.intersectionObserver?.disconnect();
      this.themeObserver?.disconnect();
      this.media?.removeEventListener('change', this.mediaListener);
      for (const [element, type, listener, options] of this.listeners) element.removeEventListener(type, listener, options);
      this.canvas.remove();
      this.probe.remove();
      this.nodes = []; this.edges = []; this.positions.clear(); this.pointers.clear();
    }

    _rememberPositions() {
      for (const node of this.nodes) this.positions.set(node.id, point(node.x, node.y));
      // Keep revisited neighborhoods stable without retaining an entire dataset.
      while (this.positions.size > 5000) this.positions.delete(this.positions.keys().next().value);
    }

    _applyLayout(fit) {
      this.simulation?.stop();
      this.simulation = null;
      this.columnHeadings = [];
      for (const node of this.nodes) { node.fx = null; node.fy = null; node.vx = 0; node.vy = 0; }
      if (this.layout === 'radial') this._radialLayout();
      else if (this.layout === 'columns') this._columnLayout();
      else if (this.nodes.length) {
        this.simulation = global.d3.forceSimulation(this.nodes).stop()
          .force('charge', global.d3.forceManyBody().strength(-150).distanceMax(600).theta(.9))
          .force('links', global.d3.forceLink(this.edges.filter(edge => edge.source !== edge.target)).id(node => node.id).distance(70).strength(.16))
          .force('collision', global.d3.forceCollide(node => nodeRadius(node) + 12).iterations(1))
          .force('x', global.d3.forceX(0).strength(.025))
          .force('y', global.d3.forceY(0).strength(.025))
          .alpha(fit ? 1 : .45).alphaDecay(.035).alphaMin(.018).velocityDecay(.4);
        this.ticks = 0;
        // A small synchronous warmup avoids presenting an initially tangled pile.
        this.simulation.tick(6);
        this.layoutDeadline = performance.now() + (fit ? 3000 : 1800);
      }
      this.fitWhenSettled = fit;
      if (fit) this.fit();
      this._requestFrame();
    }

    _radialLayout() {
      if (!this.nodes.length) return;
      const byId = new Map(this.nodes.map(node => [node.id, node]));
      const focus = byId.get(this.selection?.kind === 'node' ? String(this.selection.id) : '') ||
        [...this.nodes].sort((a, b) => this.neighbors.get(b.id).size - this.neighbors.get(a.id).size || a.id.localeCompare(b.id))[0];
      const depths = new Map([[focus.id, 0]]), queue = [focus.id];
      for (let cursor = 0; cursor < queue.length; cursor++) {
        const id = queue[cursor];
        for (const next of this.neighbors.get(id)) if (!depths.has(next)) { depths.set(next, depths.get(id) + 1); queue.push(next); }
      }
      const lastDepth = Math.max(...depths.values()) + 1, rings = new Map();
      for (const node of this.nodes) {
        const depth = depths.get(node.id) ?? lastDepth;
        if (!rings.has(depth)) rings.set(depth, []);
        rings.get(depth).push(node);
      }
      let previousRadius = 0;
      for (const [depth, ring] of [...rings].sort((a, b) => a[0] - b[0])) {
        ring.sort((a, b) => String(a.type).localeCompare(String(b.type)) || a.id.localeCompare(b.id));
        const radius = depth ? Math.max(previousRadius + 95, ring.length * 29 / (2 * Math.PI)) : 0;
        ring.forEach((node, index) => { const angle = 2 * Math.PI * index / ring.length - Math.PI / 2; node.x = Math.cos(angle) * radius; node.y = Math.sin(angle) * radius; });
        previousRadius = radius;
      }
    }

    _columnLayout() {
      const groups = new Map();
      for (const node of this.nodes) {
        const type = String(node.type || 'entity');
        if (!groups.has(type)) groups.set(type, []);
        groups.get(type).push(node);
      }
      let offset = 0;
      for (const [type, group] of [...groups].sort((a, b) => a[0].localeCompare(b[0]))) {
        group.sort((a, b) => (Number(b.degree) || 0) - (Number(a.degree) || 0) || a.id.localeCompare(b.id));
        const rows = Math.min(22, Math.max(6, Math.ceil(Math.sqrt(group.length) * 1.5)));
        const columns = Math.ceil(group.length / rows), height = Math.min(rows, group.length) * 46;
        group.forEach((node, index) => { node.x = offset + Math.floor(index / rows) * 95; node.y = index % rows * 46 - height / 2; });
        this.columnHeadings.push({ text: type, x: offset + (columns - 1) * 95 / 2, y: -height / 2 - 32 });
        offset += columns * 95 + 100;
      }
    }

    _readTheme() {
      const read = (variable, fallback) => {
        this.probe.style.color = `var(${variable}, ${fallback})`;
        return getComputedStyle(this.probe).color;
      };
      this.theme = {
        background: read('--pl-wash', '#f4f7f6'), text: read('--foreground', '#203330'),
        muted: read('--pl-quiet', '#697578'), edge: read('--pl-quiet', '#84918d'),
        line: read('--pl-line', '#dce5e2'), panel: read('--pl-panel', '#ffffff'),
        selected: read('--pl-accent', '#127568'), fraud: read('--pl-error', '#b53f50')
      };
    }

    _requestFrame() {
      if (this.destroyed || this.frame !== null || document.hidden) return;
      this.frame = requestAnimationFrame(() => {
        this.frame = null;
        if (this.destroyed) return;
        if (this.onScreen && this.simulation && this.simulation.alpha() > this.simulation.alphaMin() && this.ticks < 180 && performance.now() < this.layoutDeadline) {
          const start = performance.now();
          do { this.simulation.tick(); this.ticks++; } while (performance.now() - start < 7 && this.ticks % 4 !== 0 && this.ticks < 180);
          if (this.fitWhenSettled && (this.ticks % 20 === 0 || this.simulation.alpha() <= this.simulation.alphaMin())) this.fit();
          this._requestFrame();
        } else {
          this._rememberPositions();
          if (this.onScreen && this.fitWhenSettled) { this.fitWhenSettled = false; this.fit(); }
        }
        this._draw();
      });
    }

    _cancelFrame() { if (this.frame !== null) cancelAnimationFrame(this.frame); this.frame = null; }

    _draw() {
      const ctx = this.context, { x, y, k } = this.transform;
      ctx.setTransform(this.ratio, 0, 0, this.ratio, 0, 0);
      ctx.fillStyle = this.theme.background;
      ctx.fillRect(0, 0, this.width, this.height);
      // Quiet coordinate texture makes zoom and panning perceptible.
      const spacing = clamp(32 * k, 18, 60);
      ctx.fillStyle = this.theme.line;
      for (let gx = ((x % spacing) + spacing) % spacing; gx < this.width; gx += spacing) {
        for (let gy = ((y % spacing) + spacing) % spacing; gy < this.height; gy += spacing) ctx.fillRect(gx, gy, 1, 1);
      }
      ctx.translate(x, y); ctx.scale(k, k);
      const selectedNode = this.selection?.kind === 'node' ? this.selection.id : null;
      const selectedEdge = this.selection?.kind === 'edge' ? this.selection.id : null;
      const selectionEdge = selectedEdge && this.edges.find(edge => edge.id === selectedEdge);
      const connected = selectedNode ? this.neighbors.get(selectedNode) : null;
      for (const edge of this.edges) edge.geometry = edgeGeometry(edge);
      const batches = new Map();
      for (const edge of this.edges) {
        const selected = edge.id === selectedEdge;
        const emphasized = selected || edge.source.id === selectedNode || edge.target.id === selectedNode;
        const fraud = Number(edge.label) === 1;
        const key = `${selected ? 2 : emphasized ? 1 : 0}:${fraud}`;
        if (!batches.has(key)) batches.set(key, { edges: [], selected, emphasized, fraud });
        batches.get(key).edges.push(edge);
      }
      const arrow = Math.max(3 / k, 5 / Math.sqrt(k));
      // Batch equal styles into a few Canvas operations, even at the edge limit.
      for (const batch of [...batches.values()].sort((a, b) => Number(a.emphasized) - Number(b.emphasized) || Number(a.selected) - Number(b.selected))) {
        const { selected, emphasized, fraud } = batch;
        ctx.globalAlpha = selected ? 1 : emphasized ? .85 : (selectedNode || selectedEdge) ? .15 : fraud ? .72 : .32;
        ctx.strokeStyle = selected ? this.theme.selected : fraud ? this.theme.fraud : this.theme.edge;
        ctx.fillStyle = ctx.strokeStyle;
        ctx.lineWidth = (selected ? 2.6 : emphasized ? 1.6 : 1) / Math.sqrt(k);
        ctx.beginPath();
        for (const { geometry } of batch.edges) {
          ctx.moveTo(geometry.start.x, geometry.start.y);
          if (geometry.controls.length === 2) ctx.bezierCurveTo(geometry.controls[0].x, geometry.controls[0].y, geometry.controls[1].x, geometry.controls[1].y, geometry.end.x, geometry.end.y);
          else ctx.quadraticCurveTo(geometry.controls[0].x, geometry.controls[0].y, geometry.end.x, geometry.end.y);
        }
        ctx.stroke();
        ctx.beginPath();
        for (const { geometry } of batch.edges) {
          const angle = geometry.angle;
          ctx.moveTo(geometry.end.x, geometry.end.y);
          ctx.lineTo(geometry.end.x - arrow * Math.cos(angle - .5), geometry.end.y - arrow * Math.sin(angle - .5));
          ctx.lineTo(geometry.end.x - arrow * Math.cos(angle + .5), geometry.end.y - arrow * Math.sin(angle + .5));
          ctx.closePath();
        }
        ctx.fill();
      }
      ctx.font = '600 12px system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      ctx.globalAlpha = 1; ctx.fillStyle = this.theme.muted;
      for (const heading of this.columnHeadings || []) ctx.fillText(heading.text, heading.x, heading.y);
      // At low scale or high density only important labels are shown to reduce clutter.
      const labelAll = this.labels && k > .65 && this.nodes.length <= 110;
      for (const node of this.nodes) {
        const selected = node.id === selectedNode;
        const related = connected?.has(node.id) || (selectionEdge && [selectionEdge.source.id, selectionEdge.target.id].includes(node.id));
        const radius = nodeRadius(node), hovered = node.id === this.hoveredId;
        ctx.globalAlpha = !this.selection || selected || related || hovered ? 1 : .32;
        if (selected || hovered) {
          ctx.beginPath(); ctx.arc(node.x, node.y, radius + 5 / Math.sqrt(k), 0, Math.PI * 2);
          ctx.strokeStyle = selected ? this.theme.selected : this.theme.muted;
          ctx.lineWidth = (selected ? 2 : 1) / k; ctx.stroke();
        }
        ctx.beginPath(); ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = colorForType(node.type); ctx.fill();
        ctx.lineWidth = 1.7 / Math.sqrt(k); ctx.strokeStyle = this.theme.panel; ctx.stroke();
        const showLabel = this.labels && (labelAll || selected || hovered || (related && k > .6 && (connected?.size || 0) < 20));
        if (showLabel) {
          const label = String(node.label || node.id), text = label.length > 25 ? label.slice(0, 23) + '…' : label;
          ctx.font = `${selected ? '600' : '400'} ${Math.max(10, Math.min(14, 11 / Math.sqrt(k)))}px system-ui, sans-serif`;
          ctx.textAlign = 'center'; ctx.textBaseline = 'top';
          ctx.strokeStyle = this.theme.background; ctx.lineWidth = 4 / Math.sqrt(k); ctx.lineJoin = 'round';
          ctx.strokeText(text, node.x, node.y + radius + 5 / Math.sqrt(k));
          ctx.fillStyle = this.theme.text; ctx.fillText(text, node.x, node.y + radius + 5 / Math.sqrt(k));
        }
      }
      ctx.globalAlpha = 1;
    }

    _local(event) { const rect = this.canvas.getBoundingClientRect(); return point(event.clientX - rect.left, event.clientY - rect.top); }
    _world(p) { return point((p.x - this.transform.x) / this.transform.k, (p.y - this.transform.y) / this.transform.k); }
    _hitNode(p) {
      const world = this._world(p);
      for (let index = this.nodes.length - 1; index >= 0; index--) {
        const node = this.nodes[index];
        if (distance(world, node) <= nodeRadius(node) + 4 / this.transform.k) return node;
      }
      return null;
    }
    _hitEdge(p) {
      const world = this._world(p), tolerance = 6 / this.transform.k;
      let closest = null, minimum = tolerance;
      for (const edge of this.edges) {
        const geometry = edge.geometry || edgeGeometry(edge), bounds = geometry.bounds;
        if (world.x < bounds.minX - tolerance || world.x > bounds.maxX + tolerance || world.y < bounds.minY - tolerance || world.y > bounds.maxY + tolerance) continue;
        for (let index = 1; index < geometry.samples.length; index++) {
          const value = distanceToSegment(world, geometry.samples[index - 1], geometry.samples[index]);
          if (value < minimum) { minimum = value; closest = edge; }
        }
      }
      return closest;
    }
    _zoomAt(p, factor) {
      if (!Number.isFinite(factor) || factor <= 0) return;
      const world = this._world(p), k = clamp(this.transform.k * factor, .08, 5);
      this.transform = { x: p.x - world.x * k, y: p.y - world.y * k, k };
      this.fitWhenSettled = false;
      this._requestFrame();
    }
    _emitSelection(selection) { this.select(selection); this.callbacks.onSelect?.(selection); }
    _listen(element, type, listener, options) { element.addEventListener(type, listener, options); this.listeners.push([element, type, listener, options]); }

    _bindEvents() {
      const canvas = this.canvas;
      this._listen(canvas, 'pointerdown', event => {
        if (event.button !== 0) return;
        canvas.focus({ preventScroll: true });
        canvas.setPointerCapture(event.pointerId);
        const p = this._local(event);
        this.pointers.set(event.pointerId, p);
        this.fitWhenSettled = false;
        if (this.pointers.size === 1) {
          this.gesture = { origin: p, last: p, node: this._hitNode(p), moved: false };
        } else {
          if (this.gesture?.node) { this.gesture.node.fx = null; this.gesture.node.fy = null; }
          this.gesture = { moved: true, pinch: true };
        }
        canvas.style.cursor = 'grabbing';
      });
      this._listen(canvas, 'pointermove', event => {
        const p = this._local(event);
        if (!this.pointers.has(event.pointerId)) {
          const node = this._hitNode(p), hoveredId = node?.id;
          if (hoveredId !== this.hoveredId) { this.hoveredId = hoveredId; this._requestFrame(); }
          canvas.title = node ? `${node.label || node.id} · ${node.type || 'entity'} · ${node.degree ?? this.neighbors.get(node.id).size} relationships` : '';
          canvas.style.cursor = node ? 'pointer' : 'grab';
          return;
        }
        if (this.pointers.size > 1) {
          const before = [...this.pointers.values()].slice(0, 2);
          this.pointers.set(event.pointerId, p);
          const after = [...this.pointers.values()].slice(0, 2);
          const previousCenter = point((before[0].x + before[1].x) / 2, (before[0].y + before[1].y) / 2);
          const center = point((after[0].x + after[1].x) / 2, (after[0].y + after[1].y) / 2);
          this._zoomAt(previousCenter, distance(after[0], after[1]) / Math.max(1, distance(before[0], before[1])));
          this.transform.x += center.x - previousCenter.x; this.transform.y += center.y - previousCenter.y;
          this._requestFrame();
          return;
        }
        this.pointers.set(event.pointerId, p);
        const gesture = this.gesture;
        if (!gesture || gesture.pinch) return;
        gesture.moved ||= distance(gesture.origin, p) > 4;
        if (gesture.moved) {
          if (gesture.node) {
            const world = this._world(p);
            gesture.node.x = world.x; gesture.node.y = world.y;
            gesture.node.fx = world.x; gesture.node.fy = world.y;
            if (this.simulation) { this.simulation.alpha(Math.max(.15, this.simulation.alpha())); this.ticks = 0; this.layoutDeadline = performance.now() + 900; }
          } else {
            this.transform.x += p.x - gesture.last.x; this.transform.y += p.y - gesture.last.y;
          }
          this._requestFrame();
        }
        gesture.last = p;
      });
      const finish = (event, cancelled = false) => {
        if (!this.pointers.has(event.pointerId)) return;
        this.pointers.delete(event.pointerId);
        const gesture = this.gesture;
        if (gesture?.node) { gesture.node.fx = null; gesture.node.fy = null; }
        if (!cancelled && !gesture?.moved && !gesture?.pinch) {
          const p = this._local(event), node = this._hitNode(p), edge = node ? null : this._hitEdge(p);
          if (node) this._emitSelection({ kind: 'node', id: node.id });
          else if (edge) this._emitSelection({ kind: 'edge', id: edge.id });
          else this._emitSelection(null);
        }
        if (!this.pointers.size) this.gesture = null;
        canvas.style.cursor = 'grab';
      };
      this._listen(canvas, 'pointerup', event => finish(event));
      this._listen(canvas, 'pointercancel', event => finish(event, true));
      this._listen(canvas, 'pointerleave', () => { this.hoveredId = null; this._requestFrame(); });
      this._listen(canvas, 'dblclick', event => {
        const node = this._hitNode(this._local(event));
        if (node) { event.preventDefault(); this._emitSelection({ kind: 'node', id: node.id }); this.callbacks.onExpand?.(node.id); }
      });
      this._listen(canvas, 'wheel', event => {
        event.preventDefault();
        this._zoomAt(this._local(event), Math.exp(-clamp(event.deltaY * (event.deltaMode === 1 ? 16 : 1), -150, 150) * .003));
      }, { passive: false });
      this._listen(canvas, 'keydown', event => {
        if (event.ctrlKey || event.metaKey || event.altKey) return;
        if (event.key === '+' || event.key === '=') this.zoomBy(1.25);
        else if (event.key === '-') this.zoomBy(.8);
        else if (event.key.toLowerCase() === 'f' || event.key === '0') this.fit();
        else if (event.key === 'Escape') this._emitSelection(null);
        else if (event.key === 'Enter' && this.selection?.kind === 'node') this.callbacks.onExpand?.(this.selection.id);
        else if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) {
          this.transform.x += event.key === 'ArrowLeft' ? 40 : event.key === 'ArrowRight' ? -40 : 0;
          this.transform.y += event.key === 'ArrowUp' ? 40 : event.key === 'ArrowDown' ? -40 : 0;
          this.fitWhenSettled = false; this._requestFrame();
        } else return;
        event.preventDefault();
      });
      this._listen(document, 'visibilitychange', () => { if (document.hidden) this._cancelFrame(); else this._requestFrame(); });
    }
  }

  global.DatasetGraphRenderer = DatasetGraphRenderer;
})(globalThis);
