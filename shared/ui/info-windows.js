/* Shared, nonmodal extended help. Metadata is plain text, never executable HTML.
 * attach(host, infoOrResolver, {button, buttonParent, placement, hoverDelay})
 * returns a detach function. Delegated events and weak references allow hosts to
 * be discarded by a renderer without an explicit disposal pass.
 */
(function (global) {
  'use strict';
  if (global.PaymentInfo) return;
  const doc = global.document;
  if (!doc?.head || typeof doc.addEventListener !== 'function') return;
  const hosts = new WeakMap(), buttons = new WeakMap();
  const popupId = 'payment-extended-info';
  let active = null, pending = null, popup = null, pinned = false;
  let pendingPoint = null, activePoint = null;
  let openTimer = null, closeTimer = null, cleanupPosition = null, observer = null;
  let positionRevision = 0;
  const human = value => String(value ?? '').replace(/_/g, ' ').replace(/^./, char => char.toUpperCase());
  const present = value => value !== undefined && value !== null && value !== '';
  const display = value => typeof value === 'object' ? JSON.stringify(value) : String(value);
  const section = (title, text) => present(text) ? {title, text: display(text)} : null;
  const fact = (title, value) => present(value) ? [title, display(value)] : null;
  const statistic = value => typeof value === 'number' && Number.isFinite(value)
    ? (value !== 0 && Math.abs(value) < 1e-5 ? value.toExponential(3) : value.toLocaleString(undefined, {maximumFractionDigits: 5}))
    : '—';

  function mergeInfo(base, extra) {
    if (typeof extra === 'string') extra = {description: extra};
    extra = extra || {};
    const sections = new Map((base.sections || []).filter(Boolean).map(item => [item.title, item]));
    for (const item of extra.sections || []) if (item && present(item.text)) sections.set(item.title, item);
    const facts = new Map((base.facts || []).filter(Boolean));
    for (const item of extra.facts || []) if (Array.isArray(item) && item.length > 1) facts.set(item[0], item[1]);
    return {...base, ...extra, sections: [...sections.values()], facts: [...facts.entries()]};
  }

  function feature(value = {}) {
    if (typeof value === 'string') value = {id: value};
    const id = value.id || value.name;
    const generic = text => !present(text) || /^(?:Numeric input supplied to the model from this dataset|A stored numeric model input|Numeric model input|Original numeric dataset column, preserved without modification|An original numeric column preserved from the dataset)\.?$/i.test(String(text).trim());
    const extra = typeof value.info === 'string' ? {description:value.info} : {...value.info};
    if (generic(extra.description)) delete extra.description;
    // Recover the meaning of a known recipe when an older view only supplies
    // its ID. Never borrow a calculation, history contract, or source meaning.
    const known = value.kind !== 'source' && (global.FraudPaymentFeatures?.featureDefinitions || []).find(item => item.id === id);
    const description = [value.description,value.help].find(text => !generic(text))
      || (known ? `This column describes ${known.label.toLowerCase()}. Its saved definition is needed to confirm the calculation and history used for these values.` : '')
      || `The dataset column “${id || value.label || 'unnamed'}” has no recorded definition of what its values measure. Consult the source schema for its meaning.`;
    const stats = value.statistics;
    return mergeInfo({
      title: value.label || human(value.id || value.name),
      description,
      sections: [
        section('Calculation', value.method), section('Reading the value', value.readout),
        section('Approximation and limits', value.approximation),
        section('Availability', value.unavailable_reason),
      ],
      facts: [
        fact('Feature ID', value.id || value.name), fact('Group', value.group), fact('Source', value.source),
        fact('Units', value.unit || value.units), fact('Transform', value.transform),
        fact('History window', value.window || value.history), fact('Graph orientation', value.orientation),
        fact('Calculation parameters', value.parameters && Object.keys(value.parameters).length ? value.parameters : null),
        fact('Required data', value.requirements?.length ? value.requirements.join(', ') : null),
        fact('Bucket boundaries', value.amount_bins), fact('Recipe version', value.version),
        ...(stats ? [['Observed values', statistic(stats.count)], ['Missing values', statistic(stats.missing)],
          ['Minimum', statistic(stats.min)], ['Maximum', statistic(stats.max)],
          ['Mean', statistic(stats.mean)], ['Standard deviation', statistic(stats.std)]] : []),
      ],
    }, extra);
  }

  function parameter(value = {}) {
    const options = value.options?.map(option => option !== null && typeof option === 'object'
      ? (option.label ? option.label + ' (' + display(option.value) + ')' : display(option.value))
      : option === null ? 'None' : display(option));
    return mergeInfo({
      title: value.label || human(value.name || value.id),
      description: value.description || value.help || 'Configuration value used when preparing or running this model.',
      sections: [section('How to choose', value.guidance)],
      facts: [
        fact('Parameter', value.name || value.id), fact('Type', value.type),
        Object.hasOwn(value, 'default') ? ['Default', value.default_label || (value.default === null ? 'Automatic / not explicitly set' : display(value.default))] : null,
        fact('Minimum', value.min), fact('Maximum', value.max), fact('Step', value.step),
        fact('Options', options?.join(', ')), value.nullable ? ['Optional', 'An empty value uses no explicit setting.'] : null,
      ],
    }, value.info);
  }

  function node(tag, parent, text, className) {
    const element = doc.createElement(tag);
    if (present(text)) element.textContent = display(text);
    if (className) element.className = className;
    if (parent) parent.appendChild(element);
    return element;
  }

  const styles = node('style', doc.head);
  styles.textContent = `
    [data-payment-info] { cursor: help; }
    [data-payment-info]:is(label,summary,button,a,input,select) { cursor: inherit; }
    button.payment-info-trigger, .pl-root button.payment-info-trigger, #xgboost-analytics button.payment-info-trigger {
      display:inline-flex; align-items:center; justify-content:center; flex:none;
      position:relative; box-sizing:border-box; width:16px; height:16px; min-width:16px; min-height:16px;
      padding:0; margin:0 0 0 4px; border:0; border-radius:50%;
      background:transparent; color:var(--pl-quiet,var(--muted-foreground,#62706d));
      font:600 12px/1 system-ui,sans-serif; vertical-align:middle; cursor:help;
      text-transform:none; letter-spacing:normal;
    }
    button.payment-info-trigger::before {
      content:''; position:absolute; box-sizing:border-box; width:14px; height:14px;
      border:1px solid currentColor; border-radius:50%; background:transparent;
    }
    button.payment-info-trigger:hover, button.payment-info-trigger[aria-expanded=true],
    #xgboost-analytics button.payment-info-trigger:is(:hover,[aria-expanded=true]) {
      color:var(--pl-accent,var(--foreground,#166b5d)); background:transparent;
    }
    button.payment-info-trigger:hover::before, button.payment-info-trigger[aria-expanded=true]::before {
      background:color-mix(in srgb,currentColor 10%,transparent);
    }
    button.payment-info-trigger svg { position:relative; display:block; width:10px; height:10px; flex:none; margin:0; }
    .payment-info-trigger:focus-visible, [data-payment-info]:focus-visible,
    .payment-info-window:focus-visible, .payment-info-close:focus-visible {
      outline:2px solid var(--pl-accent,#168373); outline-offset:3px;
    }
    .payment-info-window {
      box-sizing:border-box; position:fixed; inset:0 auto auto 0; z-index:2147483000;
      width:390px; max-width:calc(100vw - 16px); max-height:calc(100dvh - 16px);
      overflow:auto; overscroll-behavior:contain; scrollbar-width:thin;
      padding:18px; border:1px solid light-dark(#cfdbd6,#4d5c59); border-radius:12px;
      background:light-dark(#fff,#202a27); color:light-dark(#23342d,#e6eeea);
      box-shadow:0 10px 36px #0003; text-align:left; white-space:normal;
      font:400 13px/1.55 var(--font-sans,system-ui,sans-serif); overflow-wrap:anywhere;
    }
    .payment-info-window * { box-sizing:border-box; }
    .payment-info-header { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; }
    .payment-info-title { margin:0; font-size:15px; font-weight:650; line-height:1.4; }
    .payment-info-description { margin:10px 0 0; }
    .payment-info-section { margin-top:13px; }
    .payment-info-section h3 { margin:0 0 3px; font-size:12px; font-weight:650; }
    .payment-info-section p { margin:0; white-space:pre-line; }
    .payment-info-facts { display:grid; grid-template-columns:minmax(90px,.7fr) minmax(0,1.3fr); gap:7px 13px; margin:15px 0 0; font-size:11px; }
    .payment-info-facts dt { color:light-dark(#62706b,#b1c2b9); }
    .payment-info-facts dd { margin:0; min-width:0; white-space:pre-line; }
    .payment-info-close {
      display:inline-flex; align-items:center; justify-content:center; flex:none;
      width:26px; height:26px; margin:-4px -5px 0 0; padding:0; border:0; border-radius:5px;
      background:transparent; color:inherit; cursor:pointer; font:400 21px/1 system-ui,sans-serif;
    }
    .payment-info-close:hover { background:light-dark(#eef3ef,#35443c); }
    @media(pointer:coarse) {
      button.payment-info-trigger, .pl-root button.payment-info-trigger, #xgboost-analytics button.payment-info-trigger { width:24px; height:24px; min-width:24px; min-height:24px; }
      .payment-info-close { width:36px; height:36px; }
    }
  `;

  function owner(target) {
    if (target?.nodeType !== 1) return null;
    const match = target.closest('[data-payment-info-button],[data-payment-info]');
    return match ? buttons.get(match) || hosts.get(match) || null : null;
  }
  function contains(record, target) {
    return !!target?.nodeType && (record.host.contains(target) || record.button?.contains(target));
  }
  function describedBy(element, enabled) {
    const ids = new Set((element.getAttribute('aria-describedby') || '').split(/\s+/).filter(Boolean));
    if (enabled) ids.add(popupId); else ids.delete(popupId);
    if (ids.size) element.setAttribute('aria-describedby', [...ids].join(' '));
    else element.removeAttribute('aria-describedby');
  }
  function cancelOpen() {
    clearTimeout(openTimer); openTimer = null; pending = null; pendingPoint = null;
  }
  function cancelClose() {
    clearTimeout(closeTimer); closeTimer = null;
  }
  function close(restoreFocus = false) {
    const previous = active, focusWasInside = popup?.contains(doc.activeElement);
    cancelOpen(); cancelClose(); positionRevision++;
    cleanupPosition?.(); cleanupPosition = null;
    observer?.disconnect(); observer = null;
    if (previous) {
      describedBy(previous.host, false);
      if (previous.button) {
        describedBy(previous.button, false);
        previous.button.setAttribute('aria-expanded', 'false');
      }
    }
    active = null; pinned = false; activePoint = null;
    popup?.remove(); popup = null;
    if (restoreFocus && focusWasInside && previous?.host.isConnected) {
      (previous.button || previous.host).focus({preventScroll: true});
      // Restoring focus must not immediately reopen the window.
      cancelOpen();
    }
  }
  function resolve(record) {
    const value = typeof record.info === 'function' ? record.info() : record.info;
    return typeof value === 'string' ? {title: 'More information', description: value} : value || {};
  }
  function geometry(record) {
    const view = global.visualViewport;
    const width = view?.width || global.innerWidth, height = view?.height || global.innerHeight;
    const left = view?.offsetLeft || 0, top = view?.offsetTop || 0;
    const rect = activePoint ? {x:activePoint.x,y:activePoint.y,left:activePoint.x,right:activePoint.x,
      top:activePoint.y,bottom:activePoint.y,width:0,height:0} : record.host.getBoundingClientRect();
    const gap = activePoint ? 32 : 14, cross = activePoint ? 12 : 0, desired = Math.min(390, width - 16);
    const rightRoom = left + width - 8 - rect.right - gap, leftRoom = rect.left - left - 8 - gap;
    // Preserve a useful reading width. Only narrow viewports fall back above
    // or below the anchor; otherwise keep a clear horizontal strip by it.
    const side = rightRoom >= desired ? 'right' : leftRoom >= desired ? 'left'
      : Math.max(rightRoom, leftRoom) >= 240 ? (rightRoom >= leftRoom ? 'right' : 'left')
      : top + height - rect.bottom >= rect.top - top ? 'bottom' : 'top';
    return {rect,width,height,left,top,gap,cross,side,placement:record.options.placement || side+'-start'};
  }
  function fallbackPosition(record, floating, layout = geometry(record)) {
    const {rect,width,height,left,top,gap,cross,side} = layout;
    const horizontal = side === 'left' || side === 'right';
    const room = side === 'right' ? left+width-8-rect.right-gap : side === 'left' ? rect.left-left-8-gap
      : side === 'bottom' ? top+height-8-rect.bottom-gap : rect.top-top-8-gap;
    floating.style.maxWidth = Math.max(1, horizontal ? Math.min(width-16, room) : width-16) + 'px';
    floating.style.maxHeight = Math.max(1, horizontal ? height-16 : Math.min(height-16, room)) + 'px';
    const bounds = floating.getBoundingClientRect();
    const requestedX = side === 'right' ? rect.right+gap : side === 'left' ? rect.left-gap-bounds.width : rect.left+cross;
    const requestedY = side === 'bottom' ? rect.bottom+gap : side === 'top' ? rect.top-gap-bounds.height : rect.top+cross;
    const x = Math.max(left+8, Math.min(requestedX, left+width-bounds.width-8));
    const y = Math.max(top+8, Math.min(requestedY, top+height-bounds.height-8));
    floating.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
    floating.style.visibility = 'visible';
  }
  function position(record, floating) {
    // The vendor scripts can load after tools; resolve them when first opened.
    const api = global.FloatingUIDOM;
    const update = () => {
      if (!record.host.isConnected) { close(); return; }
      let layout = geometry(record);
      if (activePoint && (activePoint.x < layout.left || activePoint.x > layout.left+layout.width
          || activePoint.y < layout.top || activePoint.y > layout.top+layout.height)) {
        if (!pinned) { close(); return; }
        // A frozen mouse position can leave a resized or zoomed viewport.
        // Keep pinned help readable by returning to its actual field.
        activePoint = null; layout = geometry(record);
      }
      const {rect,gap,cross,placement} = layout;
      const hostRect = record.host.getBoundingClientRect();
      if (activePoint && !pinned && (hostRect.bottom < layout.top || hostRect.top > layout.top+layout.height
          || hostRect.right < layout.left || hostRect.left > layout.left+layout.width)) { close(); return; }
      fallbackPosition(record, floating, layout);
      if (!api) return;
      const revision = ++positionRevision;
      const reference = activePoint ? {contextElement:record.host,getBoundingClientRect:()=>rect} : record.host;
      void api.computePosition(reference, floating, {
        placement, strategy: 'fixed',
        middleware: [api.offset({mainAxis:gap,crossAxis:cross}), api.shift({padding: 8}),
          api.size({padding: 8, apply({availableWidth, availableHeight, elements}) {
            elements.floating.style.maxWidth = Math.max(0, availableWidth) + 'px';
            elements.floating.style.maxHeight = Math.max(0, availableHeight) + 'px';
          }})],
      }).then(({x, y}) => {
        if (active !== record || revision !== positionRevision) return;
        floating.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
      }).catch(() => { if (active === record) fallbackPosition(record, floating); });
    };
    if (api?.autoUpdate) cleanupPosition = api.autoUpdate(record.host, floating, update);
    else {
      update(); global.addEventListener('resize', update); doc.addEventListener('scroll', update, true);
      global.visualViewport?.addEventListener('resize', update); global.visualViewport?.addEventListener('scroll', update);
      cleanupPosition = () => {
        global.removeEventListener('resize', update); doc.removeEventListener('scroll', update, true);
        global.visualViewport?.removeEventListener('resize', update); global.visualViewport?.removeEventListener('scroll', update);
      };
    }
  }
  function open(record, pin = false, focus = false, point = null) {
    if (!record.host.isConnected) return;
    cancelOpen(); cancelClose();
    if (active === record) {
      pinned = pinned || pin;
      if (!!activePoint !== !!point) {
        activePoint = point; cleanupPosition?.(); cleanupPosition = null; position(record, popup);
      }
      if (focus) popup.focus({preventScroll: true});
      return;
    }
    close(); active = record; pinned = pin; activePoint = point;
    const info = resolve(record);
    popup = node('aside', doc.body, null, 'payment-info-window');
    popup.id = popupId; popup.tabIndex = -1;
    popup.setAttribute('role', 'dialog'); popup.setAttribute('aria-modal', 'false');
    popup.setAttribute('aria-labelledby', popupId + '-title');
    const header = node('div', popup, null, 'payment-info-header');
    node('h2', header, info.title || 'More information', 'payment-info-title').id = popupId + '-title';
    const dismiss = node('button', header, '×', 'payment-info-close');
    dismiss.type = 'button'; dismiss.setAttribute('aria-label', 'Close information');
    dismiss.addEventListener('click', () => close(true));
    if (present(info.description)) node('p', popup, info.description, 'payment-info-description');
    for (const item of info.sections || []) {
      if (!item || !present(item.text)) continue;
      const block = node('section', popup, null, 'payment-info-section');
      if (present(item.title)) node('h3', block, item.title);
      node('p', block, item.text);
    }
    const entries = (info.facts || []).filter(item => Array.isArray(item) && present(item[1]));
    if (entries.length) {
      const list = node('dl', popup, null, 'payment-info-facts');
      for (const [label, value] of entries) { node('dt', list, label); node('dd', list, value); }
    }
    describedBy(record.host, true);
    if (record.button) { describedBy(record.button, true); record.button.setAttribute('aria-expanded', 'true'); }
    position(record, popup);
    observer = new MutationObserver(() => { if (active && !active.host.isConnected) close(); });
    observer.observe(doc.body, {childList: true, subtree: true});
    if (focus) popup.focus({preventScroll: true});
  }
  function requestOpen(record, immediate = false, point = null) {
    cancelClose();
    if (active === record) { if (immediate) open(record, false, false); return; }
    if (pinned || (pending === record && !immediate)) return;
    cancelOpen(); pending = record; pendingPoint = point;
    openTimer = setTimeout(() => {
      const point = pendingPoint; pending = null; pendingPoint = null; openTimer = null;
      if (point && owner(doc.elementFromPoint(point.x,point.y)) !== record) return;
      open(record, false, false, point);
    }, immediate ? 0 : record.options.hoverDelay ?? 350);
  }
  function requestClose() {
    cancelOpen();
    if (!active || pinned) return;
    cancelClose();
    closeTimer = setTimeout(() => {
      if (!active || pinned || contains(active, doc.activeElement) || popup?.contains(doc.activeElement)) return;
      close();
    }, 220);
  }

  function placeButton(record) {
    const {host, button, options} = record;
    if (!button) return;
    button.setAttribute('aria-label', 'More about ' + (resolve(record).title || 'this value'));
    const requested = options.buttonParent || options.buttonContainer;
    const container = requested?.appendChild && (requested.isConnected || host.contains(requested)) ? requested : null;
    if (container) {
      if (button.parentNode !== container) container.appendChild(button);
    } else if (host.matches('input,select,textarea,button,a')) {
      if (host.nextSibling !== button) host.after(button);
    } else if (button.parentNode !== host) host.appendChild(button);
  }

  function suppressSvgTitles(record) {
    const {host} = record;
    if (host.namespaceURI !== 'http://www.w3.org/2000/svg') return;
    record.svgTitles = (record.svgTitles || []).filter(item => item.placeholder.parentNode === host);
    const titles = [...host.children].filter(child => child.localName === 'title');
    if (!titles.length) return;
    // SVG <title> has its own browser tooltip. A nonrendered <desc> preserves
    // its text and any aria-labelledby reference while rich help is attached.
    const name = titles.map(title => title.textContent.trim()).filter(Boolean).join(' ');
    if (name && !host.hasAttribute('aria-label') && !host.hasAttribute('aria-labelledby')) host.setAttribute('aria-label', name);
    for (const title of titles) {
      const placeholder = doc.createElementNS(host.namespaceURI, 'desc');
      placeholder.textContent = title.textContent;
      if (title.hasAttribute('id')) placeholder.id = title.id;
      host.replaceChild(placeholder, title);
      record.svgTitles.push({title, placeholder});
    }
  }

  function attach(host, info, options = {}) {
    if (!host || host.nodeType !== 1) return () => {};
    const existing = hosts.get(host);
    if (existing) {
      existing.info = info; existing.options = {...existing.options, ...options};
      placeButton(existing);
      suppressSvgTitles(existing);
      if (active === existing) { const wasPinned = pinned, point = activePoint; close(); open(existing, wasPinned, false, point); }
      return existing.detach;
    }
    const record = {host, info, options, button: null};
    const original = Object.fromEntries(['title', 'data-tooltip', 'data-payment-info', 'tabindex', 'aria-label'].map(name => [name, host.getAttribute(name)]));
    host.removeAttribute('title');
    // An empty nearest data-tooltip masks the shell's brief legacy tooltip.
    host.setAttribute('data-tooltip', ''); host.setAttribute('data-payment-info', '');
    const svg = host.namespaceURI === 'http://www.w3.org/2000/svg';
    suppressSvgTitles(record);
    if (options.button !== false && !svg) {
      const button = node('button', null, null, 'payment-info-trigger');
      const icon = doc.createElementNS('http://www.w3.org/2000/svg', 'svg');
      icon.setAttribute('viewBox', '0 0 16 16'); icon.setAttribute('aria-hidden', 'true');
      const mark = doc.createElementNS(icon.namespaceURI, 'path');
      mark.setAttribute('d', 'M8 7v5M8 4v.2'); mark.setAttribute('stroke', 'currentColor');
      mark.setAttribute('stroke-width', '1.8'); mark.setAttribute('stroke-linecap', 'round');
      icon.appendChild(mark); button.appendChild(icon);
      button.type = 'button'; button.setAttribute('data-payment-info-button', '');
      button.setAttribute('aria-haspopup', 'dialog'); button.setAttribute('aria-expanded', 'false');
      button.setAttribute('aria-controls', popupId); button.setAttribute('data-tooltip', '');
      record.button = button; buttons.set(button, record);
      placeButton(record);
    } else if (!host.matches('a[href],button,input,select,textarea,summary,[tabindex]') && !host.querySelector('a[href],button,input,select,textarea,summary,[tabindex]')) {
      host.tabIndex = 0;
    }
    record.detach = () => {
      if (active === record) close();
      if (pending === record) cancelOpen();
      record.button?.remove(); hosts.delete(host);
      for (const {title, placeholder} of record.svgTitles || []) {
        if (placeholder.parentNode === host) host.replaceChild(title, placeholder);
      }
      for (const [name, value] of Object.entries(original)) {
        if (value === null) host.removeAttribute(name); else host.setAttribute(name, value);
      }
    };
    hosts.set(host, record);
    return record.detach;
  }

  doc.addEventListener('pointerover', event => {
    if (event.pointerType === 'touch') return;
    if (popup?.contains(event.target)) { cancelOpen(); cancelClose(); return; }
    const record = owner(event.target);
    if (record && !contains(record, event.relatedTarget)) requestOpen(record, false, {x:event.clientX,y:event.clientY});
  });
  doc.addEventListener('pointermove', event => {
    if (event.pointerType !== 'touch' && pendingPoint && pending && owner(event.target) === pending)
      pendingPoint = {x:event.clientX,y:event.clientY};
  });
  doc.addEventListener('pointerout', event => {
    if (event.pointerType === 'touch') return;
    if (popup?.contains(event.relatedTarget)) { cancelOpen(); cancelClose(); return; }
    const record = owner(event.target);
    if (record && contains(record, event.relatedTarget)) return;
    if (active && contains(active, event.relatedTarget)) { cancelClose(); return; }
    if (record || popup?.contains(event.target)) requestClose();
  });
  doc.addEventListener('focusin', event => {
    if (popup?.contains(event.target)) { cancelOpen(); cancelClose(); return; }
    const record = owner(event.target);
    if (active && active !== record) close();
    if (record && event.target.matches(':focus-visible')) requestOpen(record, true);
  });
  doc.addEventListener('focusout', event => {
    if (active && (contains(active, event.relatedTarget) || popup?.contains(event.relatedTarget))) return;
    if (owner(event.target) || popup?.contains(event.target)) requestClose();
  });
  doc.addEventListener('pointerdown', event => {
    if (active && !contains(active, event.target) && !popup?.contains(event.target)) close();
  }, true);
  // Capture prevents a nested info button from toggling its label, summary,
  // sortable header, or a parent control's application click handler.
  doc.addEventListener('click', event => {
    const button = event.target.closest?.('[data-payment-info-button]');
    const record = button && buttons.get(button);
    if (!record) return;
    event.preventDefault(); event.stopPropagation();
    if (active === record && pinned) close();
    else open(record, true, event.detail === 0, event.detail > 0 && event.pointerType !== 'touch'
      ? {x:event.clientX,y:event.clientY} : null);
  }, true);
  doc.addEventListener('keydown', event => {
    if (event.key === 'Escape' && (active || pending)) {
      event.preventDefault(); event.stopPropagation(); close(true);
    } else if (event.key === 'ArrowDown' && owner(event.target)?.button === event.target) {
      event.preventDefault(); open(owner(event.target), true, true);
    }
  }, true);
  global.addEventListener('pagehide', () => close());
  global.PaymentInfo = {attach, feature, parameter, close};
})(globalThis);
