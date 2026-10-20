/* Evals pages. The scores are the scored STEER-ME paper runs in static/evals_results.json
   (written by steer-scoring site-export): benchmark -> element -> model -> format ->
   [norm, exact, ece, brier, epa, n], plus domain / perspective / difficulty breakdowns of the
   multiple-choice format. Anything that was not run or not scored shows as empty, never a guess.
   Each level (both benchmarks, a setting, a module, an element) is its own page; the metric is
   kept in the query string and an open breakdown in the hash, so Back and swipe work as expected. */
(() => {
  const D = JSON.parse(document.getElementById('evals-data').textContent);
  const W = JSON.parse(document.getElementById('evals-where').textContent);
  const out = document.getElementById('ev-out'), metricsEl = document.getElementById('ev-metrics');
  const detail = document.getElementById('ev-detail'), scrim = document.getElementById('ev-scrim');
  if (!out) return;

  /* ---------- scored results ---------- */
  let R = null;
  const MI = { norm: 0, exact: 1, ece: 2, brier: 3, epa: 4 };
  const CALIBRATION = ['ece', 'brier', 'epa'];
  const IDX = Object.fromEntries(D.models.map((m, i) => [m.name, String(i)]));
  /* one element's value for a model: format 'mc' may be sliced by one of domain / perspective / difficulty */
  function value(m, bench, el, metric, format, slice) {
    const e = R && R.benchmarks[bench] && R.benchmarks[bench][el] && R.benchmarks[bench][el][IDX[m.name]];
    if (!e) return null;
    /* difficulty slices are relative levels (1 = the element's easiest); D.difficulty_keys maps them to the run's keys */
    const key = slice && slice[0] === 'difficulty' ? ((D.difficulty_keys[bench] || {})[el] || {})[slice[1]] : slice && slice[1];
    if (slice && key == null) return null;
    const v = slice ? (format === 'mc' && e.by && e.by[slice[0]] ? e.by[slice[0]][key] : null) : e[format];
    return v ? v[MI[metric]] : null;
  }
  /* how many questions one element's score rests on (the last entry of each score list) */
  function questions(m, bench, el, s) {
    const e = R && R.benchmarks[bench] && R.benchmarks[bench][el] && R.benchmarks[bench][el][IDX[m.name]];
    const v = e && !['domain', 'perspective', 'difficulty'].some(k => s[k]) ? e[s.format] : null;
    return v ? v[5] : null;
  }
  /* how many of those questions got no usable answer (counted as wrong); the release runs record the share */
  function unanswered(m, bench, el, s) {
    const e = R && R.benchmarks[bench] && R.benchmarks[bench][el] && R.benchmarks[bench][el][IDX[m.name]];
    const v = e && !['domain', 'perspective', 'difficulty'].some(k => s[k]) ? e[s.format] : null;
    return v && v[6] != null ? v[6] * v[5] : 0;
  }
  const mean = xs => { const v = xs.filter(x => x != null); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
  const elScore = (m, bench, el, s) => {
    const slice = ['domain', 'perspective', 'difficulty'].find(k => s[k]);
    return value(m, bench, el, s.metric, s.format, slice ? [slice, s[slice]] : null);
  };
  const METRIC = Object.fromEntries(D.metrics.map(m => [m.key, m]));
  const FORMAT = Object.fromEntries(D.formats.map(f => [f.key, f]));

  /* ---------- state: metric in the query, open breakdown in the hash ---------- */
  const q = new URLSearchParams(location.search);
  let saved = null, savedF = null; try { saved = localStorage.getItem('steer-metric'); savedF = localStorage.getItem('steer-format'); } catch (e) {}
  const st = { metric: METRIC[q.get('metric')] ? q.get('metric') : (METRIC[saved] ? saved : 'norm'), format: ['mc', 'free'].includes(q.get('format')) ? q.get('format') : (savedF === 'free' ? 'free' : 'mc'), domain: '', perspective: '', difficulty: '' };
  const withMetric = href => { const qs = new URLSearchParams(); if (st.metric !== 'norm') qs.set('metric', st.metric); if (st.format !== 'mc') qs.set('format', st.format); return href + (qs.toString() ? '?' + qs : ''); };
  const slug = t => t.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  const B = Object.fromEntries(D.benchmarks.map(b => [b.key, b]));
  const elementsOf = (b, scope) => {
    const res = [];
    b.settings.forEach(s => s.modules.forEach(m => m.elements.forEach(e => {
      if (!scope || scope === 's:' + s.number || scope === 'm:' + m.number || scope === 'e:' + e.id) res.push({ ...e, setting: s, module: m });
    })));
    return res;
  };
  const fmt = v => v == null ? '–' : v.toFixed(2);
  function tone(v) {
    if (v == null) return { cls: 'na', t: 0 };
    const m = METRIC[st.metric]; let z;
    if (st.metric === 'norm') z = v; else if (m.better === 'higher') z = (v - 0.25) / 0.75; else z = 1 - v / 0.3;
    return { cls: z < 0 ? 'neg' : 'pos', t: Math.min(1, Math.abs(z)) };
  }
  const noProbs = (m, metric = st.metric) => CALIBRATION.includes(metric) && !m.logprobs;
  // why a model has no option probabilities (see the Metrics page, #calibration)
  const whyNoProbs = m => !m.open ? 'the provider does not return them'
    : m.reasoning ? 'a reasoning model: its answer comes after its reasoning, so none is recorded for the answer letter'
    : 'its run did not record them';
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');

  /* ---------- the columns for this page ---------- */
  function columns() {
    if (W.level === 'root') return [st.mbench].map(k => B[k]).map(b => ({ name: b.key, bench: b, cols: b.settings.map(s => ({
      label: s.short, title: s.name, color: s.color, href: `/evals/${b.slug}/${slug(s.name)}`, bench: b, scope: 's:' + s.number })) }));
    const b = B[W.bench], s = b.settings.find(x => x.number === W.setting);
    if (W.level === 'setting') return [{ name: '', bench: b, cols: s.modules.map(m => ({ label: m.name, color: s.color, href: `${W.base}/${slug(m.name)}`, bench: b, scope: 'm:' + m.number })) }];
    const m = s.modules.find(x => x.number === W.module);
    return [{ name: '', bench: b, cols: m.elements.map(e => ({ label: e.name, color: s.color, info: '/elements/' + e.id, bench: b, scope: 'e:' + e.id })) }];
  }
  /* a column's score: the mean over its elements that have a value (k of n elements) */
  function scoreInfo(m, c, over = {}) {
    const els = elementsOf(c.bench, c.scope), s = { ...st, ...(c.format ? { format: c.format } : {}), ...over };
    const vals = els.map(e => elScore(m, c.bench.key, e.id, s)).filter(x => x != null);
    const nq = els.reduce((t, e) => t + (questions(m, c.bench.key, e.id, s) || 0), 0);
    const nu = els.reduce((t, e) => t + unanswered(m, c.bench.key, e.id, s), 0);
    return { v: mean(vals), k: vals.length, n: els.length, nq, nu };
  }
  const score = (m, c, over) => scoreInfo(m, c, over).v;

  /* the overview shows one benchmark at a time: their models are mostly different, so side by side would be mostly empty */
  try { st.mbench = localStorage.getItem('steer-mbench') === 'STEER' ? 'STEER' : 'STEER-ME'; } catch (e) { st.mbench = 'STEER-ME'; }
  if (B[q.get('bench')]) st.mbench = q.get('bench');
  function renderMetrics() {
    const opt = (v, t, cur, dis) => `<option value="${v}"${v === cur ? ' selected' : ''}${dis ? ' disabled' : ''}>${t}</option>`;
    const benchToggle = W.level === 'root' ? `<span class="seg ev-bench" role="group" aria-label="Benchmark">${['STEER-ME', 'STEER'].map(k => `<button type="button" data-mbench="${k}" aria-pressed="${st.mbench === k}">${k}</button>`).join('')}</span>` : '';
    const formatSwitch = `<span class="seg" role="group" aria-label="Question format">${D.formats.map(f => `<button type="button" data-format="${f.key}" aria-pressed="${st.format === f.key}"${f.scored ? '' : ' title="Not scored yet"'}>${f.label}</button>`).join('')}</span>`;
    if (st.format === 'free') {
      metricsEl.innerHTML = benchToggle + formatSwitch + `<span class="ev-metric-fixed">Metric: Accuracy</span>`;
      return;
    }
    metricsEl.innerHTML = benchToggle + formatSwitch + `<label>Metric <select data-k="metric">${D.metrics.map(m => opt(m.key, m.short, st.metric)).join('')}</select></label>`;
    metricsEl.querySelector('[data-k=metric]').title = METRIC[st.metric].label;
  }
  metricsEl.addEventListener('click', e => {
    const f = e.target.closest('button[data-format]'); if (f) { setControl('format', f.dataset.format); return; }
    const b = e.target.closest('button[data-mbench]'); if (!b) return;
    st.mbench = b.dataset.mbench; try { localStorage.setItem('steer-mbench', st.mbench); } catch (err) {}
    const qs = new URLSearchParams(location.search); qs.set('bench', st.mbench);
    history.replaceState(null, '', location.pathname + '?' + qs + location.hash);
    renderMetrics(); render();
  });
  metricsEl.addEventListener('change', e => { const k = e.target.dataset.k; if (k) setControl(k, e.target.value); });
  function setControl(k, v) {
    st[k] = v;
    if (st.format === 'free' && !['norm', 'exact'].includes(st.metric)) st.metric = 'norm';
    try { localStorage.setItem('steer-' + k, st[k]); } catch (err) {}
    const qs = new URLSearchParams(); if (st.metric !== 'norm') qs.set('metric', st.metric); if (st.format !== 'mc') qs.set('format', st.format);
    if (W.level === 'root' && new URLSearchParams(location.search).has('bench')) qs.set('bench', st.mbench);
    history.replaceState(null, '', location.pathname + (qs.toString() ? '?' + qs : '') + location.hash);
    renderMetrics(); render();
  }

  const empty = html => `<div class="ev-empty"><p>${html}</p></div>`;
  function render() {
    if (!R) return;
    const groups = columns(), cols = groups.flatMap(g => g.cols), narrow = cols.length > 9;
    if (!FORMAT[st.format].scored) {
      out.innerHTML = empty(`<strong>Not run yet.</strong> Free-text answers have not been scored, so there are no free-text results. Switch to <em>Multiple choice</em> to see the scored results.`);
      renderDetail(); return;
    }
    if (groups.every(g => !g.bench.scored)) {
      out.innerHTML = empty(`<strong>Not run yet.</strong> There are no scored ${groups[0].bench.key} results; only STEER-ME has been scored so far.${W.level === 'root' ? '' : ' The STEER-ME results are on the <a href="/evals">Evals overview</a>.'}`);
      renderDetail(); return;
    }
    const named = groups.length > 1;
    const head = `<thead>${named ? `<tr class="ev-band"><th></th>${groups.map(g => `<th colspan="${g.cols.length}"${g.bench.scored ? '' : ' title="No scored results for this benchmark yet"'}>${g.name}${g.bench.scored ? '' : ' · not run yet'}</th>`).join('')}</tr>` : ''}
      <tr><th></th>${cols.map(c => `<th scope="col" class="c-${c.color}" title="${c.title || c.label}">${c.href ? `<a href="${withMetric(c.href)}"><span>${c.label}</span></a>` : `<span>${c.label}${c.info ? ` <a class="info" href="${c.info}" title="About ${c.label}" aria-label="About ${c.label}">i</a>` : ''}</span>`}</th>`).join('')}</tr></thead>`;
    const bench = W.level === 'root' ? st.mbench : W.bench;
    /* open models appear once their release run of this benchmark is (nearly) complete; closed (frontier) models
       always, however limited their runs */
    const listed = m => !m.open || (m.listed && m.listed[bench]);
    const info = new Map(D.models.map(m => [m.name, cols.map(c => scoreInfo(m, c))]));
    const has = m => info.get(m.name).some(x => x.v != null);
    const shown = D.models.filter(m => has(m) && listed(m));
    const rowsFor = list => list.map(m => `<tr><th scope="row"><span class="ev-model" title="${esc(m.name)}">${esc(m.short || m.name).replace(/-/g, '-<wbr>')}</span>${m.reasoning ? '<span class="tag">reasoning</span>' : ''}</th>${cols.map((c, i) => {
      const { v } = info.get(m.name)[i], { cls, t } = tone(v);
      const why = v != null ? '' : ` title="${!c.bench.scored || !noProbs(m) ? 'Not run yet' : 'No option probabilities: ' + whyNoProbs(m)}"`;
      return `<td class="${cls}" style="--v:${t.toFixed(3)}" data-model="${esc(m.name)}" data-col="${i}" tabindex="0"${why}>${fmt(v)}</td>`; }).join('')}</tr>`).join('');
    const group = (label, list) => list.length ? `<tr class="ev-group"><th colspan="${cols.length + 1}">${label}</th></tr>${rowsFor(list)}` : '';
    out.innerHTML = `<div class="heat-wrap"><table class="heat${narrow ? ' narrow' : ''}${cols.length > 4 ? ' many' : ''}">${head}<tbody>${
      group('Closed models', shown.filter(m => !m.open))}${group('Open-weight models', shown.filter(m => m.open))}</tbody></table></div>`;
    renderDetail();
  }
  out.addEventListener('click', e => {
    const td = e.target.closest('td[data-model]'); if (!td) return;
    location.hash = 'm=' + encodeURIComponent(td.dataset.model) + '&c=' + td.dataset.col;
  });
  out.addEventListener('keydown', e => { if (e.key === 'Enter' && e.target.matches('td[data-model]')) e.target.click(); });

  /* ---------- breakdown for the cell named in the hash ---------- */
  function bars(title, rows, worst) {
    const vals = rows.map(r => r.v).filter(v => v != null);
    /* a fixed range, so a bar's length is the score itself and panels and models compare: 0 to 1, widened only
       for a score outside it; bars start at zero, so a negative (worse than guessing) one runs left of the zero line */
    const lo = Math.min(0, ...vals), hi = Math.max(1, ...vals), span = hi - lo, zero = (-lo / span) * 100;
    const li = r => { const { cls } = tone(r.v);
      const left = r.v == null ? 0 : ((Math.min(0, r.v) - lo) / span) * 100, w = r.v == null ? 0 : (Math.abs(r.v) / span) * 100;
      const axis = lo < 0 ? ` style="--zero:${zero.toFixed(1)}%"` : '';
      const track = r.v == null ? `<span class="bt-empty">${r.empty || 'not run yet'}</span>` : `<span class="bt${lo < 0 ? ' has-zero' : ''}"${axis}><i class="${cls}${r.v < 0 ? ' neg' : ''}" style="margin-left:${left.toFixed(1)}%;width:${w.toFixed(1)}%"></i></span>`;
      return `<li><span class="bl">${r.href ? `<a href="${r.href}">${r.label}</a>` : r.label}</span>${track}<span class="bv">${fmt(r.v)}</span></li>`; };
    const h = `<h3>${title}${worst != null ? `<span class="worst">worst ${fmt(worst)}</span>` : ''}</h3>`;
    if (rows.length > 8) {
      const better = METRIC[st.metric].better === 'higher';
      const sorted = rows.filter(r => r.v != null).sort((a, b) => better ? a.v - b.v : b.v - a.v);
      return `<div class="bd">${h}<ul>${sorted.slice(0, 5).map(li).join('')}</ul><details><summary>All ${rows.length} ${title.toLowerCase()}s</summary><ul>${rows.map(li).join('')}</ul></details></div>`;
    }
    return `<div class="bd">${h}<ul>${rows.map(li).join('')}</ul></div>`;
  }
  function renderDetail() {
    const h = new URLSearchParams(location.hash.slice(1));
    const m = D.models.find(x => x.name === h.get('m')), c = columns().flatMap(g => g.cols)[+h.get('c')];
    if (!R || !m || !c) { detail.hidden = true; scrim.hidden = true; return; }
    const worstOf = rows => { const v = rows.map(r => r.v).filter(x => x != null); return v.length ? (METRIC[st.metric].better === 'higher' ? Math.min(...v) : Math.max(...v)) : null; };
    const gone = noProbs(m) ? 'no probabilities' : 'not run yet';
    const els = elementsOf(c.bench, c.scope);
    const inner = c.scope.startsWith('s:') ? els[0].setting.modules.map(x => ({ label: x.name, v: score(m, { ...c, scope: 'm:' + x.number }), empty: gone }))
      : c.scope.startsWith('m:') ? els.map(e => ({ label: e.name, v: score(m, { ...c, scope: 'e:' + e.id }), empty: gone })) : null;
    const some = rows => rows.filter(r => r.v != null);
    const doms = some(D.domains.map(d => ({ label: d.label, v: score(m, c, { domain: d.key }) })));
    const pers = some(D.perspectives.map(p => ({ label: p.label, v: score(m, c, { perspective: p.key }) })));
    const diff = some(D.difficulties.map(d => ({ label: 'Level ' + d, v: score(m, c, { difficulty: d }) })));
    const forms = st.format === 'free' ? null : D.variants.map(x => ({ label: x.label, v: score(m, { ...c, format: x.key }), empty: gone }));
    const bench = c.bench.key, { k, n, nq, nu } = scoreInfo(m, c), u = nq ? nu / nq : 0;
    const row = (label, value, note) => `<dt>${label}</dt><dd>${value}${note ? `<small>${note}</small>` : ''}</dd>`;
    const overview = [
      row(st.format === 'free' ? 'Accuracy' : METRIC[st.metric].label, `<b>${fmt(score(m, c))}</b>`),
      row('Evaluated', `${k} of ${n} element${n === 1 ? '' : 's'}${nq ? `, ${nq.toLocaleString()} questions` : ''}`),
      noProbs(m) ? row('Calibration', 'Not available', `${esc(whyNoProbs(m))} (<a href="/metrics#calibration">why</a>)`) : '',
      u >= 0.05 ? row('No usable answer', `${Math.round(100 * u)}% of questions`, 'Cut off by a length limit, or out of context; counted as wrong') : '',
      m.open ? row('Model', `<code>${esc(m.name)}</code>`) : m.model && m.model !== m.name ? row('Snapshot', `<code>${esc(m.model)}</code>`) : ''].join('');
    detail.hidden = false; scrim.hidden = false;
    detail.innerHTML = `<header><div><h2>${esc(m.short || m.name)}</h2><p class="scope">${c.title || c.label}</p></div>
        <button type="button" class="close" aria-label="Close">×</button></header>
      <div class="bd"><h3>Overview</h3><dl class="bd-ov">${overview}</dl></div>
      <div class="bd-grid">${inner ? bars(c.scope.startsWith('s:') ? 'Modules' : 'Elements', inner) : ''}${forms ? bars('Multiple-choice variant', forms) : ''}${doms.length ? bars('Domain', doms, worstOf(doms)) : ''}${pers.length ? bars('Perspective', pers, worstOf(pers)) : ''}${diff.length ? bars('Difficulty (ascending)', diff) : ''}</div>`;
    detail.querySelector('.close').focus({ preventScroll: true });
  }
  const closeDetail = () => { if (location.hash) history.back(); };
  detail.addEventListener('click', e => { if (e.target.closest('.close')) closeDetail(); });
  scrim.addEventListener('click', closeDetail);
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && !detail.hidden) closeDetail(); });
  window.addEventListener('hashchange', renderDetail);

  renderMetrics();
  out.innerHTML = '<p class="ev-foot">Loading results…</p>';
  fetch(out.dataset.src).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(j => { R = j; render(); })
    .catch(() => { out.innerHTML = empty('The results could not be loaded. Reload the page to try again.'); });
})();
