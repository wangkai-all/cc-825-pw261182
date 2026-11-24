'use strict';

const $ = (id) => document.getElementById(id);

const state = {
  dataset: null,
  validation: null,
  analysis: null,
  pareto: null,
  exact: null,
  recovery: null,
  selectedSlot: 0,
  selectedDimension: 0,
  assignmentQuery: '',
};

const CHART_COLORS = {
  blue: '#2a78d6',
  orange: '#eb6834',
  green: '#1baf7a',
  gridLight: '#e1e0d9',
  axisLight: '#c3c2b7',
  text: '#52514e',
};

function fmt(x, digits = 2) {
  if (x === null || x === undefined || !Number.isFinite(Number(x))) return '—';
  return Number(x).toLocaleString('zh-CN', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
function fmtInt(x) {
  if (!Number.isFinite(Number(x))) return '—';
  return Math.round(Number(x)).toLocaleString('zh-CN');
}
function fmtProb(x) {
  if (!Number.isFinite(Number(x))) return '—';
  if (x !== 0 && Math.abs(x) < 0.001) return Number(x).toExponential(2);
  return fmt(x, 4);
}
function fmtMoney(x) { return fmt(x, 1); }
function pct(x, digits = 2) { return Number.isFinite(x) ? `${fmt(x * 100, digits)}%` : '—'; }
function signed(x, digits = 1) { return Number.isFinite(x) ? `${x >= 0 ? '+' : ''}${fmt(x, digits)}` : '—'; }
function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[ch]));
}

async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  };
  const res = await fetch(path, opts);
  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch { data = { error: text }; }
  if (!res.ok) {
    const err = new Error(data.error || `HTTP ${res.status}`);
    err.data = data;
    throw err;
  }
  return data;
}

let toastTimer;
function toast(message, isError = false) {
  const el = $('toast');
  el.hidden = false;
  el.textContent = message;
  el.style.background = isError ? 'var(--red)' : 'var(--text)';
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 5200);
}

function setStatus(text, kind = '') {
  const el = $('runStatus');
  el.className = `status-pill ${kind}`;
  el.textContent = text;
}

function showTooltip(event, html) {
  const tt = $('tooltip');
  tt.innerHTML = html;
  tt.hidden = false;
  const pad = 14;
  const rect = tt.getBoundingClientRect();
  let x = event.clientX + pad;
  let y = event.clientY + pad;
  if (x + rect.width > window.innerWidth - 8) x = event.clientX - rect.width - pad;
  if (y + rect.height > window.innerHeight - 8) y = event.clientY - rect.height - pad;
  tt.style.left = `${Math.max(8, x)}px`;
  tt.style.top = `${Math.max(8, y)}px`;
}
function hideTooltip() { $('tooltip').hidden = true; }

document.addEventListener('mouseover', (e) => {
  const target = e.target.closest('[data-tooltip]');
  if (target) showTooltip(e, target.getAttribute('data-tooltip'));
});
document.addEventListener('mousemove', (e) => {
  const target = e.target.closest('[data-tooltip]');
  const tt = $('tooltip');
  if (target && !tt.hidden) showTooltip(e, target.getAttribute('data-tooltip'));
});
document.addEventListener('mouseout', (e) => {
  if (e.target.closest('[data-tooltip]')) hideTooltip();
});

/* ----------------------------- Data loading ----------------------------- */

async function loadSample(size) {
  try {
    setStatus('载入示例…', 'running');
    const payload = await api(`/api/sample?size=${size}`, undefined);
    await validateAndStore(payload, false);
  } catch (err) {
    toast(`示例载入失败：${err.message}`, true);
    setStatus('载入失败', 'error');
  }
}

async function loadToy() {
  try {
    const payload = await api('/api/sample?toy=1', undefined);
    $('kInput').value = 3;
    $('kValue').textContent = '3';
    $('tailGuardsInput').value = 0;
    await validateAndStore(payload, true);
    loadExact();
  } catch (err) { toast(err.message, true); }
}

async function validateAndStore(payload, normalize = false) {
  const result = await api('/api/validate', { dataset: payload, options: { normalize } });
  state.dataset = result.dataset;
  state.validation = result.validation;
  state.selectedSlot = 0;
  renderValidation();
  populateDimensionSelect();
  await runAnalysis();
}

async function handleFile(file) {
  try {
    const text = await file.text();
    const imported = await api('/api/import', { filename: file.name, text });
    state.dataset = imported.dataset;
    state.validation = imported.validation;
    state.selectedSlot = 0;
    renderValidation();
    populateDimensionSelect();
    toast(`已导入 ${imported.validation.pathCountAfterDuplicateMerge} 条合并后路径`);
    await runAnalysis();
  } catch (err) {
    let detail = err.message;
    if (err.data?.validation?.errors?.length) detail += `：${err.data.validation.errors[0].message}`;
    toast(`导入失败：${detail}`, true);
  }
}

function populateDimensionSelect() {
  const select = $('pathDimensionSelect');
  const dims = state.dataset?.dimensions || [];
  select.innerHTML = dims.map((d, i) => `<option value="${i}">${esc(d.label || d.key)}${d.unit ? `（${esc(d.unit)}）` : ''}</option>`).join('');
  state.selectedDimension = 0;
  select.value = '0';
}

/* ------------------------------ Validation ------------------------------ */

function renderValidation() {
  const v = state.validation;
  if (!v) return;
  $('datasetName').textContent = state.dataset.name || '已载入数据';
  const p = v.probabilities;
  const normCheck = p.checks.find(c => c.code === 'UNNORMALIZED_PROBABILITY');
  const zeroCheck = p.checks.find(c => c.code === 'ZERO_PROBABILITY');
  const negCheck = p.checks.find(c => c.code === 'NEGATIVE_PROBABILITY');
  const normalizeBtn = $('normalizeBtn');
  normalizeBtn.disabled = p.normalized || p.sum <= 0;
  normalizeBtn.textContent = p.normalized ? '概率已归一化' : `归一化概率（当前和 ${fmt(p.sum, 6)}）`;

  $('validationPanel').innerHTML = `
    ${checkCard('重复路径', v.duplicateRowsMerged ? 'warning' : 'pass',
      v.duplicateRowsMerged ? `先合并 ${v.duplicateRowsMerged} 条重复行 / ${v.duplicateGroupCount} 组，概率相加` : '无完全重复路径；重复检查先于概率检查',
      `原始 ${v.originalPathCount} → 合并后 ${v.pathCountAfterDuplicateMerge}`)}
    ${checkCard('零概率检查', zeroCheck.status === 'pass' ? 'pass' : 'warning', zeroCheck.message,
      p.zeroProbabilityIds?.length ? `示例 ID：${p.zeroProbabilityIds.slice(0, 5).map(esc).join(', ')}` : '零概率路径不会成为代表')}
    ${checkCard('未归一化检查', normCheck.status === 'pass' ? 'pass' : 'warning', normCheck.message,
      `Δ = ${signed(p.differenceFromOne, 6)}`)}
    ${checkCard('负概率/结构', negCheck.status === 'pass' && v.valid ? 'pass' : 'error', negCheck.message,
      v.valid ? 'Schema 与数值检查通过' : '存在必须修复的错误')}
  `;
}

function checkCard(title, kind, message, detail = '') {
  return `<div class="check">
    <div class="check-title"><span class="dot ${kind}"></span>${title}</div>
    <div class="check-msg">${message}</div>
    ${detail ? `<div class="check-msg mono">${detail}</div>` : ''}
  </div>`;
}

/* ------------------------------- Options -------------------------------- */

function currentOptions() {
  return {
    k: Number($('kInput').value),
    tailWeight: Number($('tailWeightInput').value),
    tailLevel: Number($('tailLevelInput').value),
    tailGuards: Number($('tailGuardsInput').value),
    distanceMetric: $('metricInput').value,
    timeCurve: $('timeCurveInput').value,
    prefixLocked: $('prefixLockedInput').checked,
    dropZero: $('dropZeroInput').checked,
    normalize: false,
  };
}

/* ------------------------------- Analysis ------------------------------- */

async function runAnalysis() {
  if (!state.dataset) return;
  setStatus('联合优化中…', 'running');
  $('runBtn').disabled = true;
  try {
    const options = currentOptions();
    state.analysis = await api('/api/analyze', { dataset: state.dataset, options });
    state.selectedSlot = 0;
    renderAllAnalysis();
    setStatus('完成', 'ok');
    const n = state.analysis.validation.pathCountAfterDuplicateMerge;
    if (n <= 1200) runPareto(true);
  } catch (err) {
    setStatus('运行失败', 'error');
    toast(`缩减失败：${err.message}`, true);
  } finally {
    $('runBtn').disabled = false;
  }
}

async function runPareto(silent = false) {
  if (!state.dataset) return;
  if (!silent) setStatus('计算代价前沿…', 'running');
  try {
    const base = currentOptions();
    const large = state.dataset.paths.length > 1200;
    const weights = large ? [0, 1, 3, 5] : [0, 0.25, 0.75, 1.5, 3, 5];
    state.pareto = await api('/api/pareto', { dataset: state.dataset, options: base, weights });
    renderTradeoff();
    if (!silent) setStatus('完成', 'ok');
  } catch (err) {
    toast(`代价前沿失败：${err.message}`, true);
    if (!silent) setStatus('前沿失败', 'error');
  }
}

function renderAllAnalysis() {
  renderKpis();
  renderTradeoff();
  renderRepresentatives();
  renderAssignmentTable();
  renderPathChart();
  renderTree();
  renderPlanning();
}

function renderKpis() {
  const a = state.analysis;
  const r = a.reduction;
  const m = r.metrics;
  const pc = a.planningComparison;
  const kpis = [
    { label: '路径 → 代表', value: `${fmtInt(a.validation.pathCountAfterDuplicateMerge)} → ${r.representatives.length}`, hint: r.method },
    { label: '概率守恒误差', value: m.probabilityConservation.error.toExponential(2), kind: m.probabilityConservation.conserved ? 'good' : 'bad', hint: `目标质量 ${fmtProb(m.probabilityConservation.inputMass)}` },
    { label: '平均业务距离', value: fmt(m.averageDistance, 4), hint: '按业务量纲标准化 + 前缀/时间权重' },
    { label: 'CVaR 原始 → 缩减', value: fmtMoney(m.reducedRisk.cvar), kind: m.cvarUnderestimate > 1 ? 'warn' : 'good', hint: `原始 ${fmtMoney(m.originalRisk.cvar)}；误差 ${signed(m.cvarError, 1)}` },
    { label: '极端最坏损失误差', value: signed(m.worstLossError, 1), kind: m.worstUnderestimate > 1 ? 'bad' : 'good', hint: `最坏 ${fmtMoney(m.originalRisk.worst)} → ${fmtMoney(m.reducedRisk.worst)}` },
    { label: '规划 Q：原始 / 缩减', value: `${fmt(pc.qOriginal, 1)} / ${fmt(pc.qReduced, 1)}`, kind: Math.abs(pc.outOfSampleRegret) > 1 ? 'warn' : 'good', hint: `Δ ${signed(pc.decisionDelta, 1)}` },
    { label: '样本外规划遗憾', value: fmtMoney(pc.outOfSampleRegret), kind: pc.outOfSampleRegret > 1 ? 'bad' : 'good', hint: `${fmt(pc.relativeRegretPercent, 3)}%；在原始分布上评估` },
    { label: '前缀/树检查', value: r.nonAnticipativity.compatible && r.treeValidation.valid ? '通过' : '冲突', kind: r.nonAnticipativity.compatible && r.treeValidation.valid ? 'good' : 'bad', hint: `NA 违规 ${r.nonAnticipativity.violations?.length || 0}；树节点 ${r.treeValidation.nodeCount}` },
  ];
  $('kpiSection').innerHTML = kpis.map(k => `<div class="kpi ${k.kind || ''}"><div class="label">${k.label}</div><div class="value">${k.value}</div><div class="hint">${k.hint}</div></div>`).join('');
}

/* ------------------------------ Tradeoff UI ----------------------------- */

function renderTradeoff() {
  renderTradeoffCallouts();
  renderParetoTable();
  renderParetoChart();
}

function renderTradeoffCallouts() {
  const p = state.pareto;
  const box = $('tradeoffCallouts');
  if (!p) {
    box.innerHTML = `<div class="callout"><strong>等待代价前沿</strong><span>运行后同时显示“保护尾部导致平均距离增加多少”和“只优化平均距离会低估多少极端损失”。</span></div>`;
    return;
  }
  const c = p.costs;
  box.innerHTML = `
    <div class="callout">
      <strong>平均距离最优（λ=0）的尾部代价</strong>
      <span>平均距离 ${fmt(c.distanceFirstDistance, 4)}；CVaR 低估 <b>${fmtMoney(c.distanceFirstCvarUnderestimate)}</b>；规划遗憾 ${fmtMoney(c.distanceFirstPlanningRegret)}。</span>
    </div>
    <div class="callout orange">
      <strong>强尾部保护的分布距离代价</strong>
      <span>相对最小平均距离多付 <b>${fmt(c.tailStrongDistancePremium, 4)}</b> 个标准化距离单位；残余 |CVaR误差| ${fmtMoney(c.tailStrongResidualCvarError)}。</span>
    </div>
    <div class="callout green">
      <strong>必须同时看两种代价</strong>
      <span>不存在单一“最优 K/λ”；下表每一行都由一次代表选择与概率再分配联合优化产生。</span>
    </div>`;
}

function renderParetoTable() {
  const tbody = document.querySelector('#paretoTable tbody');
  if (!state.pareto) { tbody.innerHTML = ''; return; }
  tbody.innerHTML = state.pareto.grid.map(x => `
    <tr>
      <td class="mono-cell">${fmt(x.tailWeight, 2)}</td>
      <td>${fmt(x.averageDistance, 4)}</td>
      <td>${signed(x.distancePriceOfTailProtection, 4)}</td>
      <td>${fmtMoney(x.absoluteCvarError)}</td>
      <td class="${x.cvarUnderestimate > 1 ? 'mono-cell' : ''}">${fmtMoney(x.cvarUnderestimate)}</td>
      <td>${fmtMoney(x.planningRegret)}</td>
      <td>${fmt(x.qReduced, 1)}</td>
    </tr>`).join('');
}

function renderParetoChart() {
  const el = $('paretoChart');
  if (!state.pareto) { el.innerHTML = '<div class="empty-state">代价前沿尚未生成</div>'; return; }
  const data = state.pareto.grid;
  const W = 560, H = 310, M = { l: 58, r: 24, t: 18, b: 46 };
  const iw = W - M.l - M.r, ih = H - M.t - M.b;
  const xs = data.map(d => d.averageDistance);
  const ys = data.map(d => d.absoluteCvarError);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = 0, ymax = Math.max(...ys) * 1.12 || 1;
  const x = v => M.l + ((v - xmin) / ((xmax - xmin) || 1)) * iw;
  const y = v => M.t + ih - ((v - ymin) / (ymax - ymin)) * ih;
  const ticks = 4;
  let grid = '';
  for (let i = 0; i <= ticks; i++) {
    const yy = M.t + ih * i / ticks;
    const val = ymax - (ymax - ymin) * i / ticks;
    grid += `<line x1="${M.l}" x2="${W - M.r}" y1="${yy}" y2="${yy}" stroke="${CHART_COLORS.gridLight}"/>
             <text x="${M.l - 8}" y="${yy + 4}" text-anchor="end" font-size="11" fill="${CHART_COLORS.text}">${fmtMoney(val)}</text>`;
  }
  const path = data.map((d, i) => `${i ? 'L' : 'M'}${x(d.averageDistance).toFixed(1)},${y(d.absoluteCvarError).toFixed(1)}`).join(' ');
  const points = data.map(d => {
    const cx = x(d.averageDistance), cy = y(d.absoluteCvarError);
    const tip = `<div class="tt-title">λ=${fmt(d.tailWeight, 2)}</div>平均距离 ${fmt(d.averageDistance, 4)}<br>|CVaR误差| ${fmtMoney(d.absoluteCvarError)}<br>CVaR低估 ${fmtMoney(d.cvarUnderestimate)}<br>规划遗憾 ${fmtMoney(d.planningRegret)}`;
    return `<circle cx="${cx}" cy="${cy}" r="6.5" fill="${CHART_COLORS.blue}" stroke="var(--surface)" stroke-width="2" tabindex="0" data-tooltip="${tip.replace(/"/g, '&quot;')}"/>
      <text x="${cx + 8}" y="${cy - 8}" font-size="11" fill="${CHART_COLORS.text}">λ${fmt(d.tailWeight, 1)}</text>`;
  }).join('');
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="平均距离与绝对 CVaR 误差散点连线">
    ${grid}
    <line x1="${M.l}" y1="${M.t + ih}" x2="${W - M.r}" y2="${M.t + ih}" stroke="${CHART_COLORS.axisLight}"/>
    <line x1="${M.l}" y1="${M.t}" x2="${M.l}" y2="${M.t + ih}" stroke="${CHART_COLORS.axisLight}"/>
    <text x="${M.l + iw / 2}" y="${H - 12}" text-anchor="middle" font-size="12" fill="${CHART_COLORS.text}">平均业务分布距离 → 越低越好</text>
    <text transform="translate(15 ${M.t + ih / 2}) rotate(-90)" text-anchor="middle" font-size="12" fill="${CHART_COLORS.text}">|CVaR 误差| → 越低越好</text>
    <path d="${path}" fill="none" stroke="${CHART_COLORS.blue}" stroke-width="2.2" opacity=".55"/>
    ${points}
  </svg>`;
}

/* --------------------------- Assignment explorer ------------------------ */

function renderRepresentatives() {
  const reps = state.analysis.representatives;
  if (state.selectedSlot >= reps.length) state.selectedSlot = 0;
  const maxQ = Math.max(...reps.map(r => r.probability), 1e-9);
  $('repList').innerHTML = reps.map(r => `
    <button class="rep-button ${r.slot === state.selectedSlot ? 'active' : ''}" data-slot="${r.slot}">
      <div class="rep-top"><span class="mono-cell">${esc(r.id)}</span><span>${pct(r.probability)}</span></div>
      <div class="rep-meta"><span>${r.memberCount} 条原路径</span><span>均距 ${fmt(r.meanDistance, 3)}</span></div>
      <div class="rep-meta"><span>损失 ${fmtMoney(r.loss)}</span><span>${r.isTail ? '尾部代表' : ''}</span></div>
      <div class="mass-bar"><i style="width:${100 * r.probability / maxQ}%"></i></div>
    </button>`).join('');
  document.querySelectorAll('.rep-button').forEach(btn => btn.addEventListener('click', () => {
    state.selectedSlot = Number(btn.dataset.slot);
    renderRepresentatives();
    renderAssignmentTable();
    renderPathChart();
  }));
  const totalAssignments = state.analysis.assignments.length;
  $('assignmentCount').textContent = `${totalAssignments} 条已检查/分配路径；表格最多渲染 200 行，可用搜索缩小范围`;
}

function selectedRepresentative() {
  return state.analysis.representatives.find(r => r.slot === state.selectedSlot) || state.analysis.representatives[0];
}

function renderAssignmentTable() {
  if (!state.analysis) return;
  const rep = selectedRepresentative();
  const q = state.assignmentQuery.trim().toLowerCase();
  let rows = state.analysis.assignments.filter(a => a.representativeIndex === rep.pathIndex);
  if (q) rows = rows.filter(a => a.id.toLowerCase().includes(q) || (a.tags || []).join(' ').toLowerCase().includes(q) || a.representativeId.toLowerCase().includes(q));
  const shown = rows.slice(0, 200);
  document.querySelector('#assignmentTable tbody').innerHTML = shown.map(a => `
    <tr>
      <td class="id-cell">${esc(a.id)}${(a.tags || []).slice(0, 2).map(t => `<span class="tag">${esc(t)}</span>`).join('')}</td>
      <td>${fmtProb(a.probability)}</td>
      <td class="id-cell">${esc(a.representativeId)}</td>
      <td>${fmt(a.distance, 4)}</td>
      <td style="color:${a.lossGap < -1 ? 'var(--red)' : 'inherit'}">${signed(a.lossGap, 1)}</td>
      <td>${a.isTail ? '<span class="tag tail">尾部</span>' : '—'}</td>
    </tr>`).join('') || `<tr><td colspan="6" style="text-align:center;color:var(--muted)">没有匹配路径</td></tr>`;
}

function renderPathChart() {
  if (!state.analysis || !state.dataset) return;
  const rep = selectedRepresentative();
  const dim = Number($('pathDimensionSelect').value || state.selectedDimension || 0);
  state.selectedDimension = dim;
  const dimMeta = state.dataset.dimensions[dim] || { label: `D${dim + 1}`, unit: '' };
  const periods = state.dataset.periods;
  const repPath = state.dataset.paths[rep.pathIndex];
  const memberIndices = state.analysis.assignments
    .filter(a => a.representativeIndex === rep.pathIndex)
    .slice(0, 80)
    .map(a => a.pathIndex);
  const allSeries = [repPath, ...memberIndices.map(i => state.dataset.paths[i])];
  const vals = allSeries.flatMap(p => p.values.map(row => row[dim]));
  let min = Math.min(...vals), max = Math.max(...vals);
  if (min === max) { min -= 1; max += 1; }
  const padY = (max - min) * 0.1;
  min -= padY; max += padY;
  const W = 620, H = 220, M = { l: 48, r: 16, t: 14, b: 34 };
  const iw = W - M.l - M.r, ih = H - M.t - M.b;
  const x = t => M.l + (periods.length === 1 ? iw / 2 : t / (periods.length - 1) * iw);
  const y = v => M.t + ih - (v - min) / (max - min) * ih;
  const grid = [0, .25, .5, .75, 1].map(f => {
    const yy = M.t + ih * f;
    const val = max - (max - min) * f;
    return `<line x1="${M.l}" x2="${W - M.r}" y1="${yy}" y2="${yy}" stroke="${CHART_COLORS.gridLight}"/>
      <text x="${M.l - 7}" y="${yy + 4}" text-anchor="end" font-size="10.5" fill="${CHART_COLORS.text}">${fmt(val, 1)}</text>`;
  }).join('');
  const thin = memberIndices.map((idx) => {
    const p = state.dataset.paths[idx];
    const d = p.values.map((row, t) => `${t ? 'L' : 'M'}${x(t)},${y(row[dim])}`).join(' ');
    return `<path d="${d}" fill="none" stroke="${CHART_COLORS.blue}" stroke-width="1" opacity=".16"/>`;
  }).join('');
  const repD = repPath.values.map((row, t) => `${t ? 'L' : 'M'}${x(t)},${y(row[dim])}`).join(' ');
  const dots = repPath.values.map((row, t) => `<circle cx="${x(t)}" cy="${y(row[dim])}" r="4" fill="${CHART_COLORS.orange}" stroke="var(--surface)" stroke-width="1.5"/>`).join('');
  const labels = periods.map((p, t) => `<text x="${x(t)}" y="${H - 11}" text-anchor="middle" font-size="10.5" fill="${CHART_COLORS.text}">${p.label || `T${t + 1}`}</text>`).join('');
  $('pathChart').innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="代表路径与被分配路径轨迹">
    ${grid}${thin}
    <path d="${repD}" fill="none" stroke="${CHART_COLORS.orange}" stroke-width="3"/>
    ${dots}${labels}
    <text x="${M.l}" y="12" font-size="12" font-weight="750" fill="${CHART_COLORS.text}">${esc(dimMeta.label || dimMeta.key)}${dimMeta.unit ? ` / ${esc(dimMeta.unit)}` : ''} · 橙色为代表 ${esc(rep.id)}，浅蓝为被分配原路径（最多80条）</text>
  </svg>`;
}

/* -------------------------------- Tree ---------------------------------- */

function renderTree() {
  const r = state.analysis.reduction;
  const tree = r.tree;
  const tv = r.treeValidation;
  const na = r.nonAnticipativity;
  $('treeBadges').innerHTML = `
    <span class="badge ${tv.valid ? 'pass' : 'fail'}">${tv.valid ? '树结构/概率 ✓' : '树检查失败'}</span>
    <span class="badge ${na.compatible ? 'pass' : 'fail'}">${na.compatible ? '非预知映射 ✓' : `NA 违规 ${na.violations?.length || 0}`}</span>
    <span class="badge">节点 ${tv.nodeCount} · 叶 ${tv.leafCount} · 分叉 ${tv.branchPointCount}</span>
    <span class="badge">守恒误差 ${tv.conservationError.toExponential(2)}</span>`;
  const renderNode = (id) => {
    const n = tree.nodes[id];
    if (!n.children || !n.children.length) {
      return `<li><div class="tree-leaf"><b>${esc(n.pathId || id)}</b><span class="node-prob">${pct(n.probability)}</span>
        <div class="prefix-values">前缀 ${JSON.stringify(n.prefixValues)}</div></div></li>`;
    }
    const kids = n.children.map(renderNode).join('');
    return `<li><details open class="tree-node">
      <summary>${n.label}<span class="node-prob">${pct(n.probability)}</span><span class="cond-prob">条件 ${pct(n.conditionalProbability, 1)}</span></summary>
      <ul>${kids}</ul></details></li>`;
  };
  $('treeContainer').innerHTML = `<ul>${renderNode(tree.rootId)}</ul>`;
}

/* ------------------------------ Planning UI ----------------------------- */

function renderPlanning() {
  const pc = state.analysis.planningComparison;
  const fullDecision = state.analysis.fullPlanning.decision;
  const redDecision = state.analysis.reduction.reducedDataset ? pc.reduced.decision : null;
  const redOnFull = pc.reducedDecisionOnOriginal;
  $('planningKpis').innerHTML = [
    kpiMini('原始最优 Q', fmt(pc.qOriginal, 2), `目标 ${fmtMoney(pc.originalObjective)}`),
    kpiMini('缩减最优 Q', fmt(pc.qReduced, 2), `样本内目标 ${fmtMoney(pc.reducedObjectiveInSample)}`),
    kpiMini('样本外遗憾', fmtMoney(pc.outOfSampleRegret), `${fmt(pc.relativeRegretPercent, 3)}%`, pc.outOfSampleRegret > 1 ? 'bad' : 'good'),
    kpiMini('缩减 Q 的服务水平', pct(redOnFull.expectedServiceLevel, 2), `应急单位 ${fmt(redOnFull.expectedEmergencyUnits, 2)}`),
    kpiMini('缩减 Q 的95% CVaR', fmtMoney(pc.tailRiskOfReducedQOnOriginal.cvar), `最坏 ${fmtMoney(pc.tailRiskOfReducedQOnOriginal.worst)}`),
  ].join('');

  document.querySelector('#planningTable tbody').innerHTML = `
    <tr><th>第一阶段备货量 Q</th><td>${fmt(pc.qOriginal, 2)}</td><td>${fmt(pc.qReduced, 2)}</td><td>Δ ${signed(pc.decisionDelta, 2)} (${pc.decisionDeltaPercent === null ? '—' : signed(pc.decisionDeltaPercent, 2) + '%'})</td></tr>
    <tr><th>期望总成本</th><td>${fmtMoney(pc.originalObjective)}</td><td>${fmtMoney(pc.reducedObjectiveInSample)}</td><td>${fmtMoney(redOnFull.expectedCost)}（在原始分布）</td></tr>
    <tr><th>服务水平</th><td>${pct(fullDecision.expectedServiceLevel)}</td><td>${pct(redDecision.expectedServiceLevel)}</td><td>${pct(redOnFull.expectedServiceLevel)}</td></tr>
    <tr><th>加权满足率</th><td>${pct(fullDecision.fillRate)}</td><td>${pct(redDecision.fillRate)}</td><td>${pct(redOnFull.fillRate)}</td></tr>
    <tr><th>期望应急数量</th><td>${fmt(fullDecision.expectedEmergencyUnits, 2)}</td><td>${fmt(redDecision.expectedEmergencyUnits, 2)}</td><td>${fmt(redOnFull.expectedEmergencyUnits, 2)}</td></tr>
    <tr><th>样本外遗憾</th><td colspan="2">同一候选 Q 网格：${pc.candidateCount} 点 [${fmt(pc.candidateRange[0], 1)}, ${fmt(pc.candidateRange[1], 1)}]</td><td><b>${fmtMoney(pc.outOfSampleRegret)}</b></td></tr>`;
  renderPlanningChart(pc);
}
function kpiMini(label, value, hint, kind = '') {
  return `<div class="kpi ${kind}"><div class="label">${label}</div><div class="value" style="font-size:21px">${value}</div><div class="hint">${hint}</div></div>`;
}

function renderPlanningChart(pc) {
  const orig = pc.originalCurve.filter(Boolean);
  const red = pc.reducedCurve.filter(Boolean);
  const data = orig.map((p, i) => ({ q: p.q, a: p.expectedCost, b: red[i]?.expectedCost })).filter(d => Number.isFinite(d.b));
  const W = 640, H = 370, M = { l: 66, r: 20, t: 20, b: 48 };
  const iw = W - M.l - M.r, ih = H - M.t - M.b;
  const xs = data.map(d => d.q);
  const ys = data.flatMap(d => [d.a, d.b]);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys) * .985, ymax = Math.max(...ys) * 1.015;
  const x = v => M.l + (v - xmin) / ((xmax - xmin) || 1) * iw;
  const y = v => M.t + ih - (v - ymin) / ((ymax - ymin) || 1) * ih;
  let grid = '';
  for (let i = 0; i <= 5; i++) {
    const yy = M.t + ih * i / 5;
    const val = ymax - (ymax - ymin) * i / 5;
    grid += `<line x1="${M.l}" x2="${W - M.r}" y1="${yy}" y2="${yy}" stroke="${CHART_COLORS.gridLight}"/>
      <text x="${M.l - 8}" y="${yy + 4}" text-anchor="end" font-size="11" fill="${CHART_COLORS.text}">${fmtMoney(val)}</text>`;
  }
  const line = key => data.map((d, i) => `${i ? 'L' : 'M'}${x(d.q).toFixed(1)},${y(d[key]).toFixed(1)}`).join(' ');
  const qLines = `
    <line x1="${x(pc.qOriginal)}" x2="${x(pc.qOriginal)}" y1="${M.t}" y2="${M.t + ih}" stroke="${CHART_COLORS.blue}" stroke-dasharray="5 4" opacity=".65"/>
    <line x1="${x(pc.qReduced)}" x2="${x(pc.qReduced)}" y1="${M.t}" y2="${M.t + ih}" stroke="${CHART_COLORS.orange}" stroke-dasharray="5 4" opacity=".75"/>`;
  const tipCircles = data.map((d) => {
    const tip = `<div class="tt-title">Q=${fmt(d.q, 1)}</div>原始期望成本 ${fmtMoney(d.a)}<br>缩减样本内成本 ${fmtMoney(d.b)}`;
    return `<circle cx="${x(d.q)}" cy="${M.t + ih / 2}" r="14" fill="transparent" data-tooltip="${tip.replace(/"/g, '&quot;')}"/>`;
  }).join('');
  $('planningChart').innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="原始和缩减情景的规划目标曲线">
    ${grid}
    <line x1="${M.l}" y1="${M.t + ih}" x2="${W - M.r}" y2="${M.t + ih}" stroke="${CHART_COLORS.axisLight}"/>
    ${qLines}
    <path d="${line('a')}" fill="none" stroke="${CHART_COLORS.blue}" stroke-width="2.5"/>
    <path d="${line('b')}" fill="none" stroke="${CHART_COLORS.orange}" stroke-width="2.5"/>
    ${tipCircles}
    <text x="${M.l + iw / 2}" y="${H - 12}" text-anchor="middle" font-size="12" fill="${CHART_COLORS.text}">第一阶段备货量 Q</text>
    <text transform="translate(17 ${M.t + ih / 2}) rotate(-90)" text-anchor="middle" font-size="12" fill="${CHART_COLORS.text}">期望总成本</text>
    <g><rect x="${W - 178}" y="18" width="164" height="52" rx="9" fill="var(--surface)" stroke="var(--border)"/>
      <line x1="${W - 164}" x2="${W - 136}" y1="36" y2="36" stroke="${CHART_COLORS.blue}" stroke-width="3"/><text x="${W - 128}" y="40" font-size="12" fill="${CHART_COLORS.text}">原始 ${fmt(pc.qOriginal, 1)}</text>
      <line x1="${W - 164}" x2="${W - 136}" y1="57" y2="57" stroke="${CHART_COLORS.orange}" stroke-width="3"/><text x="${W - 128}" y="61" font-size="12" fill="${CHART_COLORS.text}">缩减 ${fmt(pc.qReduced, 1)}</text>
    </g>
  </svg>`;
}

/* ------------------------------ Exact check ----------------------------- */

async function loadExact() {
  const panel = $('exactPanel');
  panel.innerHTML = '<div class="empty-state"><span class="spinner"></span>正在枚举全部组合…</div>';
  try {
    state.exact = await api('/api/exact-check', { k: 3, tailWeight: 0.75 });
    renderExact();
  } catch (err) {
    panel.innerHTML = `<div class="empty-state" style="color:var(--red)">完全搜索失败：${err.message}</div>`;
  }
}

function renderExact() {
  const e = state.exact;
  if (!e) return;
  const o = e.optimality;
  const exact = e.exact, heur = e.heuristic;
  $('exactPanel').innerHTML = `
    <div class="kv-list">
      <div class="kv"><div class="k">小实例 / K</div><div class="v">${e.instance.pathCount} 条 / ${e.instance.k} 个代表</div></div>
      <div class="kv"><div class="k">完全搜索组合</div><div class="v">C(${e.instance.pathCount},${e.instance.k}) = ${e.instance.subsets}</div></div>
      <div class="kv ${o.matchesExhaustiveOptimum ? 'good' : 'warn'}"><div class="k">启发式最优性</div><div class="v">${o.matchesExhaustiveOptimum ? '命中完全最优 ✓' : `差距 ${fmt(o.relativeGap * 100, 4)}%`}</div></div>
      <div class="kv"><div class="k">目标值</div><div class="v">${fmt(o.exactObjective, 8)} / H ${fmt(o.heuristicObjective, 8)}</div></div>
    </div>
    <div class="table-wrap" style="max-height:none;margin-top:12px">
      <table>
        <thead><tr><th>方法</th><th>代表 ID</th><th>概率守恒误差</th><th>非预知</th><th>树结构</th><th>叶概率和</th></tr></thead>
        <tbody>
          <tr><td>完全枚举</td><td class="id-cell">${exact.representativeIds.map(esc).join(', ')}</td><td>${exact.probabilityConservation.error.toExponential(2)}</td><td>${exact.nonAnticipativity.compatible ? '✓' : '✗'}</td><td>${exact.treeValidation.valid ? '✓' : '✗'}</td><td>${fmtProb(exact.treeValidation.leafProbabilitySum)}</td></tr>
          <tr><td>启发式</td><td class="id-cell">${heur.representativeIds.map(esc).join(', ')}</td><td>${heur.probabilityConservation.error.toExponential(2)}</td><td>${heur.nonAnticipativity.compatible ? '✓' : '✗'}</td><td>${heur.treeValidation.valid ? '✓' : '✗'}</td><td>${fmtProb(heur.treeValidation.leafProbabilitySum)}</td></tr>
        </tbody>
      </table>
    </div>`;
}

/* ----------------------------- Recovery drill --------------------------- */

async function loadRecovery() {
  const panel = $('recoveryPanel');
  panel.innerHTML = '<div class="empty-state"><span class="spinner"></span>正在注入中断并从检查点恢复…</div>';
  try {
    state.recovery = await api('/api/recovery-drill', {});
    renderRecovery();
  } catch (err) {
    panel.innerHTML = `<div class="empty-state" style="color:var(--red)">恢复演练失败：${err.message}</div>`;
  }
}
function renderRecovery() {
  const r = state.recovery;
  const f = r.final;
  $('recoveryPanel').innerHTML = `
    <div class="kv-list">
      <div class="kv ${r.interrupted ? 'warn' : ''}"><div class="k">中断注入</div><div class="v">第 ${r.crashAfterNewCommits} 个新提交后崩溃</div></div>
      <div class="kv"><div class="k">检查点已持久化</div><div class="v">${r.loadedFromCheckpoint.totalEntries} 条 / 质量 ${fmtProb(r.loadedFromCheckpoint.allocatedMass)}</div></div>
      <div class="kv good"><div class="k">恢复后新提交</div><div class="v">${r.resume.newCommits} 条；总 ${r.resume.totalEntries}</div></div>
      <div class="kv ${f.valid && r.matchesCleanAllocation ? 'good' : 'bad'}"><div class="k">幂等/守恒</div><div class="v">${f.valid && r.matchesCleanAllocation ? '通过 ✓' : '失败'}</div></div>
    </div>
    <p class="muted" style="margin:10px 0">完成后再次重投整批：新提交 ${r.redeliveryNoOp.newCommits}（应为 0）；重复处理 ID ${f.duplicateProcessedIds.length} 个。各代表质量与一次干净分配完全相同：${r.matchesCleanAllocation ? '是' : '否'}。</p>
    <pre class="json">${JSON.stringify(f.massByRepresentative, null, 2)}</pre>`;
}

/* -------------------------------- Events -------------------------------- */

$('loadSampleBtn').addEventListener('click', () => loadSample(180));
$('load10kBtn').addEventListener('click', () => loadSample(10000));
$('loadToyBtn').addEventListener('click', loadToy);
$('runBtn').addEventListener('click', runAnalysis);
$('paretoBtn').addEventListener('click', () => runPareto(false));
$('exactBtn').addEventListener('click', loadExact);
$('recoveryBtn').addEventListener('click', loadRecovery);
$('normalizeBtn').addEventListener('click', async () => {
  try {
    const result = await api('/api/validate', { dataset: state.dataset, options: { normalize: true } });
    state.dataset = result.dataset;
    state.validation = result.validation;
    renderValidation();
    toast('概率已归一化；此前未归一化状态已单独记录');
    await runAnalysis();
  } catch (err) { toast(err.message, true); }
});
$('fileInput').addEventListener('change', (e) => {
  const file = e.target.files?.[0];
  if (file) handleFile(file);
  e.target.value = '';
});
$('kInput').addEventListener('input', (e) => { $('kValue').textContent = e.target.value; });
$('tailWeightInput').addEventListener('input', (e) => { $('tailWeightValue').textContent = Number(e.target.value).toFixed(2); });
$('assignmentSearch').addEventListener('input', (e) => { state.assignmentQuery = e.target.value; renderAssignmentTable(); });
$('pathDimensionSelect').addEventListener('change', renderPathChart);

window.addEventListener('DOMContentLoaded', async () => {
  $('tailWeightValue').textContent = Number($('tailWeightInput').value).toFixed(2);
  try {
    await loadSample(180);
    loadExact();
    loadRecovery();
  } catch (err) {
    toast(`初始化失败：${err.message}`, true);
  }
});
