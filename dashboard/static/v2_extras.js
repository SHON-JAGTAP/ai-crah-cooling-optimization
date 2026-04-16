/* v2 additions: mode/phase controls, aisle panel, BMS panel, supervised queue */

/* ── Mode & Phase controls ─────────────────────────────── */
async function setMode(mode) {
  await fetch('/api/mode', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode })
  });
  updateModeButtons(mode);
}

async function setPhase(phase) {
  const res = await fetch('/api/phase', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ phase: parseInt(phase) })
  });
  const d = await res.json();
  const el = document.getElementById('phase-label');
  if (el) el.textContent = d.label || '';
}

function updateModeButtons(mode) {
  const map = { ai: 'AI', sup: 'SUPERVISED', local: 'LOCAL_AUTO' };
  Object.keys(map).forEach(function(k) {
    const btn = document.getElementById('btn-mode-' + k);
    if (btn) btn.classList.toggle('active', mode === map[k]);
  });
}

/* ── Aisle rendering ─────────────────────────────────────── */
async function fetchAisles() {
  try {
    const res = await fetch('/api/aisles');
    const d = await res.json();
    renderAisles(d.aisles || []);
  } catch (e) {}
}

function renderAisles(aisles) {
  const grid = document.getElementById('aisle-grid');
  if (!aisles.length || !grid) return;
  grid.innerHTML = '';
  aisles.forEach(function(a) {
    const c = sevColor(a.severity);
    const card = document.createElement('div');
    card.className = 'aisle-card';
    card.style.borderColor = c + '44';
    const compound = a.compound ? '<span class="compound-badge">COMPOUND</span>' : '';
    card.innerHTML =
      '<div class="aisle-name" style="color:' + c + '">' + a.aisle + ' ' + compound + '</div>' +
      '<div class="aisle-stat">Avg Temp <span>' + (a.avg_temp || 0).toFixed(1) + '&deg;C</span></div>' +
      '<div class="aisle-stat">Max Temp <span>' + (a.max_temp || 0).toFixed(1) + '&deg;C</span></div>' +
      '<div class="aisle-stat">Zone Spread <span>' + (a.spread || 0).toFixed(1) + '&deg;C</span></div>' +
      '<div class="aisle-stat">CRAH <span>CRAH-' + a.crah_id + '</span></div>' +
      '<div><span class="aisle-badge" style="background:' + c + '22;color:' + c + ';border:1px solid ' + c + '44">' + a.severity + '</span></div>';
    grid.appendChild(card);
  });
}

/* ── BMS status rendering ───────────────────────────────── */
async function fetchBMSStatus() {
  try {
    const res = await fetch('/api/bms/status');
    const d = await res.json();
    renderBMSStatus(d);
  } catch (e) {}
}

function renderBMSStatus(d) {
  const grid  = document.getElementById('bms-grid');
  const badge = document.getElementById('bms-mode-badge');
  if (!grid || !badge) return;
  const isLive = d.mode === 'LIVE';
  badge.textContent = d.mode;
  badge.style.background = isLive ? 'rgba(34,197,94,0.15)' : 'rgba(245,158,11,0.15)';
  badge.style.color = isLive ? '#22c55e' : '#f59e0b';
  const cls = isLive ? 'bms-connected' : 'bms-simulation';
  const stats = [
    ['Mode', d.mode],
    ['Host', d.bms_host],
    ['Total Tags', (d.total_tags || 0).toLocaleString()],
    ['Registered', d.registered_tags],
    ['Poll', (d.poll_interval_s || 0) + 's'],
    ['Reads', d.read_count || 0],
    ['Writes', d.write_count || 0],
    ['Errors', d.error_count || 0],
  ];
  grid.innerHTML = '';
  stats.forEach(function(item) {
    grid.innerHTML += '<div class="bms-stat"><div class="bms-stat-label">' + item[0] + '</div>' +
      '<div class="bms-stat-value ' + cls + '">' + (item[1] != null ? item[1] : '-') + '</div></div>';
  });
}

/* ── Supervised queue rendering ─────────────────────────── */
async function fetchSupervisedQueue() {
  try {
    const res = await fetch('/api/supervised/pending');
    const d = await res.json();
    renderSupervisedQueue(d);
  } catch (e) {}
}

function renderSupervisedQueue(d) {
  const list  = document.getElementById('pending-list');
  const count = document.getElementById('sup-pending-count');
  if (!list || !count) return;
  const stats = d.stats || {};
  const items = d.pending || [];
  count.textContent = items.length;
  ['proposed', 'approved', 'rejected', 'expired'].forEach(function(k) {
    const el = document.getElementById('sup-stat-' + k);
    if (el) el.textContent = stats[k] || 0;
  });
  if (!items.length) {
    list.innerHTML = '<div class="sup-empty">No pending actions. Switch to Supervised mode to see proposals here.</div>';
    return;
  }
  list.innerHTML = '';
  items.forEach(function(a) {
    const c = sevColor(a.severity);
    const pct = Math.min(100, Math.round((a.age_seconds / a.timeout_seconds) * 100));
    const item = document.createElement('div');
    item.className = 'pending-item';
    item.innerHTML =
      '<div class="pending-header">' +
        '<div class="pending-aisle">' + a.aisle + ' / CRAH-' + a.crah_id + '</div>' +
        '<div class="pending-age" style="color:' + c + '">' + a.severity + ' &bull; ' + a.age_seconds + 's / ' + a.timeout_seconds + 's</div>' +
      '</div>' +
      '<div class="pending-reason">' + a.reason + '</div>' +
      '<div class="pending-deltas">' +
        '<div class="delta-box"><div class="delta-label">Airflow</div><div class="delta-val">' + (a.current_airflow || 0).toFixed(0) + ' &rarr; ' + (a.proposed_airflow || 0).toFixed(0) + ' CFM</div></div>' +
        '<div class="delta-box"><div class="delta-label">Discharge</div><div class="delta-val">' + (a.current_discharge || 0).toFixed(1) + ' &rarr; ' + (a.proposed_discharge || 0).toFixed(1) + '&deg;C</div></div>' +
        '<div class="delta-box"><div class="delta-label">Avg/Max</div><div class="delta-val">' + (a.avg_rack_temp || 0).toFixed(1) + ' / ' + (a.max_rack_temp || 0).toFixed(1) + '&deg;C</div></div>' +
        '<div class="delta-box"><div class="delta-label">Timeout</div><div class="delta-val">' + pct + '%</div></div>' +
      '</div>' +
      '<div class="pending-actions">' +
        '<button class="btn-approve" onclick="approveAction(\'' + a.action_id + '\')">&#10003; Approve</button>' +
        '<button class="btn-reject"  onclick="rejectAction(\'' + a.action_id + '\')">&#10007; Reject</button>' +
      '</div>';
    list.appendChild(item);
  });
}

async function approveAction(id) {
  await fetch('/api/supervised/approve/' + id, { method: 'POST' });
  fetchSupervisedQueue();
}

async function rejectAction(id) {
  var reason = prompt('Rejection reason (optional):') || '';
  await fetch('/api/supervised/reject/' + id, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ reason: reason })
  });
  fetchSupervisedQueue();
}

/* ── Patch fetchStatus to call new panels ────────────────── */
var _origFetchStatus = fetchStatus;
fetchStatus = async function() {
  await _origFetchStatus();
  fetchAisles();
  fetchBMSStatus();
  fetchSupervisedQueue();
  // Sync mode buttons from snapshot
  try {
    var snap = await fetch('/api/mode').then(function(r) { return r.json(); });
    updateModeButtons(snap.control_mode);
    var pl = document.getElementById('phase-label');
    if (pl) pl.textContent = snap.phase_label || '';
    var ps = document.getElementById('phase-select');
    if (ps && snap.phase) ps.value = snap.phase;
  } catch(e) {}
};
