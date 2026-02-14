// Pocharlies Qdrant — RAG Dashboard
'use strict';

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// ── Collections ──────────────────────────────────────────────

let _ragCrawlEventSource = null;

async function fetchRagCollections() {
  const el = document.getElementById('rag-collections');
  if (!el) return;
  try {
    const r = await fetch('/web/collections');
    const data = await r.json();
    renderRagCollections(data.collections || []);
  } catch (e) {
    el.innerHTML = `<p class="muted">Collections unavailable: ${e.message}</p>`;
  }
}

function renderRagCollections(collections) {
  const el = document.getElementById('rag-collections');
  if (!el) return;
  if (!collections.length) {
    el.innerHTML = '<p class="muted">No collections found.</p>';
    return;
  }
  el.innerHTML = collections.map(c => `
    <div class="rag-collection-card glass">
      <div class="rag-cc-header">
        <span class="rag-cc-name">${escapeHtml(c.name)}</span>
        <span class="rag-status-badge ${c.status === 'green' ? 'active' : 'inactive'}">${escapeHtml(c.status)}</span>
      </div>
      <div class="rag-cc-stats">
        <div class="rag-cc-stat">
          <span class="rag-cc-stat-val">${(c.points_count || 0).toLocaleString()}</span>
          <span class="rag-cc-stat-lbl">chunks</span>
        </div>
        <div class="rag-cc-stat">
          <span class="rag-cc-stat-val">${(c.vectors_count || 0).toLocaleString()}</span>
          <span class="rag-cc-stat-lbl">indexed vectors</span>
        </div>
      </div>
    </div>
  `).join('');
}

// ── Sources ──────────────────────────────────────────────────

async function fetchRagSources() {
  const el = document.getElementById('rag-sources');
  if (!el) return;
  try {
    const r = await fetch('/web/sources');
    const data = await r.json();
    renderRagSources(data.sources || []);
    updateSearchDomainDropdown(data.sources || []);
  } catch (e) {
    el.innerHTML = `<p class="muted">Sources unavailable: ${e.message}</p>`;
  }
}

function renderRagSources(sources) {
  const el = document.getElementById('rag-sources');
  if (!el) return;
  if (!sources.length) {
    el.innerHTML = '<p class="muted">No sources indexed yet. Use the Web Scraper above to crawl a website.</p>';
    return;
  }
  el.innerHTML = `<div class="rag-sources-list">${sources.map(s => `
    <div class="rag-source-row glass">
      <div class="rag-source-info">
        <span class="rag-source-domain">${escapeHtml(s.domain)}</span>
        <span class="rag-source-meta">${s.url_count} pages &middot; ${s.chunk_count} chunks &middot; ${s.last_indexed ? new Date(s.last_indexed).toLocaleDateString() : 'N/A'}</span>
      </div>
      <button class="btn btn-danger-sm" onclick="deleteRagSource('${escapeHtml(s.domain)}')">Delete</button>
    </div>
  `).join('')}</div>`;
}

function updateSearchDomainDropdown(sources) {
  const sel = document.getElementById('rag-search-domain');
  if (!sel) return;
  const current = sel.value;
  sel.innerHTML = '<option value="">All domains</option>';
  sources.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s.domain;
    opt.textContent = s.domain;
    sel.appendChild(opt);
  });
  sel.value = current;
}

// ── Jobs ─────────────────────────────────────────────────────

async function fetchRagJobs() {
  const el = document.getElementById('rag-jobs');
  if (!el) return;
  try {
    const r = await fetch('/web/jobs');
    const data = await r.json();
    renderRagJobs(data.jobs || []);
  } catch (e) {
    el.innerHTML = `<p class="muted">Jobs unavailable: ${e.message}</p>`;
  }
}

function _formatEta(seconds) {
  if (!seconds || seconds <= 0) return '';
  if (seconds < 60) return `~${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m < 60) return `~${m}m${s > 0 ? ` ${s}s` : ''}`;
  const h = Math.floor(m / 60);
  return `~${h}h ${m % 60}m`;
}

function renderRagJobs(jobs) {
  const el = document.getElementById('rag-jobs');
  if (!el) return;
  if (!jobs.length) {
    el.innerHTML = '<p class="muted">No crawl jobs yet.</p>';
    return;
  }
  const sorted = [...jobs].reverse();
  el.innerHTML = `<div class="rag-jobs-list">${sorted.map(j => {
    const statusClass = j.status === 'completed' ? 'active'
      : j.status === 'running' ? 'running'
      : j.status === 'failed' ? 'failed' : 'inactive';
    const eta = j.eta_seconds ? _formatEta(j.eta_seconds) : '';
    const queuePos = j.queue_position && j.status === 'queued' ? ` (queue #${j.queue_position})` : '';
    const scraped = j.pages_scraped || 0;
    const visited = j.pages_visited || j.pages_indexed;
    return `
      <div class="rag-job-row glass">
        <div class="rag-job-info">
          <span class="rag-job-url">${escapeHtml(j.url)}</span>
          <span class="rag-source-meta">depth ${j.max_depth} &middot; max ${j.max_pages.toLocaleString()} pages${queuePos}</span>
        </div>
        <div class="rag-job-stats">
          <span>${j.pages_indexed} indexed / ${scraped} scraped / ${visited} visited</span>
          <span>${j.chunks_indexed} chunks</span>
          ${eta ? `<span class="rag-eta">ETA: ${eta}</span>` : ''}
        </div>
        <div class="rag-job-actions">
          <button class="btn-logs" onclick="viewCrawlLogs('${j.job_id}')" title="View logs">Logs</button>
          <span class="rag-status-badge ${statusClass}">${escapeHtml(j.status)}</span>
        </div>
      </div>`;
  }).join('')}</div>`;
}

// ── Crawl ────────────────────────────────────────────────────

async function startRagCrawl() {
  const urlInput = document.getElementById('rag-url');
  const depthSelect = document.getElementById('rag-depth');
  const maxPagesInput = document.getElementById('rag-max-pages');
  const btn = document.getElementById('rag-crawl-btn');
  const progressWrap = document.getElementById('rag-crawl-progress');
  const progressBar = document.getElementById('rag-progress-bar');
  const infoEl = document.getElementById('rag-crawl-info');

  const url = urlInput.value.trim();
  if (!url) { urlInput.focus(); return; }

  const maxPages = parseInt(maxPagesInput.value);
  if (maxPages > 5000) {
    const mins = Math.round(maxPages / 60);
    const hrs = Math.floor(mins / 60);
    const est = hrs > 0 ? `~${hrs}h ${mins % 60}m` : `~${mins}m`;
    if (!confirm(`Crawling ${maxPages.toLocaleString()} pages will take approximately ${est}. Continue?`)) {
      return;
    }
  }

  btn.disabled = true;
  btn.textContent = 'Starting...';
  progressWrap.style.display = 'block';
  progressBar.style.width = '0%';
  infoEl.textContent = 'Queuing crawl job...';

  try {
    const r = await fetch('/web/index-url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url,
        max_depth: parseInt(depthSelect.value),
        max_pages: parseInt(maxPagesInput.value),
        smart_mode: document.getElementById('rag-smart-mode').checked,
      }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Failed to start crawl');

    btn.textContent = 'Crawling...';
    attachCrawlStream(data.job_id);
  } catch (e) {
    infoEl.textContent = `Error: ${e.message}`;
    btn.disabled = false;
    btn.textContent = 'Start Crawling';
  }
}

function attachCrawlStream(jobId) {
  if (_ragCrawlEventSource) {
    _ragCrawlEventSource.close();
  }

  const progressBar = document.getElementById('rag-progress-bar');
  const infoEl = document.getElementById('rag-crawl-info');
  const btn = document.getElementById('rag-crawl-btn');

  const es = new EventSource(`/web/status/${jobId}`);
  _ragCrawlEventSource = es;

  es.onmessage = (event) => {
    try {
      const job = JSON.parse(event.data);
      const scraped = job.pages_scraped || 0;
      const pct = job.max_pages > 0
        ? Math.min(100, Math.round((scraped / job.max_pages) * 100))
        : 0;

      progressBar.style.width = `${pct}%`;

      if (job.analysis_status === 'analyzing') {
        infoEl.textContent = 'Analyzing site structure with AI...';
      } else {
        const smartTag = job.analysis_status === 'done' ? ' [Smart]' : '';
        const eta = job.eta_seconds ? ` — ETA: ${_formatEta(job.eta_seconds)}` : '';
        infoEl.textContent = `${job.status}${smartTag} — ${scraped} scraped, ${job.pages_indexed} indexed, ${job.chunks_indexed} chunks${eta}`
          + (job.current_url ? ` — ${job.current_url}` : '');
      }

      if (job.status === 'completed' || job.status === 'failed') {
        es.close();
        _ragCrawlEventSource = null;
        btn.disabled = false;
        btn.textContent = 'Start Crawling';
        if (job.status === 'completed') {
          progressBar.style.width = '100%';
          infoEl.textContent = `Completed: ${job.pages_indexed} pages, ${job.chunks_indexed} chunks indexed (${job.pages_visited} visited)`;
        } else {
          infoEl.textContent = `Failed: ${job.errors?.[job.errors.length - 1] || 'Unknown error'}`;
        }
        fetchRagCollections();
        fetchRagSources();
        fetchRagJobs();
      }
    } catch (e) {
      // ignore parse errors
    }
  };

  es.onerror = () => {
    es.close();
    _ragCrawlEventSource = null;
    btn.disabled = false;
    btn.textContent = 'Start Crawling';
    infoEl.textContent = 'Connection lost. Check jobs list for status.';
    fetchRagJobs();
  };
}

// ── Source Management ────────────────────────────────────────

async function deleteRagSource(domain) {
  if (!confirm(`Delete all indexed content from ${domain}?`)) return;
  try {
    const r = await fetch(`/web/source/${domain}`, { method: 'DELETE' });
    if (!r.ok) throw new Error('Delete failed');
    fetchRagCollections();
    fetchRagSources();
  } catch (e) {
    alert(`Failed to delete: ${e.message}`);
  }
}

// ── Search ───────────────────────────────────────────────────

async function searchRag() {
  const queryInput = document.getElementById('rag-search-query');
  const domainSelect = document.getElementById('rag-search-domain');
  const topKInput = document.getElementById('rag-search-topk');
  const resultsEl = document.getElementById('rag-results');

  const query = queryInput.value.trim();
  if (!query) { queryInput.focus(); return; }

  resultsEl.innerHTML = '<p class="muted">Searching...</p>';

  try {
    const body = {
      query,
      top_k: parseInt(topKInput.value) || 5,
    };
    const domain = domainSelect.value;
    if (domain) body.domain = domain;

    const r = await fetch('/web/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    renderRagResults(data.results || [], data.query);
  } catch (e) {
    resultsEl.innerHTML = `<p class="muted">Search failed: ${e.message}</p>`;
  }
}

function renderRagResults(results, query) {
  const el = document.getElementById('rag-results');
  if (!el) return;
  if (!results.length) {
    el.innerHTML = `<p class="muted">No results found for "${escapeHtml(query)}".</p>`;
    return;
  }
  el.innerHTML = results.map((r, i) => `
    <div class="rag-result-card glass">
      <div class="rag-result-header">
        <span class="rag-result-rank">#${i + 1}</span>
        <a href="${escapeHtml(r.url)}" target="_blank" rel="noopener" class="rag-result-url">${escapeHtml(r.url)}</a>
        <span class="rag-result-score">${(r.score * 100).toFixed(1)}%</span>
      </div>
      ${r.title ? `<div class="rag-result-title">${escapeHtml(r.title)}</div>` : ''}
      <div class="rag-result-text">${escapeHtml(r.text.length > 400 ? r.text.slice(0, 400) + '...' : r.text)}</div>
      <div class="rag-result-meta">${escapeHtml(r.domain)} &middot; chunk #${r.chunk_idx} &middot; ${r.fetch_date ? new Date(r.fetch_date).toLocaleDateString() : ''}</div>
    </div>
  `).join('');
}

// ── Logs Modal ───────────────────────────────────────────────

let _ragLogsInterval = null;
let _ragLogsJobId = null;
let _ragLogsOffset = 0;

async function viewCrawlLogs(jobId) {
  _ragLogsJobId = jobId;
  _ragLogsOffset = 0;

  const modal = document.getElementById('rag-logs-modal');
  const logEl = document.getElementById('rag-logs-content');
  const stateEl = document.getElementById('rag-logs-state');
  const idEl = document.getElementById('rag-logs-id');
  const subtitleEl = document.getElementById('rag-logs-subtitle');

  const jobsResp = await fetch('/web/jobs');
  const jobsData = await jobsResp.json();
  const job = (jobsData.jobs || []).find(j => j.job_id === jobId);
  if (job) {
    subtitleEl.textContent = job.url;
    stateEl.textContent = job.status;
    stateEl.className = `op-state op-state-${job.status === 'completed' ? 'success' : job.status === 'failed' ? 'error' : 'running'}`;
  }
  idEl.textContent = `job: ${jobId}`;
  logEl.textContent = '';

  modal.classList.remove('hidden');
  modal.setAttribute('aria-hidden', 'false');

  await _fetchRagLogs();
  _ragLogsInterval = setInterval(_fetchRagLogs, 2000);
}

async function _fetchRagLogs() {
  if (!_ragLogsJobId) return;
  try {
    const r = await fetch(`/web/logs/${_ragLogsJobId}?offset=${_ragLogsOffset}`);
    const data = await r.json();
    const logEl = document.getElementById('rag-logs-content');
    const stateEl = document.getElementById('rag-logs-state');

    if (data.logs && data.logs.length) {
      logEl.textContent += data.logs.join('\n') + '\n';
      _ragLogsOffset = data.total;
      logEl.scrollTop = logEl.scrollHeight;
    }

    if (data.status) {
      stateEl.textContent = data.status;
      stateEl.className = `op-state op-state-${data.status === 'completed' ? 'success' : data.status === 'failed' ? 'error' : 'running'}`;
    }

    if (data.status === 'completed' || data.status === 'failed') {
      if (_ragLogsInterval) {
        clearInterval(_ragLogsInterval);
        _ragLogsInterval = null;
      }
    }
  } catch (e) {
    // ignore fetch errors during polling
  }
}

function closeRagLogs() {
  const modal = document.getElementById('rag-logs-modal');
  modal.classList.add('hidden');
  modal.setAttribute('aria-hidden', 'true');
  if (_ragLogsInterval) {
    clearInterval(_ragLogsInterval);
    _ragLogsInterval = null;
  }
  _ragLogsJobId = null;
}

function copyRagLogs() {
  const logEl = document.getElementById('rag-logs-content');
  navigator.clipboard.writeText(logEl.textContent).then(() => {
    const btn = document.getElementById('rag-logs-copy');
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = 'Copy Logs'; }, 1500);
  });
}

// Close modal on backdrop click
document.addEventListener('click', (e) => {
  if (e.target.id === 'rag-logs-backdrop') closeRagLogs();
});

// ── Auto-load on page init ───────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  fetchRagCollections();
  fetchRagSources();
  fetchRagJobs();
});
