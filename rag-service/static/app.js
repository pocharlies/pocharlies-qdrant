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

// ── Lazy-load dropdown helper ─────────────────────────────────

var _lazyLoaded = {};

function lazyLoad(detailsEl, fetchFn) {
  if (!detailsEl.open) return;
  var key = detailsEl.getAttribute('data-lazy');
  if (_lazyLoaded[key]) return;
  _lazyLoaded[key] = true;
  var hint = detailsEl.querySelector('.lazy-hint');
  if (hint) hint.style.display = 'none';
  fetchFn();
}

// Reset lazy flag so next open re-fetches
function resetLazy(key) {
  delete _lazyLoaded[key];
}

// ── Tab-based lazy loading ───────────────────────────────────

var _tabLoaded = {};

var _agentPollInterval = null;

function onTabActivated(tabId) {
  if (_tabLoaded[tabId]) return;
  _tabLoaded[tabId] = true;
  if (tabId === 'sources') {
    fetchJiraStatus();
    fetchConfluenceStatus();
    fetchProductStats();
  } else if (tabId === 'search') {
    // no-op — search tab is ready, domains load via lazy dropdown
  } else if (tabId === 'ai') {
    checkAgentStatus();
    fetchAgentTasks();
    loadGlossary();
    if (!_agentPollInterval) {
      _agentPollInterval = setInterval(fetchAgentTasks, 10000);
    }
  }
}

// ── Qdrant Health ───────────────────────────────────────────

async function fetchQdrantHealth() {
  const badge = document.getElementById('qdrant-health');
  const details = document.getElementById('qdrant-health-details');
  if (!badge || !details) return;
  try {
    const r = await fetch('/health');
    const data = await r.json();
    const q = data.qdrant || {};
    if (q.status === 'healthy') {
      badge.textContent = 'healthy';
      badge.className = 'qdrant-health-badge healthy';
      const ver = q.version ? `v${q.version}` : '';
      const count = q.collections_count || (q.collections && typeof q.collections === 'object' ? Object.keys(q.collections).length : q.collections) || 0;
      details.textContent = `${q.url} ${ver} \u2014 ${count} collection${count !== 1 ? 's' : ''}`;
    } else {
      badge.textContent = 'unreachable';
      badge.className = 'qdrant-health-badge unhealthy';
      details.textContent = `Cannot reach ${q.url || 'Qdrant'}`;
    }
  } catch (e) {
    badge.textContent = 'error';
    badge.className = 'qdrant-health-badge unhealthy';
    details.textContent = `Health check failed: ${e.message}`;
  }
}

// ── Collections ──────────────────────────────────────────────

let _ragCrawlEventSource = null;

async function fetchRagCollections() {
  const el = document.getElementById('rag-collections');
  if (!el) return;
  try {
    // Use /health which includes ALL collections (web, products, jira, devops, etc.)
    const r = await fetch('/health');
    const data = await r.json();
    var colls = data.qdrant && data.qdrant.collections ? data.qdrant.collections : {};
    var arr = Object.values(colls);
    renderRagCollections(arr);
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
      : j.status === 'stopped' ? 'stopped'
      : j.status === 'failed' ? 'failed' : 'inactive';
    const eta = j.eta_seconds ? _formatEta(j.eta_seconds) : '';
    const queuePos = j.queue_position && j.status === 'queued' ? ` (queue #${j.queue_position})` : '';
    const scraped = j.pages_scraped || 0;
    const visited = j.pages_visited || j.pages_indexed;
    const resumeNote = j.resumed_from ? ` (resumed)` : '';
    return `
      <div class="rag-job-row glass">
        <div class="rag-job-info">
          <span class="rag-job-url">${escapeHtml(j.url)}</span>
          <span class="rag-source-meta">depth ${j.max_depth} &middot; max ${j.max_pages.toLocaleString()} pages${queuePos}${resumeNote}</span>
        </div>
        <div class="rag-job-stats">
          <span>${j.pages_indexed} indexed / ${scraped} scraped / ${visited} visited</span>
          <span>${j.chunks_indexed} chunks</span>
          ${eta ? `<span class="rag-eta">ETA: ${eta}</span>` : ''}
        </div>
        <div class="rag-job-actions">
          ${j.status === 'running' || j.status === 'queued'
            ? `<button class="btn-stop" onclick="stopCrawlJob('${j.job_id}')" title="Stop crawl">Stop</button>`
            : ''}
          ${j.can_resume
            ? `<button class="btn-resume" onclick="resumeCrawlJob('${j.job_id}')" title="Resume crawl">Resume</button>`
            : ''}
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

      if (job.status === 'completed' || job.status === 'failed' || job.status === 'stopped') {
        es.close();
        _ragCrawlEventSource = null;
        btn.disabled = false;
        btn.textContent = 'Start Crawling';
        if (job.status === 'completed') {
          progressBar.style.width = '100%';
          infoEl.textContent = `Completed: ${job.pages_indexed} pages, ${job.chunks_indexed} chunks indexed (${job.pages_visited} visited)`;
        } else if (job.status === 'stopped') {
          infoEl.textContent = `Stopped: ${job.pages_indexed} pages indexed so far. Job can be resumed.`;
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

// ── Stop / Resume ───────────────────────────────────────────

async function stopCrawlJob(jobId) {
  if (!confirm('Stop this crawl job? You can resume it later.')) return;
  try {
    const r = await fetch(`/web/jobs/${jobId}/stop`, { method: 'POST' });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Stop failed');
    fetchRagJobs();
  } catch (e) {
    alert(`Failed to stop job: ${e.message}`);
  }
}

async function resumeCrawlJob(jobId) {
  try {
    const r = await fetch(`/web/jobs/${jobId}/resume`, { method: 'POST' });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Resume failed');
    attachCrawlStream(data.job_id);
    const progressWrap = document.getElementById('rag-crawl-progress');
    const infoEl = document.getElementById('rag-crawl-info');
    const btn = document.getElementById('rag-crawl-btn');
    progressWrap.style.display = 'block';
    infoEl.textContent = `Resuming from job ${jobId}...`;
    btn.disabled = true;
    btn.textContent = 'Crawling...';
    fetchRagJobs();
  } catch (e) {
    alert(`Failed to resume job: ${e.message}`);
  }
}

// ── Search ───────────────────────────────────────────────────

var _searchFiltersLoaded = {};
function onSearchCollectionChange(collection) {
  var filtersRow = document.getElementById('search-filters');
  var filters = filtersRow.querySelectorAll('.search-filter');
  var anyVisible = false;

  filters.forEach(function(el) {
    var cols = (el.getAttribute('data-collections') || '').split(',');
    if (cols.indexOf(collection) !== -1) {
      el.style.display = '';
      anyVisible = true;
    } else {
      el.style.display = 'none';
    }
  });

  filtersRow.style.display = anyVisible ? 'flex' : 'none';

  // Lazy-load filter options
  if (collection === 'jira' && !_searchFiltersLoaded.jira) {
    _searchFiltersLoaded.jira = true;
    loadJiraSearchFilters();
  }
  if ((collection === 'web' || collection === 'competitors') && !_searchFiltersLoaded.domains) {
    _searchFiltersLoaded.domains = true;
    loadDomainFilters();
  }
  if (collection === 'confluence' && !_searchFiltersLoaded.confluence) {
    _searchFiltersLoaded.confluence = true;
    loadConfluenceSearchFilters();
  }
}

async function loadJiraSearchFilters() {
  try {
    var r = await fetch('/jira/filters');
    var d = await r.json();
    var projSel = document.getElementById('rag-search-jira-project');
    if (projSel && d.projects) {
      d.projects.forEach(function(p) {
        var opt = document.createElement('option');
        opt.value = p; opt.textContent = p;
        projSel.appendChild(opt);
      });
    }
    var assignSel = document.getElementById('rag-search-jira-assignee');
    if (assignSel && d.assignees) {
      d.assignees.forEach(function(a) {
        var opt = document.createElement('option');
        opt.value = a; opt.textContent = a;
        assignSel.appendChild(opt);
      });
    }
  } catch (e) { console.warn('Failed to load Jira search filters:', e); }
}

async function loadDomainFilters() {
  try {
    var r = await fetch('/web/sources');
    var d = await r.json();
    var sel = document.getElementById('rag-search-domain');
    if (sel && d.sources) {
      var domains = {};
      d.sources.forEach(function(s) {
        if (s.domain && !domains[s.domain]) {
          domains[s.domain] = true;
          var opt = document.createElement('option');
          opt.value = s.domain; opt.textContent = s.domain;
          sel.appendChild(opt);
        }
      });
    }
  } catch (e) { console.warn('Failed to load domain filters:', e); }
}

async function searchRag() {
  const queryInput = document.getElementById('rag-search-query');
  const collectionSelect = document.getElementById('rag-search-collection');
  const topKInput = document.getElementById('rag-search-topk');
  const resultsEl = document.getElementById('rag-results');

  const query = queryInput.value.trim();
  if (!query) { queryInput.focus(); return; }

  const collection = collectionSelect.value || 'all';
  resultsEl.innerHTML = '<p class="muted">Searching...</p>';

  try {
    const body = {
      query,
      top_k: parseInt(topKInput.value) || 10,
      collection: collection,
    };

    // Collection-specific filters
    var domain = document.getElementById('rag-search-domain').value;
    if (domain) body.domain = domain;
    var repo = document.getElementById('rag-search-repo').value.trim();
    if (repo) body.repo = repo;
    var brand = document.getElementById('rag-search-brand').value.trim();
    if (brand) body.brand = brand;
    var category = document.getElementById('rag-search-category').value.trim();
    if (category) body.category = category;
    var jiraProject = document.getElementById('rag-search-jira-project').value;
    if (jiraProject) body.project = jiraProject;
    var jiraAssignee = document.getElementById('rag-search-jira-assignee').value;
    if (jiraAssignee) body.assignee = jiraAssignee;
    var jiraType = document.getElementById('rag-search-jira-type').value;
    if (jiraType) body.issue_type = jiraType;
    var confSpace = document.getElementById('rag-search-confluence-space').value;
    if (confSpace) body.space = confSpace;
    var confAuthor = document.getElementById('rag-search-confluence-author').value;
    if (confAuthor) body.author = confAuthor;
    var confDept = document.getElementById('rag-search-confluence-department').value;
    if (confDept) body.department = confDept;
    var confTopic = document.getElementById('rag-search-confluence-topic').value;
    if (confTopic) body.topic = confTopic;

    const r = await fetch('/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    renderRagResults(data.results || [], data.query, data.collection);
  } catch (e) {
    resultsEl.innerHTML = `<p class="muted">Search failed: ${e.message}</p>`;
  }
}

function renderRagResults(results, query, collection) {
  const el = document.getElementById('rag-results');
  if (!el) return;
  if (!results.length) {
    el.innerHTML = `<p class="muted">No results found for "${escapeHtml(query)}".</p>`;
    return;
  }
  el.innerHTML = results.map((r, i) => {
    const isCode = r.source_type === 'code';
    const badgeClass = isCode ? 'code' : 'web';
    const badgeText = isCode ? 'CODE' : 'WEB';
    const scoreDisplay = `${(r.score * 100).toFixed(1)}%`;

    if (isCode) {
      const symbols = (r.symbols && r.symbols.length) ? r.symbols.join(', ') : '';
      const path = `${escapeHtml(r.repo)}/${escapeHtml(r.path)}`;
      const lines = `L${r.start_line}-${r.end_line}`;
      return `<div class="rag-result-card glass">
        <div class="rag-result-header">
          <span class="rag-result-rank">#${i + 1}</span>
          <span class="collection-badge ${badgeClass}">${badgeText}</span>
          <span class="rag-result-path">${path}#${lines}</span>
          <span class="rag-result-score">${scoreDisplay}</span>
        </div>
        <pre class="rag-result-code">${escapeHtml(r.text.length > 600 ? r.text.slice(0, 600) + '...' : r.text)}</pre>
        <div class="rag-result-meta">${escapeHtml(r.repo)} &middot; ${lines}${symbols ? ' &middot; ' + escapeHtml(symbols) : ''}</div>
      </div>`;
    } else {
      return `<div class="rag-result-card glass">
        <div class="rag-result-header">
          <span class="rag-result-rank">#${i + 1}</span>
          <span class="collection-badge ${badgeClass}">${badgeText}</span>
          <a href="${escapeHtml(r.url || '')}" target="_blank" rel="noopener" class="rag-result-url">${escapeHtml(r.url || '')}</a>
          <span class="rag-result-score">${scoreDisplay}</span>
        </div>
        ${r.title ? `<div class="rag-result-title">${escapeHtml(r.title)}</div>` : ''}
        <div class="rag-result-text">${escapeHtml(r.text.length > 400 ? r.text.slice(0, 400) + '...' : r.text)}</div>
        <div class="rag-result-meta">${escapeHtml(r.domain || '')} &middot; chunk #${r.chunk_idx || 0} &middot; ${r.fetch_date ? new Date(r.fetch_date).toLocaleDateString() : ''}</div>
      </div>`;
    }
  }).join('');
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
    stateEl.className = `op-state op-state-${job.status === 'completed' ? 'success' : job.status === 'failed' ? 'error' : job.status === 'stopped' ? 'warning' : 'running'}`;
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
      stateEl.className = `op-state op-state-${data.status === 'completed' ? 'success' : data.status === 'failed' ? 'error' : data.status === 'stopped' ? 'warning' : 'running'}`;
    }

    if (data.status === 'completed' || data.status === 'failed' || data.status === 'stopped') {
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

// ── Agent ─────────────────────────────────────────────────────

let _agentEventSource = null;
let _agentModalTaskId = null;
let _agentLogsOffset = 0;
let _agentLogsInterval = null;

async function checkAgentStatus() {
  const bar = document.getElementById('agent-status-bar');
  const form = document.getElementById('agent-form');
  try {
    const r = await fetch('/agent/status');
    const d = await r.json();
    if (d.available) {
      bar.innerHTML = `<span class="agent-status-dot online"></span><span class="muted">LLM connected: <strong>${escapeHtml(d.model_id)}</strong></span>`;
      form.style.display = 'block';
    } else {
      bar.innerHTML = `<span class="agent-status-dot offline"></span><span class="muted">${escapeHtml(d.reason)}</span>`;
      form.style.display = 'none';
    }
  } catch (e) {
    bar.innerHTML = `<span class="agent-status-dot offline"></span><span class="muted">Agent unavailable</span>`;
    form.style.display = 'none';
  }
}

async function fetchAgentTasks() {
  try {
    const r = await fetch('/agent/tasks');
    const d = await r.json();
    renderAgentTasks(d.tasks || []);
  } catch (e) { /* silent */ }
}

function renderAgentTasks(tasks) {
  const el = document.getElementById('agent-tasks-list');
  if (!el) return;
  if (!tasks.length) { el.textContent = ''; return; }
  // Note: all user-supplied values go through escapeHtml() before insertion
  el.innerHTML = tasks.map(t => {
    const sc = t.status === 'completed' ? 'active'
      : t.status === 'running' ? 'running'
      : t.status === 'failed' || t.status === 'cancelled' ? 'failed' : 'inactive';
    const tools = (t.tools_called || []).length ? t.tools_called.join(', ') : 'none';
    const started = t.started_at ? new Date(t.started_at).toLocaleTimeString() : '';
    const srcLabel = t.source === 'cli' ? 'CLI' : 'WEB';
    const srcColor = t.source === 'cli' ? '#6366f1' : '#22c55e';
    return `<div class="agent-task-row">
      <div class="agent-task-info">
        <span class="agent-task-prompt"><span style="background:${srcColor};color:#fff;font-size:10px;padding:1px 6px;border-radius:3px;margin-right:4px">${srcLabel}</span>${escapeHtml(t.prompt)}</span>
        <div class="agent-task-meta">
          <span>${escapeHtml(started)}</span>
          <span class="agent-task-tools">tools: ${escapeHtml(tools)}</span>
          <span>${parseInt(t.log_count || t.step_count || 0)} logs</span>
        </div>
      </div>
      <div class="rag-job-actions">
        <button class="btn-logs" onclick="viewAgentTask('${escapeHtml(t.task_id)}')">Details</button>
        <span class="rag-status-badge ${sc}">${escapeHtml(t.status)}</span>
      </div>
    </div>`;
  }).join('');
}

async function startAgentTask() {
  const promptEl = document.getElementById('agent-prompt');
  const btn = document.getElementById('agent-run-btn');
  const prompt = promptEl.value.trim();
  if (!prompt) { promptEl.focus(); return; }
  btn.disabled = true;
  btn.textContent = 'Starting...';
  try {
    const r = await fetch('/agent/task', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Failed');
    promptEl.value = '';
    viewAgentTask(d.task_id);
    fetchAgentTasks();
  } catch (e) {
    alert('Error: ' + e.message);
  }
  btn.disabled = false;
  btn.textContent = 'Run Agent';
}

function _stepIcon(type) {
  const m = { thinking: 'T', tool_call: '>', tool_result: '<', response: 'R', user_message: 'U' };
  return m[type] || '?';
}

function renderAgentSteps(steps) {
  const el = document.getElementById('agent-steps');
  if (!el) return;
  if (!steps.length) { el.innerHTML = '<p class="muted" style="padding:4px">No steps yet...</p>'; return; }
  el.innerHTML = steps.map(s => {
    const txt = s.content.length > 300 ? s.content.slice(0, 300) + '...' : s.content;
    return `<div class="agent-step">
      <span class="agent-step-icon ${escapeHtml(s.type)}">${_stepIcon(s.type)}</span>
      <span class="agent-step-content">${escapeHtml(txt)}</span>
    </div>`;
  }).join('');
  el.scrollTop = el.scrollHeight;
}

function _updateAgentModal(t) {
  document.getElementById('agent-modal-subtitle').textContent =
    t.prompt && t.prompt.length > 80 ? t.prompt.slice(0, 80) + '...' : (t.prompt || '');
  document.getElementById('agent-modal-id').textContent = 'task: ' + (t.task_id || '-');
  document.getElementById('agent-modal-model').textContent = t.model_id ? 'model: ' + t.model_id : '';

  const sc = t.status === 'completed' ? 'success'
    : (t.status === 'failed' || t.status === 'cancelled') ? 'error' : 'running';
  const stateEl = document.getElementById('agent-modal-state');
  stateEl.textContent = t.status;
  stateEl.className = 'op-state op-state-' + sc;

  renderAgentSteps(t.steps || []);

  // Show message form for all task states (running = queued, completed/failed = continue)
  const resumeForm = document.getElementById('agent-resume-form');
  if (resumeForm) {
    resumeForm.style.display = 'flex';
    const btn = document.getElementById('agent-resume-btn');
    if (btn) {
      btn.textContent = t.status === 'running' ? 'Send' : 'Continue';
    }
    const input = document.getElementById('agent-resume-input');
    if (input) {
      input.placeholder = t.status === 'running'
        ? 'Send message to running agent (will be queued)...'
        : 'Send follow-up message to continue...';
    }
  }
}

async function viewAgentTask(taskId) {
  _agentModalTaskId = taskId;
  _agentLogsOffset = 0;

  const modal = document.getElementById('agent-modal');
  const logsEl = document.getElementById('agent-logs-content');
  logsEl.textContent = '';
  modal.classList.remove('hidden');
  modal.setAttribute('aria-hidden', 'false');

  // First, fetch current state from API (works for both in-memory and Redis-persisted tasks)
  try {
    const r = await fetch('/agent/task/' + taskId);
    if (r.ok) {
      const t = await r.json();
      _updateAgentModal(t);

      // If not running, just show the historical data — no SSE needed
      if (t.status !== 'running') {
        await _fetchAgentLogs();
        return;
      }
    }
  } catch (e) { /* continue to SSE */ }

  // Connect SSE for running tasks
  if (_agentEventSource) _agentEventSource.close();
  const es = new EventSource('/agent/task/' + taskId + '/stream');
  _agentEventSource = es;

  es.onmessage = (event) => {
    try {
      const t = JSON.parse(event.data);
      _updateAgentModal(t);

      if (t.status !== 'running') {
        es.close();
        _agentEventSource = null;
        fetchAgentTasks();
        _fetchAgentLogs();
      }
    } catch (e) { /* ignore */ }
  };

  es.onerror = () => { es.close(); _agentEventSource = null; };

  await _fetchAgentLogs();
  _agentLogsInterval = setInterval(_fetchAgentLogs, 2000);
}

async function _fetchAgentLogs() {
  if (!_agentModalTaskId) return;
  try {
    const r = await fetch('/agent/task/' + _agentModalTaskId + '/logs?offset=' + _agentLogsOffset);
    const d = await r.json();
    const logsEl = document.getElementById('agent-logs-content');
    if (d.logs && d.logs.length) {
      logsEl.textContent += d.logs.join('\n') + '\n';
      _agentLogsOffset = d.total;
      logsEl.scrollTop = logsEl.scrollHeight;
    }
    if (d.status === 'completed' || d.status === 'failed' || d.status === 'cancelled') {
      if (_agentLogsInterval) { clearInterval(_agentLogsInterval); _agentLogsInterval = null; }
    }
  } catch (e) { /* ignore */ }
}

function closeAgentModal() {
  document.getElementById('agent-modal').classList.add('hidden');
  document.getElementById('agent-modal').setAttribute('aria-hidden', 'true');
  if (_agentEventSource) { _agentEventSource.close(); _agentEventSource = null; }
  if (_agentLogsInterval) { clearInterval(_agentLogsInterval); _agentLogsInterval = null; }
  _agentModalTaskId = null;
}

async function continueAgentTask() {
  const input = document.getElementById('agent-resume-input');
  const btn = document.getElementById('agent-resume-btn');
  const message = input.value.trim();
  if (!message) { input.focus(); return; }

  const originalText = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Sending...';

  try {
    const r = await fetch('/agent/task/' + _agentModalTaskId + '/message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Failed');

    input.value = '';

    if (d.status === 'queued') {
      // Message queued for running task — show confirmation, keep form visible
      btn.textContent = 'Queued!';
      setTimeout(() => { btn.textContent = 'Send'; btn.disabled = false; }, 1500);
      return;
    }

    // Task was continued (completed/failed → running) — reconnect SSE
    viewAgentTask(_agentModalTaskId);
    fetchAgentTasks();
  } catch (e) {
    alert('Error: ' + e.message);
  }
  btn.disabled = false;
  btn.textContent = originalText;
}

function copyAgentLogs() {
  const el = document.getElementById('agent-logs-content');
  navigator.clipboard.writeText(el.textContent).then(() => {
    const btn = document.getElementById('agent-copy-btn');
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = 'Copy Logs'; }, 1500);
  });
}

document.addEventListener('click', (e) => {
  if (e.target.id === 'agent-modal-backdrop') closeAgentModal();
});

// ── Product Catalog ──────────────────────────────────────────

async function fetchProductStats() {
  const el = document.getElementById('product-stats');
  if (!el) return;
  try {
    const r = await fetch('/products/stats');
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Failed');
    const chips = [];
    if (d.total_products != null) chips.push('<strong>' + d.total_products.toLocaleString() + '</strong> products');
    if (d.total_indexed != null) chips.push('<strong>' + d.total_indexed.toLocaleString() + '</strong> indexed');
    if (d.last_sync) chips.push('Last sync: ' + new Date(d.last_sync).toLocaleString());
    el.textContent = '';
    const wrap = document.createElement('div');
    wrap.className = 'stat-chips';
    chips.forEach(function(html) {
      const span = document.createElement('span');
      span.className = 'stat-chip';
      span.innerHTML = html;
      wrap.appendChild(span);
    });
    el.appendChild(wrap);
  } catch (_e) {
    el.textContent = 'Product catalog not synced yet.';
    el.className = 'product-stats-bar muted';
  }
}

let _productSyncInterval = null;

async function syncProducts(syncType) {
  const statusEl = document.getElementById('product-sync-status');
  const barEl = document.getElementById('product-sync-bar');
  const infoEl = document.getElementById('product-sync-info');
  statusEl.style.display = 'block';
  barEl.style.width = '0%';
  infoEl.textContent = 'Starting ' + syncType + ' sync...';
  try {
    const r = await fetch('/products/sync', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sync_type: syncType }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Sync failed');
    _pollProductSync(d.job_id);
  } catch (e) {
    infoEl.textContent = 'Error: ' + e.message;
  }
}

function _pollProductSync(jobId) {
  if (_productSyncInterval) clearInterval(_productSyncInterval);
  _productSyncInterval = setInterval(async function() {
    try {
      const r = await fetch('/products/sync/' + jobId);
      const d = await r.json();
      const barEl = document.getElementById('product-sync-bar');
      const infoEl = document.getElementById('product-sync-info');
      const pct = d.total_products > 0 ? Math.min(100, Math.round((d.products_synced / d.total_products) * 100)) : 0;
      barEl.style.width = pct + '%';
      infoEl.textContent = d.status + ' — ' + (d.products_synced || 0) + ' synced';
      if (d.status === 'completed' || d.status === 'failed') {
        clearInterval(_productSyncInterval);
        _productSyncInterval = null;
        if (d.status === 'completed') {
          barEl.style.width = '100%';
          infoEl.textContent = 'Sync complete: ' + d.products_synced + ' products indexed';
        }
        fetchProductStats();
        fetchRagCollections();
      }
    } catch (_e) { /* ignore */ }
  }, 2000);
}

async function searchProducts() {
  const queryEl = document.getElementById('product-search-query');
  const query = queryEl.value.trim();
  const brand = document.getElementById('product-search-brand').value.trim();
  const category = document.getElementById('product-search-category').value.trim();
  const el = document.getElementById('product-results');
  if (!query) { queryEl.focus(); return; }
  el.textContent = 'Searching...';
  el.className = 'muted';
  try {
    const body = { query: query, top_k: 10, rerank: true };
    if (brand) body.brand = brand;
    if (category) body.category = category;
    const r = await fetch('/products/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Search failed');
    renderProductResults(d.results || []);
  } catch (e) {
    el.textContent = 'Search failed: ' + e.message;
  }
}

function renderProductResults(results) {
  const el = document.getElementById('product-results');
  el.className = '';
  if (!results.length) { el.textContent = 'No products found.'; el.className = 'muted'; return; }
  el.textContent = '';
  results.forEach(function(r, i) {
    const card = document.createElement('div');
    card.className = 'rag-result-card glass';
    const title = r.title || (r.text ? r.text.slice(0, 60) : '');
    const score = (r.score * 100).toFixed(1) + '%';
    let html = '<div class="rag-result-header">'
      + '<span class="rag-result-rank">#' + (i + 1) + '</span>'
      + '<span class="collection-badge product">PRODUCT</span>'
      + '<span class="rag-result-title" style="flex:1">' + escapeHtml(title) + '</span>'
      + '<span class="rag-result-score">' + score + '</span></div>';
    if (r.brand) html += '<div class="rag-result-meta">Brand: ' + escapeHtml(r.brand) + '</div>';
    if (r.price) html += '<div class="product-price">' + escapeHtml(r.currency || '\u20AC') + Number(r.price).toFixed(2) + '</div>';
    if (r.text) html += '<div class="rag-result-text">' + escapeHtml(r.text.slice(0, 300)) + '</div>';
    card.innerHTML = html;
    el.appendChild(card);
  });
}

// ── Competitor Intelligence ─────────────────────────────────

async function fetchCompetitorSources() {
  const el = document.getElementById('competitor-sources');
  const actionsEl = document.getElementById('competitor-actions');
  const selectEl = document.getElementById('competitor-domain-select');
  if (!el) return;
  try {
    const r = await fetch('/competitor/sources');
    const d = await r.json();
    const sources = d.sources || [];
    if (!sources.length) {
      el.textContent = 'No competitor sources indexed yet.';
      el.className = 'muted';
      actionsEl.style.display = 'none';
      return;
    }
    actionsEl.style.display = 'block';
    selectEl.textContent = '';
    var defOpt = document.createElement('option');
    defOpt.value = '';
    defOpt.textContent = 'Select domain...';
    selectEl.appendChild(defOpt);
    sources.forEach(function(s) {
      var opt = document.createElement('option');
      opt.value = s.domain;
      opt.textContent = s.domain + ' (' + (s.url_count || s.page_count || 0) + ' pages)';
      selectEl.appendChild(opt);
    });
    el.textContent = '';
    var list = document.createElement('div');
    list.className = 'rag-sources-list';
    sources.forEach(function(s) {
      var row = document.createElement('div');
      row.className = 'rag-source-row glass';
      row.innerHTML = '<div class="rag-source-info">'
        + '<span class="rag-source-domain">' + escapeHtml(s.domain) + '</span>'
        + '<span class="rag-source-meta">' + (s.url_count || s.page_count || 0) + ' pages &middot; ' + (s.chunk_count || 0) + ' chunks</span>'
        + '</div>';
      var btn = document.createElement('button');
      btn.className = 'btn btn-danger-sm';
      btn.textContent = 'Delete';
      btn.onclick = function() { deleteCompetitorSource(s.domain); };
      row.appendChild(btn);
      list.appendChild(row);
    });
    el.appendChild(list);
  } catch (e) {
    el.textContent = 'Sources unavailable: ' + e.message;
    el.className = 'muted';
  }
}

async function indexCompetitor() {
  var url = document.getElementById('competitor-url').value.trim();
  if (!url) { document.getElementById('competitor-url').focus(); return; }
  var progressEl = document.getElementById('competitor-progress');
  var infoEl = document.getElementById('competitor-progress-info');
  progressEl.style.display = 'block';
  document.getElementById('competitor-progress-bar').style.width = '0%';
  infoEl.textContent = 'Starting competitor crawl...';
  try {
    var r = await fetch('/competitor/index-url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url: url,
        max_depth: parseInt(document.getElementById('competitor-depth').value),
        max_pages: parseInt(document.getElementById('competitor-max-pages').value),
        smart_mode: document.getElementById('competitor-smart').checked,
      }),
    });
    var d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Failed');
    infoEl.textContent = 'Queued as job ' + d.job_id + '. Check Crawl Jobs for progress.';
    attachCrawlStream(d.job_id);
    fetchRagJobs();
  } catch (e) {
    infoEl.textContent = 'Error: ' + e.message;
  }
}

async function deleteCompetitorSource(domain) {
  if (!confirm('Delete all competitor data from ' + domain + '?')) return;
  try {
    var r = await fetch('/competitor/source/' + domain, { method: 'DELETE' });
    if (!r.ok) throw new Error('Delete failed');
    fetchCompetitorSources();
    fetchRagCollections();
  } catch (e) {
    alert('Failed: ' + e.message);
  }
}

var _classifyInterval = null;

async function extractCompetitorProducts() {
  var domain = document.getElementById('competitor-domain-select').value;
  if (!domain) { alert('Select a domain first'); return; }
  var el = document.getElementById('competitor-results');
  el.textContent = 'Extracting products with AI...';
  el.className = 'muted';
  try {
    var r = await fetch('/classify/extract', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ domain: domain }),
    });
    var d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Failed');
    _pollClassify(d.job_id);
  } catch (e) {
    el.textContent = 'Error: ' + e.message;
  }
}

function _pollClassify(jobId) {
  if (_classifyInterval) clearInterval(_classifyInterval);
  var el = document.getElementById('competitor-results');
  _classifyInterval = setInterval(async function() {
    try {
      var r = await fetch('/classify/status/' + jobId);
      var d = await r.json();
      el.textContent = 'Extracting: ' + (d.processed || 0) + '/' + (d.total || '?') + ' pages processed (' + d.status + ')';
      if (d.status === 'completed' || d.status === 'failed') {
        clearInterval(_classifyInterval);
        _classifyInterval = null;
        if (d.status === 'completed') {
          var pr = await fetch('/classify/products/' + jobId);
          var pd = await pr.json();
          renderExtractedProducts(pd.products || []);
        }
      }
    } catch (_e) { /* ignore */ }
  }, 3000);
}

function renderExtractedProducts(products) {
  var el = document.getElementById('competitor-results');
  el.textContent = '';
  el.className = '';
  if (!products.length) { el.textContent = 'No products extracted.'; el.className = 'muted'; return; }
  var header = document.createElement('p');
  header.className = 'muted';
  header.textContent = products.length + ' products extracted';
  header.style.marginBottom = '8px';
  el.appendChild(header);
  products.slice(0, 20).forEach(function(p) {
    var card = document.createElement('div');
    card.className = 'rag-result-card glass';
    var html = '<div class="rag-result-header">'
      + '<span class="collection-badge competitor">COMPETITOR</span>'
      + '<span class="rag-result-title" style="flex:1">' + escapeHtml(p.name || '') + '</span>';
    if (p.price) html += '<span class="product-price">' + escapeHtml(p.currency || '\u20AC') + Number(p.price).toFixed(2) + '</span>';
    html += '</div>';
    if (p.brand) html += '<div class="rag-result-meta">Brand: ' + escapeHtml(p.brand) + '</div>';
    card.innerHTML = html;
    el.appendChild(card);
  });
  if (products.length > 20) {
    var more = document.createElement('p');
    more.className = 'muted';
    more.textContent = '...and ' + (products.length - 20) + ' more';
    el.appendChild(more);
  }
}

async function resolveCompetitorProducts() {
  var domain = document.getElementById('competitor-domain-select').value;
  if (!domain) { alert('Select a domain first'); return; }
  var el = document.getElementById('competitor-results');
  el.textContent = 'Matching competitor products against catalog...';
  el.className = 'muted';
  try {
    var r = await fetch('/classify/resolve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ domain: domain }),
    });
    var d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Failed');
    renderPriceMatches(d.matches || [], d.price_report);
  } catch (e) {
    el.textContent = 'Error: ' + e.message;
  }
}

function renderPriceMatches(matches, report) {
  var el = document.getElementById('competitor-results');
  el.textContent = '';
  el.className = '';
  if (report) {
    var rpt = document.createElement('div');
    rpt.className = 'price-report glass';
    var html = '<strong>Price Report</strong>';
    if (report.avg_diff != null) html += ' — Avg diff: ' + (report.avg_diff > 0 ? '+' : '') + report.avg_diff.toFixed(1) + '%';
    if (report.cheaper_count != null) html += ' <span class="stat-chip">Cheaper: ' + report.cheaper_count + '</span>';
    if (report.pricier_count != null) html += ' <span class="stat-chip">Pricier: ' + report.pricier_count + '</span>';
    rpt.innerHTML = html;
    el.appendChild(rpt);
  }
  if (matches.length) {
    matches.slice(0, 20).forEach(function(m) {
      var card = document.createElement('div');
      card.className = 'rag-result-card glass';
      var h = '<div class="rag-result-header">'
        + '<span class="rag-result-score">' + (m.confidence * 100).toFixed(0) + '%</span>'
        + '<span style="flex:1">' + escapeHtml(m.source && m.source.name || '') + ' &harr; ' + escapeHtml(m.target && m.target.title || '') + '</span></div>';
      if (m.source && m.source.price && m.target && m.target.price) {
        h += '<div class="rag-result-meta">Competitor: \u20AC' + Number(m.source.price).toFixed(2)
          + ' vs Ours: \u20AC' + Number(m.target.price).toFixed(2);
        if (m.price_diff) h += ' (' + (m.price_diff > 0 ? '+' : '') + m.price_diff.toFixed(1) + '%)';
        h += '</div>';
      }
      card.innerHTML = h;
      el.appendChild(card);
    });
  } else {
    var msg = document.createElement('p');
    msg.className = 'muted';
    msg.textContent = 'No matches found. Try extracting products first.';
    el.appendChild(msg);
  }
}

// ── Translation ─────────────────────────────────────────────

async function translateBatch() {
  var input = document.getElementById('translate-input').value.trim();
  if (!input) { document.getElementById('translate-input').focus(); return; }
  var el = document.getElementById('translate-results');
  el.textContent = 'Translating...';
  el.className = 'muted';
  var texts = input.split('\n').filter(function(t) { return t.trim(); });
  try {
    var r = await fetch('/translate/batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        texts: texts,
        source_lang: document.getElementById('translate-source').value,
        target_lang: document.getElementById('translate-target').value,
      }),
    });
    if (!r.ok) {
      var ct = r.headers.get('content-type') || '';
      if (ct.includes('application/json')) {
        var err = await r.json();
        throw new Error(err.detail || 'Server error ' + r.status);
      }
      throw new Error('Server error ' + r.status + ' — the service may be restarting, try again in a moment');
    }
    var d = await r.json();
    if (d.job_id && d.status !== 'completed') {
      _pollTranslation(d.job_id, texts);
    } else {
      renderTranslations(texts, d.translations || []);
    }
  } catch (e) {
    el.textContent = 'Translation failed: ' + e.message;
  }
}

function _pollTranslation(jobId, originals) {
  var el = document.getElementById('translate-results');
  var iv = setInterval(async function() {
    try {
      var r = await fetch('/translate/status/' + jobId);
      var d = await r.json();
      if (d.status === 'completed' || d.status === 'failed') {
        clearInterval(iv);
        if (d.status === 'completed') renderTranslations(originals, d.results || []);
        else { el.textContent = 'Translation failed.'; el.className = 'muted'; }
      } else {
        el.textContent = 'Translating... (' + d.status + ')';
      }
    } catch (_e) { /* ignore */ }
  }, 2000);
}

function renderTranslations(originals, translations) {
  var el = document.getElementById('translate-results');
  el.textContent = '';
  el.className = '';
  translations.forEach(function(t, i) {
    var pair = document.createElement('div');
    pair.className = 'translate-pair';
    var orig = document.createElement('div');
    orig.className = 'translate-original';
    orig.textContent = originals[i] || '';
    var arrow = document.createElement('div');
    arrow.className = 'translate-arrow-sm';
    arrow.textContent = '\u2192';
    var trans = document.createElement('div');
    trans.className = 'translate-translated';
    trans.textContent = t;
    pair.appendChild(orig);
    pair.appendChild(arrow);
    pair.appendChild(trans);
    el.appendChild(pair);
  });
}

// ── Glossary ────────────────────────────────────────────────

var _glossaryData = { builtin: {}, custom: {} };

async function loadGlossary() {
  try {
    var r = await fetch('/glossary');
    var d = await r.json();
    _glossaryData = d;
    var stats = document.getElementById('glossary-stats');
    if (stats) stats.textContent = d.builtin_count + ' built-in + ' + d.custom_count + ' custom terms';
    filterGlossary();
  } catch (e) {
    var el = document.getElementById('glossary-list');
    if (el) el.textContent = 'Failed to load glossary.';
  }
}

function filterGlossary() {
  var el = document.getElementById('glossary-list');
  if (!el) return;
  var search = (document.getElementById('glossary-search').value || '').toLowerCase();
  var showBuiltin = document.getElementById('glossary-show-builtin').checked;
  var showCustom = document.getElementById('glossary-show-custom').checked;

  var rows = [];

  // Custom entries first (editable)
  if (showCustom) {
    Object.keys(_glossaryData.custom || {}).forEach(function(k) {
      var v = _glossaryData.custom[k];
      if (search && k.indexOf(search) === -1 && v.toLowerCase().indexOf(search) === -1) return;
      rows.push({ source: k, target: v, type: 'custom' });
    });
  }

  // Built-in entries
  if (showBuiltin) {
    Object.keys(_glossaryData.builtin || {}).forEach(function(k) {
      var v = _glossaryData.builtin[k];
      if (search && k.indexOf(search) === -1 && v.toLowerCase().indexOf(search) === -1) return;
      // Skip if overridden by custom
      if (_glossaryData.custom && _glossaryData.custom[k]) return;
      rows.push({ source: k, target: v, type: 'builtin' });
    });
  }

  // Sort alphabetically
  rows.sort(function(a, b) { return a.source.localeCompare(b.source); });

  el.innerHTML = '';
  if (!rows.length) {
    el.innerHTML = '<p class="muted">No matching terms.</p>';
    return;
  }

  // Render as compact table
  var table = document.createElement('table');
  table.style.cssText = 'width:100%;border-collapse:collapse;font-size:.85rem';
  rows.forEach(function(row) {
    var tr = document.createElement('tr');
    tr.style.borderBottom = '1px solid rgba(255,255,255,.06)';

    var tdSource = document.createElement('td');
    tdSource.style.cssText = 'padding:4px 8px;color:var(--text)';
    tdSource.textContent = row.source;

    var tdArrow = document.createElement('td');
    tdArrow.style.cssText = 'padding:4px;color:var(--text-muted);text-align:center;width:30px';
    tdArrow.textContent = '\u2192';

    var tdTarget = document.createElement('td');
    tdTarget.style.cssText = 'padding:4px 8px;color:var(--accent)';
    tdTarget.textContent = row.target;

    var tdType = document.createElement('td');
    tdType.style.cssText = 'padding:4px 8px;width:70px;text-align:right';
    if (row.type === 'custom') {
      var badge = document.createElement('span');
      badge.style.cssText = 'background:rgba(168,130,255,.2);color:#a882ff;padding:1px 6px;border-radius:4px;font-size:.75rem';
      badge.textContent = 'custom';
      tdType.appendChild(badge);
    } else {
      var badge2 = document.createElement('span');
      badge2.style.cssText = 'background:rgba(255,255,255,.06);color:var(--text-muted);padding:1px 6px;border-radius:4px;font-size:.75rem';
      badge2.textContent = 'built-in';
      tdType.appendChild(badge2);
    }

    var tdActions = document.createElement('td');
    tdActions.style.cssText = 'padding:4px;width:30px;text-align:center';
    if (row.type === 'custom') {
      var del = document.createElement('button');
      del.style.cssText = 'background:none;border:none;color:#e55;cursor:pointer;font-size:.85rem;padding:2px 4px';
      del.textContent = '\u00d7';
      del.title = 'Delete';
      del.onclick = (function(term) { return function() { deleteGlossaryEntry(term); }; })(row.source);
      tdActions.appendChild(del);
    }

    tr.appendChild(tdSource);
    tr.appendChild(tdArrow);
    tr.appendChild(tdTarget);
    tr.appendChild(tdType);
    tr.appendChild(tdActions);
    table.appendChild(tr);
  });
  el.appendChild(table);
}

async function addGlossaryEntry() {
  var srcEl = document.getElementById('glossary-source');
  var tgtEl = document.getElementById('glossary-target');
  var src = srcEl.value.trim();
  var tgt = tgtEl.value.trim();
  if (!src || !tgt) { srcEl.focus(); return; }
  try {
    var r = await fetch('/glossary', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: src, target: tgt }),
    });
    if (!r.ok) { var e = await r.json(); alert(e.detail || 'Failed'); return; }
    srcEl.value = '';
    tgtEl.value = '';
    await loadGlossary();
  } catch (e) {
    alert('Error: ' + e.message);
  }
}

async function deleteGlossaryEntry(term) {
  try {
    var r = await fetch('/glossary/' + encodeURIComponent(term), { method: 'DELETE' });
    if (!r.ok) { var e = await r.json(); alert(e.detail || 'Failed'); return; }
    await loadGlossary();
  } catch (e) {
    alert('Error: ' + e.message);
  }
}

// ── DevOps Docs ─────────────────────────────────────────────

async function fetchDevopsSources() {
  var el = document.getElementById('devops-sources');
  if (!el) return;
  try {
    var r = await fetch('/devops/sources');
    var d = await r.json();
    var sources = d.sources || [];
    if (!sources.length) {
      el.textContent = 'No DevOps docs indexed.';
      el.className = 'muted';
      return;
    }
    el.textContent = '';
    el.className = '';
    var list = document.createElement('div');
    list.className = 'rag-sources-list';
    sources.forEach(function(s) {
      var row = document.createElement('div');
      row.className = 'rag-source-row glass';
      row.innerHTML = '<div class="rag-source-info">'
        + '<span class="rag-source-domain">' + escapeHtml(s.path || s.source_path || '') + '</span>'
        + '<span class="rag-source-meta">' + (s.doc_count || 0) + ' docs</span></div>';
      var btn = document.createElement('button');
      btn.className = 'btn btn-danger-sm';
      btn.textContent = 'Delete';
      btn.onclick = function() { deleteDevopsSource(s.path || s.source_path || ''); };
      row.appendChild(btn);
      list.appendChild(row);
    });
    el.appendChild(list);
  } catch (e) {
    el.textContent = 'Sources unavailable: ' + e.message;
    el.className = 'muted';
  }
}

async function indexDevopsDocs() {
  var path = document.getElementById('devops-path').value.trim();
  if (!path) { document.getElementById('devops-path').focus(); return; }
  var recursive = document.getElementById('devops-recursive').checked;
  try {
    var r = await fetch('/devops/index', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: path, recursive: recursive }),
    });
    var d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Failed');
    alert('Indexed ' + (d.docs_indexed || 0) + ' documents');
    fetchDevopsSources();
    fetchRagCollections();
  } catch (e) {
    alert('Error: ' + e.message);
  }
}

async function deleteDevopsSource(sourcePath) {
  if (!confirm('Delete DevOps docs from ' + sourcePath + '?')) return;
  try {
    var r = await fetch('/devops/source/' + encodeURIComponent(sourcePath), { method: 'DELETE' });
    if (!r.ok) throw new Error('Delete failed');
    fetchDevopsSources();
    fetchRagCollections();
  } catch (e) {
    alert('Failed: ' + e.message);
  }
}

async function searchDevops() {
  var query = document.getElementById('devops-search-query').value.trim();
  if (!query) { document.getElementById('devops-search-query').focus(); return; }
  var el = document.getElementById('devops-results');
  el.textContent = 'Searching...';
  el.className = 'muted';
  try {
    var r = await fetch('/devops/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query, top_k: 10 }),
    });
    var d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Failed');
    var results = d.results || [];
    el.textContent = '';
    el.className = '';
    if (!results.length) { el.textContent = 'No results found.'; el.className = 'muted'; return; }
    results.forEach(function(res, i) {
      var card = document.createElement('div');
      card.className = 'rag-result-card glass';
      card.innerHTML = '<div class="rag-result-header">'
        + '<span class="rag-result-rank">#' + (i + 1) + '</span>'
        + '<span class="collection-badge devops">DEVOPS</span>'
        + '<span class="rag-result-path" style="flex:1">' + escapeHtml(res.path || res.source || '') + '</span>'
        + '<span class="rag-result-score">' + (res.score * 100).toFixed(1) + '%</span></div>'
        + '<div class="rag-result-text">' + escapeHtml((res.text || '').slice(0, 400)) + '</div>';
      el.appendChild(card);
    });
  } catch (e) {
    el.textContent = 'Search failed: ' + e.message;
  }
}

async function analyzeLogs() {
  var logText = document.getElementById('devops-log-input').value.trim();
  if (!logText) { document.getElementById('devops-log-input').focus(); return; }
  var service = document.getElementById('devops-log-service').value.trim();
  var el = document.getElementById('devops-log-results');
  el.textContent = 'Analyzing logs...';
  el.className = 'muted';
  try {
    var body = { log_text: logText };
    if (service) body.service = service;
    var r = await fetch('/devops/analyze-logs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    var d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Failed');
    if (d.job_id && d.status !== 'completed') _pollLogAnalysis(d.job_id);
    else renderLogAnalysis(d);
  } catch (e) {
    el.textContent = 'Error: ' + e.message;
  }
}

function _pollLogAnalysis(jobId) {
  var el = document.getElementById('devops-log-results');
  var iv = setInterval(async function() {
    try {
      var r = await fetch('/devops/analyze-logs/' + jobId);
      var d = await r.json();
      if (d.status === 'completed' || d.status === 'failed') { clearInterval(iv); renderLogAnalysis(d); }
      else { el.textContent = 'Analyzing... (' + d.status + ')'; }
    } catch (_e) { /* ignore */ }
  }, 2000);
}

function renderLogAnalysis(data) {
  var el = document.getElementById('devops-log-results');
  el.textContent = '';
  el.className = '';
  var results = data.results || [];
  if (!results.length) {
    el.textContent = 'No issues found. Categories: ' + ((data.categories || []).join(', ') || 'none');
    el.className = 'muted';
    return;
  }
  results.forEach(function(res) {
    var card = document.createElement('div');
    card.className = 'rag-result-card glass';
    var sev = res.severity === 'error' ? 'failed' : 'warning';
    var h = '<div class="rag-result-header">'
      + '<span class="collection-badge ' + sev + '">' + escapeHtml(res.category || res.severity || 'info') + '</span>'
      + '<span style="flex:1">' + escapeHtml(res.summary || res.message || '') + '</span></div>';
    if (res.details) h += '<div class="rag-result-text">' + escapeHtml(res.details) + '</div>';
    if (res.recommendation) h += '<div class="rag-result-meta" style="color:var(--accent-2)">Fix: ' + escapeHtml(res.recommendation) + '</div>';
    card.innerHTML = h;
    el.appendChild(card);
  });
}

// ── Code Indexing ───────────────────────────────────────────

async function indexCodeRepo() {
  var repoPath = document.getElementById('code-repo-path').value.trim();
  if (!repoPath) { document.getElementById('code-repo-path').focus(); return; }
  var repoName = document.getElementById('code-repo-name').value.trim();
  var statusEl = document.getElementById('code-index-status');
  var infoEl = document.getElementById('code-index-info');
  statusEl.style.display = 'block';
  infoEl.textContent = 'Indexing repository...';
  try {
    var body = { repo_path: repoPath };
    if (repoName) body.repo_name = repoName;
    var r = await fetch('/index', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    var d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Failed');
    infoEl.textContent = 'Indexed ' + (d.chunks_indexed || 0) + ' chunks from ' + repoPath;
    fetchRagCollections();
  } catch (e) {
    infoEl.textContent = 'Error: ' + e.message;
  }
}

async function searchCode() {
  var query = document.getElementById('code-search-query').value.trim();
  if (!query) { document.getElementById('code-search-query').focus(); return; }
  var repo = document.getElementById('code-search-repo').value.trim();
  var topK = parseInt(document.getElementById('code-search-topk').value) || 10;
  var el = document.getElementById('code-results');
  el.textContent = 'Searching code...';
  el.className = 'muted';
  try {
    var body = { query: query, top_k: topK, rerank: true };
    if (repo) body.repo = repo;
    var r = await fetch('/retrieve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    var d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Failed');
    var results = d.results || [];
    el.textContent = '';
    el.className = '';
    if (!results.length) { el.textContent = 'No code results found.'; el.className = 'muted'; return; }
    results.forEach(function(res, i) {
      var path = escapeHtml(res.repo || '') + '/' + escapeHtml(res.path || '');
      var lines = 'L' + res.start_line + '-' + res.end_line;
      var symbols = (res.symbols && res.symbols.length) ? res.symbols.join(', ') : '';
      var card = document.createElement('div');
      card.className = 'rag-result-card glass';
      card.innerHTML = '<div class="rag-result-header">'
        + '<span class="rag-result-rank">#' + (i + 1) + '</span>'
        + '<span class="collection-badge code">CODE</span>'
        + '<span class="rag-result-path">' + path + '#' + lines + '</span>'
        + '<span class="rag-result-score">' + (res.score * 100).toFixed(1) + '%</span></div>'
        + '<pre class="rag-result-code">' + escapeHtml(res.text.length > 600 ? res.text.slice(0, 600) + '...' : res.text) + '</pre>'
        + '<div class="rag-result-meta">' + escapeHtml(res.repo || '') + ' &middot; ' + lines + (symbols ? ' &middot; ' + escapeHtml(symbols) : '') + '</div>';
      el.appendChild(card);
    });
  } catch (e) {
    el.textContent = 'Search failed: ' + e.message;
  }
}

// ── Jira Tickets ────────────────────────────────────────────

let _jiraImportInterval = null;
let _jiraStatusInterval = null;
let _jiraBaseUrl = null;

async function fetchJiraStatus() {
  var badge = document.getElementById('jira-connection-badge');
  var statsRow = document.getElementById('jira-stats-row');
  if (!badge) return;
  try {
    var r = await fetch('/jira/status');
    var d = await r.json();
    var configured = d.configured;

    // Store base URL for ticket links
    if (d.base_url) _jiraBaseUrl = d.base_url;

    // Connection badge
    badge.innerHTML = '';
    var dot = document.createElement('span');
    dot.className = 'agent-status-dot ' + (configured ? 'online' : 'offline');
    badge.appendChild(dot);
    var label = document.createElement('span');
    if (configured) {
      label.textContent = d.base_url || 'Connected';
      label.className = '';
    } else {
      label.textContent = 'Not configured';
      label.className = 'muted';
    }
    badge.appendChild(label);

    // Stats row
    if (configured) {
      statsRow.style.display = 'flex';
      var projectCount = (d.sources || []).length;
      document.getElementById('jira-stat-tickets').textContent = (d.total_tickets || 0).toLocaleString();
      document.getElementById('jira-stat-chunks').textContent = (d.total_chunks || 0).toLocaleString();
      document.getElementById('jira-stat-projects').textContent = projectCount;
      var countBadge = document.getElementById('jira-sources-count');
      if (countBadge) countBadge.textContent = '(' + projectCount + ')';

      var lastEl = document.getElementById('jira-stat-last-indexed');
      if (d.last_indexed) {
        var dt = new Date(d.last_indexed);
        var now = new Date();
        var diffMs = now - dt;
        var diffMin = Math.floor(diffMs / 60000);
        if (diffMin < 1) lastEl.textContent = 'Just now';
        else if (diffMin < 60) lastEl.textContent = diffMin + 'm ago';
        else if (diffMin < 1440) lastEl.textContent = Math.floor(diffMin / 60) + 'h ago';
        else lastEl.textContent = dt.toLocaleDateString();
        lastEl.title = dt.toLocaleString();
      } else {
        lastEl.textContent = 'Never';
        lastEl.title = '';
      }

      var nightlyEl = document.getElementById('jira-stat-nightly');
      if (d.nightly_sync_enabled) {
        nightlyEl.textContent = d.nightly_sync_hour + ':00 UTC';
        nightlyEl.title = 'Nightly sync enabled';
      } else {
        nightlyEl.textContent = 'Off';
        nightlyEl.title = 'Set JIRA_SYNC_ENABLED=true to enable';
      }

      // Show active jobs if any running
      var activeJobs = (d.jobs || []).filter(function(j) { return j.status === 'running'; });
      var jobsEl = document.getElementById('jira-active-jobs');
      if (activeJobs.length) {
        jobsEl.style.display = 'block';
        jobsEl.innerHTML = '';
        activeJobs.forEach(function(j) {
          var row = document.createElement('div');
          row.className = 'jira-active-job glass';
          var st = document.createElement('span');
          st.className = 'rag-status-badge running';
          st.textContent = 'Running';
          row.appendChild(st);
          var info = document.createElement('span');
          info.className = 'muted';
          info.textContent = (j.tickets_indexed || 0) + '/' + (j.tickets_found || '?') + ' tickets, ' + (j.chunks_indexed || 0) + ' chunks';
          row.appendChild(info);
          jobsEl.appendChild(row);
        });
      } else {
        jobsEl.style.display = 'none';
      }
      // Populate filter dropdowns
      fetchJiraFilters();
    } else {
      statsRow.style.display = 'none';
    }
  } catch (e) {
    badge.innerHTML = '';
    var dotErr = document.createElement('span');
    dotErr.className = 'agent-status-dot offline';
    badge.appendChild(dotErr);
    var msg = document.createElement('span');
    msg.className = 'muted';
    msg.textContent = 'Status unavailable';
    badge.appendChild(msg);
  }
}

var _jiraFiltersLoaded = false;
async function fetchJiraFilters() {
  if (_jiraFiltersLoaded) return;
  try {
    var r = await fetch('/jira/filters');
    var d = await r.json();
    _jiraFiltersLoaded = true;

    var projSel = document.getElementById('jira-search-project');
    if (projSel && d.projects) {
      d.projects.forEach(function(p) {
        var opt = document.createElement('option');
        opt.value = p;
        opt.textContent = p;
        projSel.appendChild(opt);
      });
    }

    var assignSel = document.getElementById('jira-search-assignee');
    if (assignSel && d.assignees) {
      d.assignees.forEach(function(a) {
        var opt = document.createElement('option');
        opt.value = a;
        opt.textContent = a;
        assignSel.appendChild(opt);
      });
    }
  } catch (e) {
    console.warn('Failed to load Jira filters:', e);
  }
}

async function fetchJiraSources() {
  var el = document.getElementById('jira-sources');
  if (!el) return;
  try {
    var r = await fetch('/jira/sources');
    var d = await r.json();
    var sources = d.sources || [];
    if (!sources.length) {
      el.textContent = 'No Jira projects indexed yet.';
      el.className = 'muted';
      return;
    }
    el.textContent = '';
    el.className = '';
    var list = document.createElement('div');
    list.className = 'rag-sources-list';
    sources.forEach(function(s) {
      var row = document.createElement('div');
      row.className = 'rag-source-row glass';
      var info = document.createElement('div');
      info.className = 'rag-source-info';
      var domain = document.createElement('span');
      domain.className = 'rag-source-domain';
      domain.textContent = s.project;
      var meta = document.createElement('span');
      meta.className = 'rag-source-meta';
      var metaText = (s.ticket_count || 0) + ' tickets \u00b7 ' + (s.chunk_count || 0) + ' chunks';
      if (s.last_indexed) metaText += ' \u00b7 Last indexed: ' + new Date(s.last_indexed).toLocaleString();
      meta.textContent = metaText;
      info.appendChild(domain);
      info.appendChild(meta);
      row.appendChild(info);
      var btn = document.createElement('button');
      btn.className = 'btn btn-danger-sm';
      btn.textContent = 'Delete';
      btn.onclick = function() { deleteJiraProject(s.project); };
      row.appendChild(btn);
      list.appendChild(row);
    });
    el.appendChild(list);
  } catch (e) {
    el.textContent = 'Sources unavailable: ' + e.message;
    el.className = 'muted';
  }
}

async function startJiraImport() {
  var project = document.getElementById('jira-import-project').value.trim();
  var btn = document.getElementById('jira-import-btn');
  var syncBtn = document.getElementById('jira-sync-btn');
  var progressEl = document.getElementById('jira-import-progress');
  var barEl = document.getElementById('jira-progress-bar');
  var infoEl = document.getElementById('jira-import-info');

  btn.disabled = true;
  btn.textContent = 'Importing...';
  if (syncBtn) syncBtn.disabled = true;
  progressEl.style.display = 'block';
  barEl.style.width = '5%';
  infoEl.textContent = 'Starting full import...';

  try {
    var body = {};
    if (project) body.project = project;
    var r = await fetch('/jira/import', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    var d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Import failed');
    _pollJiraImport(d.job_id);
  } catch (e) {
    infoEl.textContent = 'Error: ' + e.message;
    btn.disabled = false;
    btn.textContent = 'Full Import';
    if (syncBtn) syncBtn.disabled = false;
  }
}

async function startJiraSync() {
  var btn = document.getElementById('jira-import-btn');
  var syncBtn = document.getElementById('jira-sync-btn');
  var progressEl = document.getElementById('jira-import-progress');
  var barEl = document.getElementById('jira-progress-bar');
  var infoEl = document.getElementById('jira-import-info');

  if (btn) btn.disabled = true;
  if (syncBtn) { syncBtn.disabled = true; syncBtn.textContent = 'Syncing...'; }
  progressEl.style.display = 'block';
  barEl.style.width = '5%';
  infoEl.textContent = 'Starting sync (last 24h)...';

  try {
    var r = await fetch('/jira/sync', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ since_hours: 24 }),
    });
    var d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Sync failed');
    _pollJiraImport(d.job_id);
  } catch (e) {
    infoEl.textContent = 'Error: ' + e.message;
    if (btn) btn.disabled = false;
    if (syncBtn) { syncBtn.disabled = false; syncBtn.textContent = 'Sync Recent (24h)'; }
  }
}

function _pollJiraImport(jobId) {
  if (_jiraImportInterval) clearInterval(_jiraImportInterval);
  var btn = document.getElementById('jira-import-btn');
  var syncBtn = document.getElementById('jira-sync-btn');
  _jiraImportInterval = setInterval(async function() {
    try {
      var r = await fetch('/jira/import/status/' + jobId);
      if (r.status === 404) {
        // Job not registered yet (background task still starting), wait
        return;
      }
      var d = await r.json();
      var barEl = document.getElementById('jira-progress-bar');
      var infoEl = document.getElementById('jira-import-info');

      var tickets = d.tickets_imported || d.tickets_indexed || 0;
      var found = d.tickets_found || 0;
      var chunks = d.chunks_indexed || 0;
      var pct = found > 0 ? Math.round((tickets / found) * 100) : (d.status === 'running' ? 30 : 100);
      barEl.style.width = Math.min(pct, 100) + '%';
      var statusLabel = d.status === 'running' ? 'Importing' : d.status;
      infoEl.textContent = statusLabel + ' \u2014 ' + tickets + '/' + found + ' tickets, ' + chunks + ' chunks';
      if (d.errors && d.errors.length) {
        infoEl.textContent += ' (' + d.errors.length + ' errors)';
      }

      // Also refresh the stats
      fetchJiraStatus();

      if (d.status === 'completed' || d.status === 'failed') {
        clearInterval(_jiraImportInterval);
        _jiraImportInterval = null;
        btn.disabled = false;
        btn.textContent = 'Full Import';
        if (syncBtn) { syncBtn.disabled = false; }
        if (d.status === 'completed') {
          barEl.style.width = '100%';
          infoEl.textContent = 'Complete: ' + tickets + ' tickets, ' + chunks + ' chunks indexed';
          if (d.ended_at) {
            infoEl.textContent += ' \u2014 ' + new Date(d.ended_at).toLocaleTimeString();
          }
        } else {
          var lastErr = d.errors && d.errors.length ? d.errors[d.errors.length - 1] : 'Unknown error';
          infoEl.textContent = 'Failed: ' + lastErr;
        }
        fetchJiraSources();
        fetchRagCollections();
      }
    } catch (_e) { /* ignore */ }
  }, 3000);
}

async function deleteJiraProject(project) {
  if (!confirm('Delete all indexed tickets from project ' + project + '?')) return;
  try {
    var r = await fetch('/jira/source/' + encodeURIComponent(project), { method: 'DELETE' });
    if (!r.ok) throw new Error('Delete failed');
    fetchJiraSources();
    fetchJiraStatus();
    fetchRagCollections();
  } catch (e) {
    alert('Failed to delete: ' + e.message);
  }
}

async function searchJira() {
  var queryEl = document.getElementById('jira-search-query');
  var query = queryEl.value.trim();
  if (!query) { queryEl.focus(); return; }
  var project = document.getElementById('jira-search-project').value.trim();
  var assignee = document.getElementById('jira-search-assignee').value.trim();
  var issueType = document.getElementById('jira-search-type').value;
  var el = document.getElementById('jira-results');

  el.textContent = 'Searching...';
  el.className = 'muted';

  try {
    var params = new URLSearchParams({ query: query, top_k: '10' });
    if (project) params.set('project', project);
    if (assignee) params.set('assignee', assignee);
    if (issueType) params.set('issue_type', issueType);

    var r = await fetch('/jira/search?' + params.toString());
    var d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Search failed');
    renderJiraResults(d.results || []);
  } catch (e) {
    el.textContent = 'Search failed: ' + e.message;
    el.className = 'muted';
  }
}

function renderJiraResults(results) {
  var el = document.getElementById('jira-results');
  el.textContent = '';
  el.className = '';
  if (!results.length) {
    el.textContent = 'No results found.';
    el.className = 'muted';
    return;
  }
  results.forEach(function(r, i) {
    var score = (r.score * 100).toFixed(1) + '%';
    var statusClass = r.status === 'Done' ? 'active'
      : r.status === 'In Progress' ? 'running'
      : r.status === 'To Do' ? 'inactive' : 'inactive';
    var meta = [];
    if (r.status) meta.push(r.status);
    if (r.assignee) meta.push(r.assignee);
    if (r.issue_type) meta.push(r.issue_type);
    if (r.project) meta.push(r.project);

    var card = document.createElement('div');
    card.className = 'rag-result-card glass';

    var header = document.createElement('div');
    header.className = 'rag-result-header';

    var rank = document.createElement('span');
    rank.className = 'rag-result-rank';
    rank.textContent = '#' + (i + 1);

    var badge = document.createElement('span');
    badge.className = 'collection-badge jira';
    badge.textContent = 'JIRA';

    var key;
    if (_jiraBaseUrl && r.ticket_key) {
      key = document.createElement('a');
      key.href = _jiraBaseUrl + '/browse/' + r.ticket_key;
      key.target = '_blank';
      key.rel = 'noopener';
    } else {
      key = document.createElement('span');
    }
    key.className = 'jira-ticket-key';
    key.textContent = r.ticket_key || '';

    var title = document.createElement('span');
    title.className = 'rag-result-title';
    title.style.flex = '1';
    title.textContent = r.summary || '';

    var statusBadge = document.createElement('span');
    statusBadge.className = 'rag-status-badge ' + statusClass;
    statusBadge.textContent = r.status || '';

    var scoreEl = document.createElement('span');
    scoreEl.className = 'rag-result-score';
    scoreEl.textContent = score;

    header.appendChild(rank);
    header.appendChild(badge);
    header.appendChild(key);
    header.appendChild(title);
    header.appendChild(statusBadge);
    header.appendChild(scoreEl);
    card.appendChild(header);

    if (r.text) {
      var text = document.createElement('div');
      text.className = 'rag-result-text';
      text.textContent = r.text.length > 400 ? r.text.slice(0, 400) + '...' : r.text;
      card.appendChild(text);
    }

    if (meta.length) {
      var metaEl = document.createElement('div');
      metaEl.className = 'rag-result-meta';
      metaEl.textContent = meta.join(' \u00b7 ');
      card.appendChild(metaEl);
    }

    el.appendChild(card);
  });
}

// ── Confluence Pages ─────────────────────────────────────────

let _confluenceImportInterval = null;
let _confluenceBaseUrl = null;

async function fetchConfluenceStatus() {
  var badge = document.getElementById('confluence-connection-badge');
  var statsRow = document.getElementById('confluence-stats-row');
  if (!badge) return;
  try {
    var r = await fetch('/confluence/status');
    var d = await r.json();
    var configured = d.configured;

    if (d.base_url) _confluenceBaseUrl = d.base_url;

    badge.innerHTML = '';
    var dot = document.createElement('span');
    dot.className = 'agent-status-dot ' + (configured ? 'online' : 'offline');
    badge.appendChild(dot);
    var label = document.createElement('span');
    if (configured) {
      label.textContent = d.base_url || 'Connected';
      label.className = '';
    } else {
      label.textContent = 'Not configured';
      label.className = 'muted';
    }
    badge.appendChild(label);

    if (configured) {
      statsRow.style.display = 'flex';
      var spaceCount = (d.sources || []).length;
      document.getElementById('confluence-stat-pages').textContent = (d.total_pages || 0).toLocaleString();
      document.getElementById('confluence-stat-chunks').textContent = (d.total_chunks || 0).toLocaleString();
      document.getElementById('confluence-stat-spaces').textContent = spaceCount;
      var countBadge = document.getElementById('confluence-sources-count');
      if (countBadge) countBadge.textContent = '(' + spaceCount + ')';

      var lastEl = document.getElementById('confluence-stat-last-indexed');
      if (d.last_indexed) {
        var dt = new Date(d.last_indexed);
        var now = new Date();
        var diffMs = now - dt;
        var diffMin = Math.floor(diffMs / 60000);
        if (diffMin < 1) lastEl.textContent = 'Just now';
        else if (diffMin < 60) lastEl.textContent = diffMin + 'm ago';
        else if (diffMin < 1440) lastEl.textContent = Math.floor(diffMin / 60) + 'h ago';
        else lastEl.textContent = dt.toLocaleDateString();
        lastEl.title = dt.toLocaleString();
      } else {
        lastEl.textContent = 'Never';
        lastEl.title = '';
      }

      var nightlyEl = document.getElementById('confluence-stat-nightly');
      if (d.nightly_sync_enabled) {
        nightlyEl.textContent = d.nightly_sync_hour + ':00 UTC';
        nightlyEl.title = 'Nightly sync enabled';
      } else {
        nightlyEl.textContent = 'Off';
        nightlyEl.title = 'Set CONFLUENCE_SYNC_ENABLED=true to enable';
      }

      // Show active jobs if any running
      var activeJobs = (d.jobs || []).filter(function(j) { return j.status === 'running'; });
      var jobsEl = document.getElementById('confluence-active-jobs');
      if (activeJobs.length) {
        jobsEl.style.display = 'block';
        jobsEl.innerHTML = '';
        activeJobs.forEach(function(j) {
          var row = document.createElement('div');
          row.className = 'jira-active-job glass';
          var st = document.createElement('span');
          st.className = 'rag-status-badge running';
          st.textContent = 'Running';
          row.appendChild(st);
          var info = document.createElement('span');
          info.className = 'muted';
          info.textContent = (j.pages_indexed || 0) + '/' + (j.pages_found || '?') + ' pages, ' + (j.chunks_indexed || 0) + ' chunks';
          row.appendChild(info);
          jobsEl.appendChild(row);
        });
      } else {
        jobsEl.style.display = 'none';
      }
    } else {
      statsRow.style.display = 'none';
    }
  } catch (e) {
    badge.innerHTML = '';
    var dotErr = document.createElement('span');
    dotErr.className = 'agent-status-dot offline';
    badge.appendChild(dotErr);
    var msg = document.createElement('span');
    msg.className = 'muted';
    msg.textContent = 'Status unavailable';
    badge.appendChild(msg);
  }
}

async function fetchConfluenceSources() {
  var el = document.getElementById('confluence-sources');
  if (!el) return;
  try {
    var r = await fetch('/confluence/sources');
    var d = await r.json();
    var sources = d.sources || [];
    if (!sources.length) {
      el.textContent = 'No Confluence spaces indexed yet.';
      el.className = 'muted';
      return;
    }
    el.textContent = '';
    el.className = '';
    var list = document.createElement('div');
    list.className = 'rag-sources-list';
    sources.forEach(function(s) {
      var row = document.createElement('div');
      row.className = 'rag-source-row glass';
      var info = document.createElement('div');
      info.className = 'rag-source-info';
      var domain = document.createElement('span');
      domain.className = 'rag-source-domain';
      domain.textContent = s.space;
      var meta = document.createElement('span');
      meta.className = 'rag-source-meta';
      var metaText = (s.page_count || 0) + ' pages \u00b7 ' + (s.chunk_count || 0) + ' chunks';
      if (s.last_indexed) metaText += ' \u00b7 Last indexed: ' + new Date(s.last_indexed).toLocaleString();
      meta.textContent = metaText;
      info.appendChild(domain);
      info.appendChild(meta);
      row.appendChild(info);
      var btn = document.createElement('button');
      btn.className = 'btn btn-danger-sm';
      btn.textContent = 'Delete';
      btn.onclick = function() { deleteConfluenceSpace(s.space); };
      row.appendChild(btn);
      list.appendChild(row);
    });
    el.appendChild(list);
  } catch (e) {
    el.textContent = 'Sources unavailable: ' + e.message;
    el.className = 'muted';
  }
}

async function startConfluenceImport() {
  var space = document.getElementById('confluence-import-space').value.trim();
  var btn = document.getElementById('confluence-import-btn');
  var syncBtn = document.getElementById('confluence-sync-btn');
  var progressEl = document.getElementById('confluence-import-progress');
  var barEl = document.getElementById('confluence-progress-bar');
  var infoEl = document.getElementById('confluence-import-info');

  btn.disabled = true;
  btn.textContent = 'Importing...';
  if (syncBtn) syncBtn.disabled = true;
  progressEl.style.display = 'block';
  barEl.style.width = '5%';
  infoEl.textContent = 'Starting full import...';

  try {
    var body = {};
    if (space) body.space = space;
    var r = await fetch('/confluence/import', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    var d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Import failed');
    _pollConfluenceImport(d.job_id);
  } catch (e) {
    infoEl.textContent = 'Error: ' + e.message;
    btn.disabled = false;
    btn.textContent = 'Full Import';
    if (syncBtn) syncBtn.disabled = false;
  }
}

async function startConfluenceSync() {
  var btn = document.getElementById('confluence-import-btn');
  var syncBtn = document.getElementById('confluence-sync-btn');
  var progressEl = document.getElementById('confluence-import-progress');
  var barEl = document.getElementById('confluence-progress-bar');
  var infoEl = document.getElementById('confluence-import-info');

  if (btn) btn.disabled = true;
  if (syncBtn) { syncBtn.disabled = true; syncBtn.textContent = 'Syncing...'; }
  progressEl.style.display = 'block';
  barEl.style.width = '5%';
  infoEl.textContent = 'Starting sync (last 24h)...';

  try {
    var r = await fetch('/confluence/sync', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ since_hours: 24 }),
    });
    var d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Sync failed');
    _pollConfluenceImport(d.job_id);
  } catch (e) {
    infoEl.textContent = 'Error: ' + e.message;
    if (btn) btn.disabled = false;
    if (syncBtn) { syncBtn.disabled = false; syncBtn.textContent = 'Sync Recent (24h)'; }
  }
}

function _pollConfluenceImport(jobId) {
  if (_confluenceImportInterval) clearInterval(_confluenceImportInterval);
  var btn = document.getElementById('confluence-import-btn');
  var syncBtn = document.getElementById('confluence-sync-btn');
  _confluenceImportInterval = setInterval(async function() {
    try {
      var r = await fetch('/confluence/import/status/' + jobId);
      if (r.status === 404) return;
      var d = await r.json();
      var barEl = document.getElementById('confluence-progress-bar');
      var infoEl = document.getElementById('confluence-import-info');

      var pages = d.pages_imported || d.pages_indexed || 0;
      var found = d.pages_found || 0;
      var chunks = d.chunks_indexed || 0;
      var pct = found > 0 ? Math.round((pages / found) * 100) : (d.status === 'running' ? 30 : 100);
      barEl.style.width = Math.min(pct, 100) + '%';
      var statusLabel = d.status === 'running' ? 'Importing' : d.status;
      infoEl.textContent = statusLabel + ' \u2014 ' + pages + '/' + found + ' pages, ' + chunks + ' chunks';
      if (d.errors && d.errors.length) {
        infoEl.textContent += ' (' + d.errors.length + ' errors)';
      }

      fetchConfluenceStatus();

      if (d.status === 'completed' || d.status === 'failed') {
        clearInterval(_confluenceImportInterval);
        _confluenceImportInterval = null;
        btn.disabled = false;
        btn.textContent = 'Full Import';
        if (syncBtn) { syncBtn.disabled = false; syncBtn.textContent = 'Sync Recent (24h)'; }
        if (d.status === 'completed') {
          barEl.style.width = '100%';
          infoEl.textContent = 'Complete: ' + pages + ' pages, ' + chunks + ' chunks indexed';
          if (d.ended_at) {
            infoEl.textContent += ' \u2014 ' + new Date(d.ended_at).toLocaleTimeString();
          }
        } else {
          var lastErr = d.errors && d.errors.length ? d.errors[d.errors.length - 1] : 'Unknown error';
          infoEl.textContent = 'Failed: ' + lastErr;
        }
        fetchConfluenceSources();
        fetchRagCollections();
      }
    } catch (_e) { /* ignore */ }
  }, 3000);
}

async function deleteConfluenceSpace(space) {
  if (!confirm('Delete all indexed pages from space ' + space + '?')) return;
  try {
    var r = await fetch('/confluence/source/' + encodeURIComponent(space), { method: 'DELETE' });
    if (!r.ok) throw new Error('Delete failed');
    fetchConfluenceSources();
    fetchConfluenceStatus();
    fetchRagCollections();
  } catch (e) {
    alert('Failed to delete: ' + e.message);
  }
}

async function loadConfluenceSearchFilters() {
  try {
    var r = await fetch('/confluence/filters');
    var d = await r.json();
    var spaceSel = document.getElementById('rag-search-confluence-space');
    if (spaceSel && d.spaces) {
      d.spaces.forEach(function(s) {
        var opt = document.createElement('option');
        opt.value = s; opt.textContent = s;
        spaceSel.appendChild(opt);
      });
    }
    var authorSel = document.getElementById('rag-search-confluence-author');
    if (authorSel && d.authors) {
      d.authors.forEach(function(a) {
        var opt = document.createElement('option');
        opt.value = a; opt.textContent = a;
        authorSel.appendChild(opt);
      });
    }
    var deptSel = document.getElementById('rag-search-confluence-department');
    if (deptSel && d.departments) {
      d.departments.forEach(function(dep) {
        var opt = document.createElement('option');
        opt.value = dep; opt.textContent = dep;
        deptSel.appendChild(opt);
      });
    }
    var topicSel = document.getElementById('rag-search-confluence-topic');
    if (topicSel && d.topics) {
      d.topics.forEach(function(t) {
        var opt = document.createElement('option');
        opt.value = t; opt.textContent = t;
        topicSel.appendChild(opt);
      });
    }
  } catch (e) { console.warn('Failed to load Confluence search filters:', e); }
}

// ── Chat with RAG ───────────────────────────────────────────

async function sendChat() {
  var input = document.getElementById('chat-input');
  var query = input.value.trim();
  if (!query) { input.focus(); return; }
  input.value = '';
  var messagesEl = document.getElementById('chat-messages');

  var userMsg = document.createElement('div');
  userMsg.className = 'chat-message user';
  var userRole = document.createElement('span');
  userRole.className = 'chat-role';
  userRole.textContent = 'You';
  var userText = document.createElement('div');
  userText.className = 'chat-text';
  userText.textContent = query;
  userMsg.appendChild(userRole);
  userMsg.appendChild(userText);
  messagesEl.appendChild(userMsg);

  var pendingMsg = document.createElement('div');
  pendingMsg.className = 'chat-message assistant';
  pendingMsg.id = 'chat-pending';
  var aiRole = document.createElement('span');
  aiRole.className = 'chat-role';
  aiRole.textContent = 'AI';
  var aiText = document.createElement('div');
  aiText.className = 'chat-text muted';
  aiText.textContent = 'Thinking...';
  pendingMsg.appendChild(aiRole);
  pendingMsg.appendChild(aiText);
  messagesEl.appendChild(pendingMsg);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  try {
    var r = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        use_rag: document.getElementById('chat-use-rag').checked,
        use_tools: document.getElementById('chat-use-tools').checked,
      }),
    });
    var d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Failed');
    var pending = document.getElementById('chat-pending');
    if (pending) {
      pending.removeAttribute('id');
      aiText = pending.querySelector('.chat-text');
      aiText.className = 'chat-text';
      aiText.textContent = d.response || '';
      if (d.tools_used && d.tools_used.length) {
        var toolsDiv = document.createElement('div');
        toolsDiv.className = 'chat-tools';
        toolsDiv.textContent = 'Tools: ' + d.tools_used.join(', ');
        pending.appendChild(toolsDiv);
      }
    }
  } catch (e) {
    var pend = document.getElementById('chat-pending');
    if (pend) {
      pend.removeAttribute('id');
      var errText = pend.querySelector('.chat-text');
      errText.className = 'chat-text';
      errText.style.color = 'var(--danger)';
      errText.textContent = 'Error: ' + e.message;
    }
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ── Tabs ─────────────────────────────────────────────────────

function initTabs() {
  var buttons = document.querySelectorAll('.tab-btn[data-tab]');
  var panes = document.querySelectorAll('.tab-pane[data-tab]');

  function switchTab(tabId) {
    buttons.forEach(function(btn) {
      var active = btn.getAttribute('data-tab') === tabId;
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-selected', active ? 'true' : 'false');
    });
    panes.forEach(function(pane) {
      pane.classList.toggle('active', pane.getAttribute('data-tab') === tabId);
    });
    try { localStorage.setItem('rag-active-tab', tabId); } catch (_e) { /* ignore */ }
    history.replaceState(null, '', '#' + tabId);
    onTabActivated(tabId);
  }

  buttons.forEach(function(btn) {
    btn.addEventListener('click', function() {
      switchTab(btn.getAttribute('data-tab'));
    });
  });

  // Restore from URL hash or localStorage
  var hash = location.hash.replace('#', '');
  var validTabs = Array.from(panes).map(function(p) { return p.getAttribute('data-tab'); });
  var saved = null;
  try { saved = localStorage.getItem('rag-active-tab'); } catch (_e) { /* ignore */ }
  var initial = validTabs.indexOf(hash) >= 0 ? hash
    : (saved && validTabs.indexOf(saved) >= 0 ? saved : 'sources');
  switchTab(initial);
}

// ── Auto-load on page init ───────────────────────────────────

document.addEventListener('DOMContentLoaded', function() {
  initTabs();
  // Only load essentials on page load — tab data loads lazily via onTabActivated
  fetchQdrantHealth();
  fetchRagCollections();
  // Active tab's data is loaded by initTabs → switchTab → onTabActivated
});
