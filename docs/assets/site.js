(() => {
  const input = document.querySelector('[data-roadmap-search]');
  if (!input) return;
  const entries = [...document.querySelectorAll('.day')];
  const sections = [...document.querySelectorAll('.phase')];
  const count = document.querySelector('[data-roadmap-count]');
  function filter() {
    const term = input.value.toLowerCase().trim();
    let shown = 0;
    entries.forEach(el => {
      const visible = !term || el.textContent.toLowerCase().includes(term);
      el.dataset.hidden = String(!visible);
      if (visible) shown++;
    });
    sections.forEach(section => {
      section.dataset.hidden = String(!section.querySelector('.day:not([data-hidden="true"])'));
    });
    if (count) count.textContent = `${shown} of ${entries.length} ${input.dataset.searchKind || "learning days"}`;
  }
  input.addEventListener('input', filter);
  filter();
})();

(() => {
  const list = document.querySelector('[data-recent-commits]');
  if (!list) return;

  const allCommitsUrl = 'https://github.com/ZachJon1/ai-infra-learning/commits/main';
  fetch('https://api.github.com/repos/ZachJon1/ai-infra-learning/commits?per_page=5')
    .then(response => {
      if (!response.ok) throw new Error('GitHub API request failed');
      return response.json();
    })
    .then(commits => {
      list.replaceChildren();
      commits.forEach(commit => {
        const item = document.createElement('li');
        const link = document.createElement('a');
        const message = commit.commit.message.split('\n', 1)[0];
        const date = new Date(commit.commit.author.date || commit.commit.committer.date);
        link.href = commit.html_url;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.textContent = message;
        item.append(link);
        if (!Number.isNaN(date.getTime())) {
          const time = document.createElement('small');
          time.className = 'muted';
          time.textContent = ` · ${date.toLocaleDateString()}`;
          item.append(time);
        }
        list.append(item);
      });
      if (!commits.length) list.innerHTML = '<li class="muted">No commits found yet.</li>';
    })
    .catch(() => {
      list.innerHTML = `<li class="muted">Latest updates could not be loaded. <a href="${allCommitsUrl}" target="_blank" rel="noopener noreferrer">View commits on GitHub ↗</a></li>`;
    });
})();

(() => {
  const config = {
    siteCode: 'zakjon',
    productionHost: 'zachjon1.github.io',
    path: '/portfolio',
  };

  // Do not count embedded pages. In particular, animation iframes must not
  // generate a second visit for the page that contains them.
  if (window.top !== window.self) return;

  const footer = document.querySelector('.site-footer .footer-row');
  if (!footer || footer.querySelector('.visitor-count')) return;

  const counter = document.createElement('span');
  counter.className = 'visitor-count';
  counter.append('Visitors: ');
  const value = document.createElement('span');
  value.textContent = '—';
  value.setAttribute('aria-live', 'polite');
  value.setAttribute('aria-atomic', 'true');
  counter.append(value);
  footer.append(counter);

  const siteCode = config.siteCode.trim();
  if (window.location.hostname !== config.productionHost || !/^[a-z0-9-]+$/i.test(siteCode)) return;

  const endpoint = `https://${siteCode}.goatcounter.com`;
  const tracker = document.createElement('script');
  tracker.async = true;
  tracker.src = 'https://gc.zgo.at/count.js';
  tracker.dataset.goatcounter = `${endpoint}/count`;
  tracker.dataset.goatcounterSettings = JSON.stringify({ path: config.path });
  document.head.append(tracker);

  const countUrl = `${endpoint}/counter/${encodeURIComponent(config.path)}.json`;
  function loadVisitorCount(retryMissingPath = true) {
    fetch(countUrl, { mode: 'cors', credentials: 'omit' })
      .then(response => {
        if (response.status === 404 && retryMissingPath) {
          // GoatCounter may not have processed the first /portfolio hit yet.
          window.setTimeout(() => loadVisitorCount(false), 12000);
          return null;
        }
        if (!response.ok) throw new Error('GoatCounter visitor count is unavailable');
        return response.json();
      })
      .then(result => {
        if (result && typeof result.count === 'string' && result.count.trim()) {
          value.textContent = result.count;
        }
      })
      .catch(() => {
        // Keep the neutral placeholder if the public counter is blocked/offline.
      });
  }
  loadVisitorCount();
})();
