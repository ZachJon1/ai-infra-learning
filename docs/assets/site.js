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
