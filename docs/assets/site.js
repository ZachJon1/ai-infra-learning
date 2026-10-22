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
