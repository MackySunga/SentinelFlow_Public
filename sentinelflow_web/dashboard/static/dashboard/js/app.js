document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('table').forEach(table => {
    table.addEventListener('mousemove', e => {
      const row = e.target.closest('tr');
      if (!row) return;
      row.style.outline = '1px solid rgba(54,215,255,.35)';
      row.addEventListener('mouseleave', () => row.style.outline = 'none', { once: true });
    });
  });
});
