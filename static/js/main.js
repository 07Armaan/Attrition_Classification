// ==========================================================================
// Shared behaviors: nav toggle, scroll-reveal, FAQ accordion, factor bars
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {

  /* ---- Mobile nav toggle ---- */
  const navToggle = document.getElementById('navToggle');
  const navLinks = document.getElementById('navLinks');
  if (navToggle && navLinks) {
    navToggle.addEventListener('click', () => {
      navLinks.classList.toggle('open');
    });
    navLinks.querySelectorAll('a').forEach(link => {
      link.addEventListener('click', () => navLinks.classList.remove('open'));
    });
  }

  /* ---- Scroll reveal ---- */
  const revealEls = document.querySelectorAll('[data-reveal]');
  if ('IntersectionObserver' in window && revealEls.length) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });

    revealEls.forEach(el => observer.observe(el));
  } else {
    revealEls.forEach(el => el.classList.add('is-visible'));
  }

  /* ---- FAQ accordion ---- */
  document.querySelectorAll('.faq-item').forEach(item => {
    const q = item.querySelector('.faq-q');
    const a = item.querySelector('.faq-a');
    if (!q || !a) return;

    const setHeight = () => {
      a.style.maxHeight = item.classList.contains('open') ? a.scrollHeight + 'px' : '0px';
    };
    setHeight();

    q.addEventListener('click', () => {
      const isOpen = item.classList.contains('open');
      // close all
      document.querySelectorAll('.faq-item.open').forEach(other => {
        if (other !== item) {
          other.classList.remove('open');
          other.querySelector('.faq-a').style.maxHeight = '0px';
        }
      });
      item.classList.toggle('open', !isOpen);
      setHeight();
    });

    window.addEventListener('resize', () => {
      if (item.classList.contains('open')) setHeight();
    });
  });

  /* ---- Animated factor bars (info page "The model" section) ---- */
  const factorBars = document.querySelectorAll('#factorBars .factor-fill');
  if (factorBars.length && 'IntersectionObserver' in window) {
    const barObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.style.width = entry.target.dataset.width + '%';
          barObserver.unobserve(entry.target);
        }
      });
    }, { threshold: 0.3 });
    factorBars.forEach(bar => barObserver.observe(bar));
  }

});