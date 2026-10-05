(() => {
  "use strict";

  /* ---------- mobile menu ---------- */
  const burger = document.querySelector(".burger");
  const menu = document.getElementById("mobile-menu");
  const overlay = document.querySelector(".menu-overlay");

  function setMenu(open) {
    burger.setAttribute("aria-expanded", String(open));
    burger.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    menu.hidden = !open;
    overlay.hidden = !open;
    document.body.classList.toggle("menu-open", open);
  }

  burger.addEventListener("click", () => setMenu(burger.getAttribute("aria-expanded") !== "true"));
  overlay.addEventListener("click", () => setMenu(false));
  menu.querySelectorAll("a").forEach((a) => a.addEventListener("click", () => setMenu(false)));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && burger.getAttribute("aria-expanded") === "true") {
      setMenu(false);
      burger.focus();
    }
  });
  window.addEventListener("resize", () => {
    if (window.innerWidth > 720 && burger.getAttribute("aria-expanded") === "true") setMenu(false);
  });

  /* ---------- stats count-up ---------- */
  const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const values = [...document.querySelectorAll(".stat-value")];

  function format(el, v) {
    const decimals = Number(el.dataset.decimals || 0);
    el.textContent = v.toFixed(decimals) + (el.dataset.suffix || "");
  }

  function countUp(el, i) {
    const target = Number(el.dataset.target);
    if (reduced) {
      format(el, target);
      return;
    }
    const duration = 1500 + i * 80;
    setTimeout(() => {
      const start = performance.now();
      const step = (now) => {
        const t = Math.min(1, (now - start) / duration);
        format(el, target * easeOutCubic(t));
        if (t < 1) requestAnimationFrame(step);
        else format(el, target);
      };
      requestAnimationFrame(step);
    }, 480 + i * 90);
  }

  values.forEach((el) => format(el, 0));
  const stats = document.querySelector(".stats");
  if ("IntersectionObserver" in window && stats) {
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          values.forEach(countUp);
          io.disconnect(); // once
        }
      },
      { threshold: 0.25 }
    );
    io.observe(stats);
  } else {
    values.forEach(countUp);
  }
})();
