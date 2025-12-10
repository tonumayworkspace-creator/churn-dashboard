(function () {
  "use strict";

  function extractAppShell(htmlText) {
    const parser = new DOMParser();
    const doc = parser.parseFromString(htmlText, "text/html");
    return doc.querySelector(".app-shell");
  }

  function createLoadingOverlay() {
    let overlay = document.getElementById("loading-overlay");
    if (overlay) return overlay;

    overlay = document.createElement("div");
    overlay.id = "loading-overlay";
    overlay.style.position = "fixed";
    overlay.style.inset = "0";
    overlay.style.background = "rgba(255,255,255,0.6)";
    overlay.style.display = "flex";
    overlay.style.alignItems = "center";
    overlay.style.justifyContent = "center";
    overlay.style.zIndex = "9999";
    overlay.style.backdropFilter = "blur(3px)";

    const box = document.createElement("div");
    box.style.padding = "16px 20px";
    box.style.borderRadius = "12px";
    box.style.background = "white";
    box.style.boxShadow = "0 8px 24px rgba(15,23,42,0.08)";
    box.style.display = "flex";
    box.style.alignItems = "center";
    box.style.gap = "12px";

    box.innerHTML = `<div class="spinner" style="width:36px;height:36px;border-radius:50%;border:4px solid rgba(255,255,255,0.6);border-top-color:var(--brand);animation:spin 0.9s linear infinite"></div>
                     <div style="font-weight:700">Predicting…</div>`;

    const style = document.createElement("style");
    style.textContent = "@keyframes spin{to{transform:rotate(360deg)}}";
    document.head.appendChild(style);

    overlay.appendChild(box);
    document.body.appendChild(overlay);
    overlay.style.display = "none";
    return overlay;
  }

  function showLoading() { createLoadingOverlay().style.display = "flex"; }
  function hideLoading() {
    const overlay = document.getElementById("loading-overlay");
    if (overlay) overlay.style.display = "none";
  }

  function ensureAuthenticatedInteract(redirect = true) {
    if (!window.CURRENT_USER) {
      if (redirect) {
        const next = encodeURIComponent(window.location.pathname);
        window.location.href = `/login?next=${next}`;
      }
      return false;
    }
    return true;
  }

  function validateForm() {
    const t = Number(document.getElementById("tenure").value);
    const m = Number(document.getElementById("monthly_charges").value);
    const tot = Number(document.getElementById("total_charges").value);

    if (t < 0 || isNaN(t)) return false;
    if (m < 0 || isNaN(m)) return false;
    if (tot < 0 || isNaN(tot)) return false;

    return true;
  }

  function bindAjaxForm() {
    const form = document.getElementById("predict-form");
    if (!form) return;

    form.addEventListener("submit", function (ev) {
      if (!ensureAuthenticatedInteract(true)) return;
      ev.preventDefault();

      if (!validateForm()) {
        alert("Please fill valid numeric values.");
        return;
      }

      showLoading();
      const data = new FormData(form);

      fetch(form.action, {
        method: "POST",
        body: data,
      })
        .then(r => r.text())
        .then(htmlText => {
          const newShell = extractAppShell(htmlText);
          if (newShell) {
            document.querySelector(".app-shell").replaceWith(newShell);
            bindAjaxForm();
          }
        })
        .finally(hideLoading);
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    bindAjaxForm();
  });

})();
