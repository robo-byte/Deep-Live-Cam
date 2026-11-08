(function () {
  if (window.__ANYWEAR_LOADED__) {
    if (window.__ANYWEAR_TOAST__) {
      window.__ANYWEAR_TOAST__("Anywear is already active! Click any clothing image on this page to wear it.", true);
    }
    return;
  }
  window.__ANYWEAR_LOADED__ = true;

  const toast = document.createElement("div");
  toast.id = "anywear-live-toast";
  Object.assign(toast.style, {
    position: "fixed",
    top: "20px",
    right: "20px",
    zIndex: "9999999",
    background: "linear-gradient(135deg, #6366f1, #a855f7)",
    color: "#ffffff",
    padding: "14px 22px",
    borderRadius: "12px",
    boxShadow: "0 10px 30px rgba(0,0,0,0.35)",
    fontFamily: "system-ui, -apple-system, sans-serif",
    fontSize: "14px",
    fontWeight: "600",
    display: "flex",
    alignItems: "center",
    gap: "10px",
    transition: "all 0.3s ease",
    cursor: "pointer",
  });
  toast.innerHTML = "✨ <span><strong>Anywear Live Try-On Active!</strong> Click any clothing item on this page to wear it on your camera.</span>";
  document.body.appendChild(toast);

  window.__ANYWEAR_TOAST__ = function (msg, isSuccess) {
    toast.innerHTML = (isSuccess ? "✅ " : "⚠️ ") + "<span>" + msg + "</span>";
    toast.style.background = isSuccess ? "linear-gradient(135deg, #10b981, #059669)" : "linear-gradient(135deg, #ef4444, #dc2626)";
    setTimeout(function () {
      toast.innerHTML = "✨ <span><strong>Anywear Live Active!</strong> Click any clothing item to try it on.</span>";
      toast.style.background = "linear-gradient(135deg, #6366f1, #a855f7)";
    }, 4000);
  };

  async function tryOnUrl(url) {
    window.__ANYWEAR_TOAST__("Sending garment to your webcam studio...", true);
    try {
      const resp = await fetch("http://localhost:7860/api/import_url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url, category: "t-shirt" }),
      });
      const data = await resp.json();
      if (data.ok) {
        window.__ANYWEAR_TOAST__("✨ Now wearing " + (data.garment_name || "garment") + " on your live webcam!", true);
      } else {
        window.__ANYWEAR_TOAST__("Error: " + (data.message || "Failed to load garment"), false);
      }
    } catch (err) {
      window.__ANYWEAR_TOAST__("Cannot reach studio! Ensure http://localhost:7860 is running.", false);
    }
  }

  document.addEventListener("click", function (e) {
    let el = e.target;
    if (el.tagName !== "IMG") {
      el = el.closest("img") || el.querySelector("img");
    }
    if (el && el.tagName === "IMG" && (el.naturalWidth > 100 || el.clientWidth > 100)) {
      const src = el.currentSrc || el.src;
      if (src && (src.startsWith("http") || src.startsWith("data:image/"))) {
        e.preventDefault();
        e.stopPropagation();
        tryOnUrl(src);
      }
    }
  }, true);

  const style = document.createElement("style");
  style.textContent = `
    img:hover {
      outline: 3px solid #a855f7 !important;
      cursor: crosshair !important;
    }
  `;
  document.head.appendChild(style);
})();
