/* PPE Safety AI — shared utilities */

// ---- Toast notifications ------------------------------------------------
function showToast(msg, type = "info", duration = 4000) {
  let container = document.querySelector(".toast-container");
  if (!container) {
    container = document.createElement("div");
    container.className = "toast-container";
    document.body.appendChild(container);
  }
  const t = document.createElement("div");
  t.className = `toast toast-${type}`;
  t.textContent = msg;
  container.appendChild(t);
  setTimeout(() => t.remove(), duration);
}

// ---- Dashboard stats ----------------------------------------------------
async function loadDashboardStats() {
  try {
    const r = await fetch("/api/stats");
    const d = await r.json();
    const el = (id) => document.getElementById(id);
    if (el("stat-total-scans"))    el("stat-total-scans").textContent    = d.total_scans;
    if (el("stat-compliance"))     el("stat-compliance").textContent     = d.compliance_rate + "%";
    if (el("stat-violations"))     el("stat-violations").textContent     = d.violations_today;
    if (el("stat-people"))         el("stat-people").textContent         = d.people_detected;
  } catch (_) {}
}

// ---- Dashboard charts ---------------------------------------------------
let chartDaily = null, chartVtype = null;

async function loadDashboardCharts() {
  try {
    const r = await fetch("/api/charts");
    const d = await r.json();

    // 7-day compliance line chart
    const ctx1 = document.getElementById("chart-daily");
    if (ctx1) {
      if (chartDaily) chartDaily.destroy();
      chartDaily = new Chart(ctx1, {
        type: "line",
        data: {
          labels:   d.daily.labels,
          datasets: [{
            label: "Compliance %",
            data:  d.daily.data,
            borderColor: "#6c63ff",
            backgroundColor: "rgba(108,99,255,.15)",
            borderWidth: 2,
            tension: 0.4,
            fill: true,
            pointBackgroundColor: "#6c63ff",
            pointRadius: 4,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { ticks: { color: "#64748b" }, grid: { color: "rgba(255,255,255,.05)" } },
            y: { min: 0, max: 100, ticks: { color: "#64748b", callback: v => v + "%" },
                 grid: { color: "rgba(255,255,255,.05)" } },
          },
        },
      });
    }

    // Violation types donut chart
    const ctx2 = document.getElementById("chart-vtype");
    if (ctx2) {
      if (chartVtype) chartVtype.destroy();
      chartVtype = new Chart(ctx2, {
        type: "doughnut",
        data: {
          labels:   d.vtype.labels,
          datasets: [{
            data:             d.vtype.data,
            backgroundColor:  ["#ef4444","#f59e0b","#3b82f6","#10b981","#6c63ff"],
            borderColor:      "transparent",
            hoverBorderColor: "#1e2235",
            hoverBorderWidth: 3,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: {
            legend: { position: "bottom", labels: { color: "#e2e8f0", boxWidth: 12, padding: 12 } },
          },
          cutout: "65%",
        },
      });
    }
  } catch (_) {}
}

// ---- Violations table ---------------------------------------------------
async function loadViolations(date = "", ppeType = "") {
  const params = new URLSearchParams();
  if (date)    params.set("date",     date);
  if (ppeType) params.set("ppe_type", ppeType);

  const r   = await fetch("/api/violations/list?" + params);
  const rows = await r.json();
  const tbody = document.getElementById("violations-tbody");
  if (!tbody) return;

  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="table-empty">No violations found</td></tr>`;
    return;
  }
  tbody.innerHTML = rows.map(row => {
    const missing = (row.missing_ppe || "").split(",").map(s =>
      `<span class="badge badge-missing">${s.trim()}</span>`).join(" ");
    const present = (row.present_ppe || "").split(",").filter(Boolean).map(s =>
      `<span class="badge badge-compliant">${s.trim()}</span>`).join(" ");
    return `<tr>
      <td>${row.id}</td>
      <td>${row.timestamp}</td>
      <td>${missing || "—"}</td>
      <td>${present  || "—"}</td>
      <td>${row.location || "—"}</td>
      <td>${row.person_count || 1}</td>
    </tr>`;
  }).join("");
}
