"use strict";

(function () {
  // ---------------------------------------------------------------
  // Payload bootstrap
  // ---------------------------------------------------------------
  const dataNode = document.getElementById("data");
  const PAYLOAD = dataNode ? JSON.parse(dataNode.textContent) : {};

  // Register Chart.js annotation plugin if both globals are present.
  if (window.Chart && window["chartjs-plugin-annotation"]) {
    Chart.register(window["chartjs-plugin-annotation"]);
  }

  // Theme colours -- single source of truth lives in app.css :root.
  // Read once at boot; if anyone toggles a theme later they can
  // re-evaluate via CHART.refresh().
  const CHART = (function readChartTheme() {
    const cs = getComputedStyle(document.documentElement);
    const get = (name, fallback) => {
      const v = cs.getPropertyValue(name).trim();
      return v || fallback;
    };
    return {
      text: get('--chart-text', '#6b7280'),
      grid: get('--chart-grid', 'rgba(17,24,39,0.08)'),
      axis: get('--chart-axis', '#374151'),
    };
  })();

  // ---------------------------------------------------------------
  // Sticky bar (panel 6)
  // ---------------------------------------------------------------
  function renderStickyBar(container, state, kpi) {
    container.innerHTML = "";
    if (!state) {
      container.appendChild(cell("status", "no run.json"));
      return;
    }
    const stepPair = kpi && kpi.step ? kpi.step : null;
    const fwdPair  = kpi && kpi.fwd  ? kpi.fwd  : null;
    const bwdPair  = kpi && kpi.bwd  ? kpi.bwd  : null;

    // Fall back to ``state.best_score`` when ``kpi_summary`` is missing
    // (older payloads / partial state) so the cell still shows a number.
    const stepFallback = state.best_score && state.best_score.step_geomean;

    const cells = [
      ["campaign", state.campaign_id || "—"],
      ["phase", state.current_phase || "—", `phase-${(state.current_phase || "").split(" ")[0]}`],
      ["round", String(state.current_round ?? "—")],
      ["best round", state.best_round != null ? `#${state.best_round}` : "—"],
      ["best step geomean", fmtKpiTrio(stepPair, stepFallback)],
      ["forward",  fmtKpiTrio(fwdPair)],
      ["backward", fmtKpiTrio(bwdPair)],
      ["rollback streak", String(state.rollback_streak ?? 0)],
      ["started", state.started_at || "—"],
      ["last update", state.last_update || "—"],
    ];
    for (const [label, value, extra] of cells) {
      container.appendChild(cell(label, value, extra));
    }
  }

  // ``baseline -> best (+pct %)``.  Falls back to the plain best value
  // (or em-dash) when baseline / best are missing or non-numeric.
  function fmtKpiTrio(pair, fallbackBest) {
    const b = pair && Number.isFinite(pair.baseline) ? pair.baseline : null;
    const x = pair && Number.isFinite(pair.best)
      ? pair.best
      : (Number.isFinite(fallbackBest) ? fallbackBest : null);
    if (b == null && x == null) return "—";
    if (b == null || b === 0) {
      return x != null ? fmtTflops(x) : "—";
    }
    if (x == null) return fmtTflops(b);
    const pct = ((x - b) / b) * 100;
    const sign = pct >= 0 ? "+" : "";
    return `${fmtTflops(b)}->${fmtTflops(x)} (${sign}${pct.toFixed(1)} %)`;
  }

  function cell(label, value, valueClass) {
    const root = document.createElement("div");
    root.className = "sticky-cell";
    const l = document.createElement("span");
    l.className = "label";
    l.textContent = label;
    const v = document.createElement("span");
    v.className = "value" + (valueClass ? " " + valueClass : "");
    v.textContent = value;
    root.appendChild(l);
    root.appendChild(v);
    return root;
  }

  // ---------------------------------------------------------------
  // Perf trend (panel 1)
  // ---------------------------------------------------------------
  function renderPerfTrend(container, panel) {
    const canvas = container.querySelector('[data-role="perf-chart"]');
    const empty  = container.querySelector('[data-role="perf-empty"]');
    if (!panel || !panel.points || panel.points.length === 0) {
      if (canvas) canvas.style.display = "none";
      if (empty)  empty.hidden = false;
      return;
    }
    if (empty) empty.hidden = true;
    if (!window.Chart || !canvas) return;

    if (canvas._chart) {
      canvas._chart.destroy();
    }

    const allRounds = panel.points.map(p => p.round);
    const xMin = Math.min(...allRounds, 0);
    const xMax = Math.max(...allRounds) + 0.5;

    const annotations = {};
    if (panel.baseline != null) {
      annotations.baseline = {
        type: "line",
        yMin: panel.baseline,
        yMax: panel.baseline,
        borderColor: "rgba(153,161,173,0.7)",
        borderWidth: 1,
        borderDash: [4, 4],
        label: { display: true, content: "baseline", position: "start",
                 backgroundColor: "rgba(153,161,173,0.2)", color: CHART.text,
                 font: { size: 10 } },
      };
    }
    if (panel.best != null) {
      annotations.best = {
        type: "line",
        yMin: panel.best,
        yMax: panel.best,
        borderColor: "rgba(93,212,158,0.7)",
        borderWidth: 1,
        borderDash: [2, 4],
        label: { display: true, content: "best", position: "end",
                 backgroundColor: "rgba(93,212,158,0.2)", color: "#5dd49e",
                 font: { size: 10 } },
      };
    }
    (panel.annotations && panel.annotations.accept || []).forEach((a, i) => {
      annotations["accept_" + i] = {
        type: "line",
        xMin: a.round,
        xMax: a.round,
        borderColor: "rgba(90,168,255,0.4)",
        borderWidth: 1,
        borderDash: [3, 3],
      };
    });

    const datasetPoints = panel.points.map(p => ({
      x: p.round, y: p.y,
      _status: p.status, _description: p.description,
      _fwd: p.fwd, _bwd: p.bwd,
    }));
    const datasetAccepted = (panel.accepted || []).map(p => ({
      x: p.round, y: p.y, _status: p.status,
    }));
    const fwdLine = panel.points
      .filter(p => p.fwd != null)
      .map(p => ({ x: p.round, y: p.fwd }));
    const bwdLine = panel.points
      .filter(p => p.bwd != null)
      .map(p => ({ x: p.round, y: p.bwd }));

    canvas._chart = new Chart(canvas.getContext("2d"), {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "fwd avg",
            data: fwdLine,
            type: "line",
            borderColor: "rgba(59,130,246,0.75)",
            backgroundColor: "rgba(59,130,246,0)",
            pointRadius: 0,
            borderWidth: 1.5,
            tension: 0.15,
            borderDash: [4, 3],
            order: 3,
          },
          {
            label: "bwd avg",
            data: bwdLine,
            type: "line",
            borderColor: "rgba(168,85,247,0.75)",
            backgroundColor: "rgba(168,85,247,0)",
            pointRadius: 0,
            borderWidth: 1.5,
            tension: 0.15,
            borderDash: [4, 3],
            order: 3,
          },
          {
            label: "step geomean",
            data: datasetPoints,
            backgroundColor: datasetPoints.map(p => statusColor(p._status, 0.85)),
            borderColor: datasetPoints.map(p => statusColor(p._status, 1)),
            pointRadius: 5,
            pointHoverRadius: 7,
            order: 1,
          },
          {
            label: "accepted (step-line)",
            data: datasetAccepted,
            type: "line",
            borderColor: "rgba(16,185,129,0.7)",
            backgroundColor: "rgba(16,185,129,0.12)",
            stepped: true,
            pointRadius: 0,
            borderWidth: 2,
            tension: 0,
            order: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: { labels: { color: CHART.text, font: { size: 11 } } },
          tooltip: {
            callbacks: {
              title: (items) => "round " + items[0].raw.x,
              label: (item) => {
                const p = item.raw;
                if (item.dataset.label === "fwd avg") {
                  return `fwd avg: ${fmtTflops(p.y)}`;
                }
                if (item.dataset.label === "bwd avg") {
                  return `bwd avg: ${fmtTflops(p.y)}`;
                }
                if (item.dataset.label === "accepted (step-line)") {
                  return `accepted geomean: ${fmtTflops(p.y)}`;
                }
                const lines = [`step geomean: ${fmtTflops(p.y)}`];
                if (p._fwd != null) lines.push(`  fwd avg: ${fmtTflops(p._fwd)}`);
                if (p._bwd != null) lines.push(`  bwd avg: ${fmtTflops(p._bwd)}`);
                if (p._status) lines.push(`status: ${p._status}`);
                if (p._description) lines.push(p._description);
                return lines;
              },
            },
          },
          annotation: { annotations },
        },
        scales: {
          x: {
            type: "linear", min: xMin, max: xMax,
            title: { display: true, text: "round", color: CHART.text },
            ticks: { color: CHART.text, stepSize: 1 },
            grid:  { color: CHART.grid },
          },
          y: {
            title: { display: true, text: "step geomean TFLOPS", color: CHART.text },
            ticks: { color: CHART.text },
            grid:  { color: CHART.grid },
          },
        },
      },
    });
  }

  function statusColor(status, alpha) {
    const a = alpha == null ? 1 : alpha;
    switch ((status || "").toUpperCase()) {
      case "BASELINE": return `rgba(197,201,211,${a})`;
      case "ACCEPTED":
      case "ACCEPTED (NOISE-BOUNDED)": return `rgba(93,212,158,${a})`;
      case "ROLLBACK": return `rgba(241,124,138,${a})`;
      default:         return `rgba(107,114,128,${a})`;
    }
  }

  // ---------------------------------------------------------------
  // Pagination helper
  // ---------------------------------------------------------------
  // Mounts a prev/next + "page X / N" + jump-input control into
  // ``footer`` and re-invokes ``renderPage(items, pageIdx)`` with the
  // page slice each time the user navigates. Returns the controller
  // so callers can re-render without rebuilding the footer.
  function paginate(footer, items, pageSize, renderPage) {
    footer.innerHTML = "";
    footer.classList.add("paginator");
    const total = items.length;
    const pages = Math.max(1, Math.ceil(total / pageSize));
    let page = 0;

    const prev = document.createElement("button");
    prev.type = "button";
    prev.textContent = "‹ prev";
    const next = document.createElement("button");
    next.type = "button";
    next.textContent = "next ›";
    const label = document.createElement("span");
    label.className = "paginator-label";
    const jump = document.createElement("input");
    jump.type = "number";
    jump.min = "1";
    jump.max = String(pages);
    jump.className = "paginator-jump";

    footer.appendChild(prev);
    footer.appendChild(label);
    footer.appendChild(next);
    if (pages > 1) {
      const sep = document.createElement("span");
      sep.className = "paginator-sep";
      sep.textContent = "go to";
      footer.appendChild(sep);
      footer.appendChild(jump);
    }

    function paint() {
      page = Math.max(0, Math.min(pages - 1, page));
      const start = page * pageSize;
      const slice = items.slice(start, start + pageSize);
      label.textContent =
        `page ${page + 1} / ${pages}  ·  ${start + 1}-${start + slice.length} of ${total}`;
      prev.disabled = page === 0;
      next.disabled = page >= pages - 1;
      jump.value = String(page + 1);
      renderPage(slice, page);
    }

    prev.addEventListener("click", () => { page -= 1; paint(); });
    next.addEventListener("click", () => { page += 1; paint(); });
    jump.addEventListener("change", () => {
      const v = parseInt(jump.value, 10);
      if (!Number.isNaN(v)) { page = v - 1; paint(); }
    });

    if (pages <= 1) {
      footer.classList.add("paginator-single");
    }
    paint();
    return { repaint: paint };
  }

  // ---------------------------------------------------------------
  // Rounds table (panel 5)
  // ---------------------------------------------------------------
  function renderRoundsTable(container, rounds, profilePanels, profileGlobal, profileDiffs) {
    const tbody = container.querySelector('[data-role="rounds-table"] tbody');
    const empty = container.querySelector('[data-role="rounds-empty"]');
    let footer = container.querySelector('[data-role="rounds-paginator"]');
    if (!footer) {
      footer = document.createElement("div");
      footer.setAttribute("data-role", "rounds-paginator");
      container.appendChild(footer);
    }
    if (!tbody) return;
    tbody.innerHTML = "";
    footer.innerHTML = "";
    if (!rounds || rounds.length === 0) {
      if (empty) empty.hidden = false;
      return;
    }
    if (empty) empty.hidden = true;

    paginate(footer, rounds, 20, (page) => {
      tbody.innerHTML = "";
      page.forEach(appendRoundRow);
    });

    function appendRoundRow(r) {
      const head = document.createElement("tr");
      const decisionClass = (r.decision || "PENDING").replace(/[^A-Za-z]+/g, "-");
      head.className = "head-row decision-" + decisionClass;

      const profilePanel = profilePanels && profilePanels[String(r.n)];
      const expandable = !!(r.summary_md_html || profilePanel);

      const tdN = document.createElement("td");
      const caret = document.createElement("span");
      caret.className = "expand-caret";
      caret.textContent = expandable ? "▸" : " ";
      tdN.appendChild(caret);
      tdN.appendChild(document.createTextNode(" #" + r.n));
      head.appendChild(tdN);

      const tdDecision = document.createElement("td");
      const badge = document.createElement("span");
      badge.className = "decision-badge decision-" + decisionClass;
      badge.textContent = r.decision || "PENDING";
      tdDecision.appendChild(badge);
      head.appendChild(tdDecision);

      head.appendChild(td(r.description || "—"));
      head.appendChild(td(fmtTflops(extractStepGeomean(r)), "tabular"));
      head.appendChild(td(r.perf ? r.perf.vs_baseline : "—", "tabular"));
      head.appendChild(td(r.at || "—", "tabular"));

      tbody.appendChild(head);

      if (expandable) {
        const detailRow = document.createElement("tr");
        detailRow.className = "summary-row";
        const detailCell = document.createElement("td");
        detailCell.colSpan = 6;
        const detailDiv = document.createElement("div");
        detailDiv.className = "round-summary";
        detailDiv.hidden = true;

        if (r.summary_md_html) {
          const md = document.createElement("div");
          md.innerHTML = r.summary_md_html;
          detailDiv.appendChild(md);
        }

        if (profilePanel) {
          const pane = document.createElement("div");
          pane.className = "profile-pane";
          renderProfilePane(pane, profilePanel, profileGlobal, profileDiffs);
          detailDiv.appendChild(pane);
        }

        detailCell.appendChild(detailDiv);
        detailRow.appendChild(detailCell);
        tbody.appendChild(detailRow);

        head.addEventListener("click", () => {
          const willOpen = detailDiv.hidden;
          detailDiv.hidden = !willOpen;
          caret.textContent = willOpen ? "▾" : "▸";
          head.classList.toggle("expanded", willOpen);
        });
      }
    }
  }

  // ---------------------------------------------------------------
  // Profile sub-panels (P1 / P2 / P3 / P4 / P5 / P9 / P10 / P11)
  // ---------------------------------------------------------------
  function renderProfilePane(pane, panel, globals, diffs) {
    const header = document.createElement("div");
    header.className = "profile-header";
    const label = document.createElement("span");
    label.textContent = "profile · round " + panel.round;
    header.appendChild(label);
    const flavor = document.createElement("span");
    flavor.className = "flavor";
    flavor.textContent = panel.flavor || "—";
    header.appendChild(flavor);
    if (panel.perfetto_results_path) {
      const link = document.createElement("a");
      link.className = "perfetto-link";
      link.href = panel.perfetto_results_path;
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = "results.json ↗";
      header.appendChild(link);

      const ui = document.createElement("a");
      ui.className = "perfetto-link";
      ui.href = "https://ui.perfetto.dev/#!/?url=" + encodeURIComponent(panel.perfetto_results_path);
      ui.target = "_blank";
      ui.rel = "noopener";
      ui.textContent = "Open in Perfetto ↗";
      header.appendChild(ui);
    }
    pane.appendChild(header);

    if (panel.summary_md_html) {
      const md = document.createElement("div");
      md.innerHTML = panel.summary_md_html;
      pane.appendChild(md);
    }

    // Per-round visualisations only -- the over-rounds globals (P2 / P3 /
    // P4) are rendered once in the dedicated "Profile · trends over rounds"
    // panel below the rounds table, not duplicated per row.
    const grid = document.createElement("div");
    grid.className = "profile-grid profile-grid-single";
    pane.appendChild(grid);

    const c1 = chartCanvasCell(grid);
    drawTopN(c1, panel.top_n);

    const treeWrap = document.createElement("div");
    treeWrap.className = "treemap-wrap";
    pane.appendChild(treeWrap);
    drawTreemap(treeWrap, panel.treemap);

    if (diffs && diffs.pairs && diffs.pairs.length > 0) {
      pane.appendChild(buildDiffTable(panel.round, diffs));
    }
  }

  // Stand-alone panel below the rounds table that hosts the three
  // over-rounds charts (P2 top-N over rounds, P3 family rollup, P4 GPU
  // resources). They consume ``profile_global`` only, so they render
  // once for the whole campaign instead of once per expanded row.
  function renderProfileGlobalPanel(container, globals) {
    if (!container) return;
    const grid = container.querySelector('[data-role="profile-global-grid"]');
    const empty = container.querySelector('[data-role="profile-global-empty"]');
    if (!grid) return;
    grid.innerHTML = "";

    const has = !!(globals && (
      (globals.round_over_round && globals.round_over_round.rounds && globals.round_over_round.rounds.length > 0) ||
      (globals.family_rollup    && globals.family_rollup.rounds    && globals.family_rollup.rounds.length > 0) ||
      (globals.resources        && globals.resources.rounds        && globals.resources.rounds.length > 0)
    ));
    if (!has) {
      container.hidden = true;
      if (empty) empty.hidden = false;
      return;
    }
    container.hidden = false;
    if (empty) empty.hidden = true;

    if (globals.round_over_round && globals.round_over_round.rounds.length > 0) {
      const c = chartCanvasCell(grid);
      drawRoundOverRound(c, globals.round_over_round);
    }
    if (globals.family_rollup && globals.family_rollup.rounds.length > 0) {
      const c = chartCanvasCell(grid);
      drawFamilyRollup(c, globals.family_rollup);
    }
    if (globals.resources && globals.resources.rounds.length > 0) {
      const c = chartCanvasCell(grid);
      drawResources(c, globals.resources);
    }
  }

  function chartCanvasCell(grid) {
    const wrap = document.createElement("div");
    wrap.className = "chart-wrap";
    const c = document.createElement("canvas");
    wrap.appendChild(c);
    grid.appendChild(wrap);
    return c;
  }

  function drawTopN(canvas, rows) {
    if (!window.Chart || !rows || rows.length === 0) return;
    if (canvas._chart) canvas._chart.destroy();
    const labels = rows.map((r) => r.name_clean);
    canvas._chart = new Chart(canvas.getContext("2d"), {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: "total µs",
          data: rows.map((r) => r.total_us),
          backgroundColor: rows.map((r) => familyColor(r.family, 0.7)),
          borderColor:     rows.map((r) => familyColor(r.family, 1)),
          borderWidth: 1,
        }],
      },
      options: {
        ...barChartOptions("total µs"),
        indexAxis: "y",
        plugins: {
          legend: { display: false },
          title: { display: true, text: "P1 · top kernels", color: CHART.text, font: { size: 11 } },
          tooltip: {
            callbacks: { label: (i) => `${rows[i.dataIndex].name_raw}: ${i.parsed.x.toFixed(1)} µs` },
          },
        },
      },
    });
  }

  function drawRoundOverRound(canvas, panel) {
    if (!window.Chart || !panel || !panel.rounds || panel.rounds.length === 0) return;
    if (canvas._chart) canvas._chart.destroy();
    const labels = panel.rounds.map((r) => "r" + r);
    canvas._chart = new Chart(canvas.getContext("2d"), {
      type: "line",
      data: {
        labels,
        datasets: panel.series.map((s, i) => ({
          label: s.name_clean,
          data: s.totals_us,
          borderColor: cycleColor(i, 1),
          backgroundColor: cycleColor(i, 0.18),
          tension: 0.2,
          pointRadius: 2,
        })),
      },
      options: {
        ...lineChartOptions("total µs"),
        plugins: {
          legend: { labels: { color: CHART.text, font: { size: 10 } } },
          title: { display: true, text: "P2 · top-N over rounds", color: CHART.text, font: { size: 11 } },
        },
      },
    });
  }

  function drawFamilyRollup(canvas, panel) {
    if (!window.Chart || !panel || !panel.rounds || panel.rounds.length === 0) return;
    if (canvas._chart) canvas._chart.destroy();
    const labels = panel.rounds.map((r) => "r" + r);
    canvas._chart = new Chart(canvas.getContext("2d"), {
      type: "line",
      data: {
        labels,
        datasets: panel.families.map((fam) => ({
          label: fam,
          data: panel.series[fam],
          borderColor: familyColor(fam, 1),
          backgroundColor: familyColor(fam, 0.45),
          fill: true,
          tension: 0.2,
          stack: "fam",
        })),
      },
      options: {
        ...lineChartOptions("µs"),
        plugins: {
          legend: { labels: { color: CHART.text, font: { size: 10 } } },
          title: { display: true, text: "P3 · family rollup", color: CHART.text, font: { size: 11 } },
        },
        scales: {
          x: { ticks: { color: CHART.text }, grid: { color: CHART.grid } },
          y: { stacked: true, beginAtZero: true,
               ticks: { color: CHART.text }, grid: { color: CHART.grid } },
        },
      },
    });
  }

  function drawResources(canvas, panel) {
    if (!window.Chart || !panel || !panel.rounds || panel.rounds.length === 0) return;
    if (canvas._chart) canvas._chart.destroy();
    const labels = panel.rounds.map((r) => "r" + r);
    canvas._chart = new Chart(canvas.getContext("2d"), {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "VGPR", data: panel.vgpr, borderColor: cycleColor(0, 1), backgroundColor: cycleColor(0, 0.18), yAxisID: "y" },
          { label: "SGPR", data: panel.sgpr, borderColor: cycleColor(1, 1), backgroundColor: cycleColor(1, 0.18), yAxisID: "y" },
          { label: "LDS",  data: panel.lds,  borderColor: cycleColor(2, 1), backgroundColor: cycleColor(2, 0.18), yAxisID: "y2" },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: { labels: { color: CHART.text, font: { size: 10 } } },
          title: { display: true,
                   text: "P4 · " + (panel.target_clean || "VGPR/SGPR/LDS"),
                   color: CHART.text, font: { size: 11 } },
        },
        scales: {
          x: { ticks: { color: CHART.text }, grid: { color: CHART.grid } },
          y:  { position: "left",  ticks: { color: CHART.text }, grid: { color: CHART.grid } },
          y2: { position: "right", ticks: { color: CHART.text }, grid: { display: false } },
        },
      },
    });
  }

  function drawTreemap(wrap, cells) {
    wrap.innerHTML = "";
    if (!cells || cells.length === 0) return;
    cells.forEach((c) => {
      const el = document.createElement("div");
      el.className = "treemap-cell family-" + c.family;
      el.style.left   = (c.x * 100).toFixed(2) + "%";
      el.style.top    = (c.y * 100).toFixed(2) + "%";
      el.style.width  = (c.w * 100).toFixed(2) + "%";
      el.style.height = (c.h * 100).toFixed(2) + "%";
      el.title = `${c.name_raw}\nfamily: ${c.family}\ntotal: ${c.total_us.toFixed(1)} µs · count ${c.count}`;
      el.textContent = c.name_clean;
      wrap.appendChild(el);
    });
  }

  function buildDiffTable(currentRound, diffs) {
    const wrap = document.createElement("div");

    const controls = document.createElement("div");
    controls.className = "diff-controls";
    controls.appendChild(textNode("R-vs-R diff: "));

    const sel = document.createElement("select");
    diffs.pairs.forEach((p, i) => {
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = `r${p.left_round} → r${p.right_round}`;
      sel.appendChild(opt);
    });
    controls.appendChild(sel);
    wrap.appendChild(controls);

    const tableWrap = document.createElement("div");
    wrap.appendChild(tableWrap);

    function paint(idx) {
      const pair = diffs.pairs[idx];
      tableWrap.innerHTML = "";
      const t = document.createElement("table");
      t.className = "diff-table";
      t.innerHTML = `<tr>
        <th>kernel</th>
        <th>r${pair.left_round} µs</th>
        <th>r${pair.right_round} µs</th>
        <th>Δ%</th>
      </tr>`;
      pair.rows.slice(0, 20).forEach((row) => {
        const tr = document.createElement("tr");
        tr.appendChild(td(row.name_clean));
        tr.appendChild(td(row.left_total_us == null ? "—" : row.left_total_us.toFixed(1)));
        tr.appendChild(td(row.right_total_us == null ? "—" : row.right_total_us.toFixed(1)));
        const dCell = document.createElement("td");
        if (row.delta_pct == null) {
          dCell.textContent = "—";
        } else {
          dCell.textContent = (row.delta_pct >= 0 ? "+" : "") + row.delta_pct.toFixed(1) + "%";
          dCell.className = row.delta_pct >= 0 ? "delta-pos" : "delta-neg";
        }
        tr.appendChild(dCell);
        t.appendChild(tr);
      });
      tableWrap.appendChild(t);
    }

    sel.addEventListener("change", () => paint(Number(sel.value)));
    // Pre-select pair whose right side equals the current round, fallback to last pair.
    const startIdx = Math.max(0, diffs.pairs.findIndex((p) => p.right_round === currentRound));
    sel.value = String(startIdx);
    paint(startIdx);
    return wrap;
  }

  function textNode(s) { return document.createTextNode(s); }

  function familyColor(fam, alpha) {
    const a = alpha == null ? 1 : alpha;
    switch (fam) {
      case "fwd_dgrad":   return `rgba(90,168,255,${a})`;
      case "wgrad":       return `rgba(180,131,245,${a})`;
      case "quant":       return `rgba(245,185,92,${a})`;
      case "amax":        return `rgba(241,124,138,${a})`;
      case "elementwise": return `rgba(93,212,158,${a})`;
      case "other":       return `rgba(154,160,166,${a})`;
      default:            return `rgba(154,160,166,${a})`;
    }
  }

  const _PALETTE = [
    "rgba(90,168,255,",
    "rgba(180,131,245,",
    "rgba(245,185,92,",
    "rgba(93,212,158,",
    "rgba(241,124,138,",
    "rgba(180,180,180,",
  ];
  function cycleColor(i, alpha) {
    return _PALETTE[i % _PALETTE.length] + alpha + ")";
  }

  function td(text, cls) {
    const el = document.createElement("td");
    if (cls) el.className = cls;
    el.textContent = text;
    return el;
  }

  function extractStepGeomean(r) {
    if (r.score && typeof r.score.step_geomean === "number") return r.score.step_geomean;
    if (r.perf && typeof r.perf.step_geomean === "number")   return r.perf.step_geomean;
    return null;
  }

  // ---------------------------------------------------------------
  // Cost panel (panel 2)
  // ---------------------------------------------------------------
  function renderCostPanel(container, panel) {
    const totals = container.querySelector('[data-role="cost-totals"]');
    const cumCanvas = container.querySelector('[data-role="cost-cumulative"]');
    const phaseCanvas = container.querySelector('[data-role="cost-per-phase"]');
    const empty = container.querySelector('[data-role="cost-empty"]');
    if (!panel || !panel.cumulative || panel.cumulative.length === 0) {
      if (empty) empty.hidden = false;
      if (cumCanvas) cumCanvas.style.display = "none";
      if (phaseCanvas) phaseCanvas.style.display = "none";
      return;
    }
    if (empty) empty.hidden = true;

    if (totals) {
      totals.innerHTML = "";
      totals.appendChild(totalChip("total cost USD", "$" + Number(panel.total_usd).toFixed(4)));
      totals.appendChild(totalChip("total wall s", Number(panel.total_wall_s || 0).toFixed(1)));
      totals.appendChild(totalChip("total turns", String(panel.total_turns || 0)));
    }

    if (window.Chart && cumCanvas) {
      if (cumCanvas._chart) cumCanvas._chart.destroy();
      const cumPoints = panel.cumulative.map((p) => ({
        x: Date.parse(p.x), y: p.y, _phase: p.phase, _round: p.round,
      }));
      cumCanvas._chart = new Chart(cumCanvas.getContext("2d"), {
        type: "line",
        data: {
          datasets: [{
            label: "cumulative USD",
            data: cumPoints,
            borderColor: "rgba(245,185,92,0.9)",
            backgroundColor: "rgba(245,185,92,0.15)",
            fill: true,
            stepped: true,
            pointRadius: 2,
          }],
        },
        options: lineChartOptions("cumulative USD", { isLinearTime: true }),
      });
    }

    if (window.Chart && phaseCanvas) {
      if (phaseCanvas._chart) phaseCanvas._chart.destroy();
      const labels = panel.per_phase.map((p) => p.phase);
      const costData = panel.per_phase.map((p) => p.cost_usd);
      const total = costData.reduce((s, v) => s + (v || 0), 0);
      phaseCanvas._chart = new Chart(phaseCanvas.getContext("2d"), {
        type: "pie",
        data: {
          labels,
          datasets: [{
            label: "cost USD by phase",
            data: costData,
            backgroundColor: labels.map((p) => phaseColor(p, 0.85)),
            borderColor: "#ffffff",
            borderWidth: 1,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          plugins: {
            legend: {
              position: "right",
              labels: { color: CHART.text, font: { size: 11 }, boxWidth: 10 },
            },
            tooltip: {
              callbacks: {
                label: (i) => {
                  const v = Number(i.parsed) || 0;
                  const pct = total > 0 ? ((v / total) * 100).toFixed(1) : "0.0";
                  return `${i.label}: $${v.toFixed(4)}  (${pct}%)`;
                },
              },
            },
          },
        },
      });
    }
  }

  function totalChip(label, value) {
    const el = document.createElement("span");
    el.textContent = label + ": ";
    const strong = document.createElement("strong");
    strong.textContent = value;
    el.appendChild(strong);
    return el;
  }

  // ---------------------------------------------------------------
  // Gantt (panel 3) — one row per round, segments coloured by status
  // ---------------------------------------------------------------
  function renderGantt(container, panel) {
    const wrap = container.querySelector('[data-role="gantt-wrap"]');
    const empty = container.querySelector('[data-role="gantt-empty"]');
    let footer = container.querySelector('[data-role="gantt-paginator"]');
    if (!footer) {
      footer = document.createElement("div");
      footer.setAttribute("data-role", "gantt-paginator");
      container.appendChild(footer);
    }
    if (!wrap) return;
    wrap.innerHTML = "";
    footer.innerHTML = "";
    if (!panel || !panel.blocks || panel.blocks.length === 0) {
      if (empty) empty.hidden = false;
      return;
    }
    if (empty) empty.hidden = true;

    const legend = container.querySelector('[data-role="gantt-legend"]');
    if (legend) {
      legend.innerHTML = "";
      const seen = new Set();
      const phaseOrder = [
        "DEFINE_TARGET", "PREPARE_ENVIRONMENT", "BASELINE",
        "ANALYZE", "OPTIMIZE", "VALIDATE", "DECIDE",
        "REPORT", "DONE",
      ];
      panel.blocks.forEach((b) => {
        const root = (b.phase || "").split(/\s|\(/)[0].toUpperCase();
        if (root) seen.add(root);
      });
      const ordered = phaseOrder.filter((p) => seen.has(p));
      seen.forEach((p) => { if (!ordered.includes(p)) ordered.push(p); });
      ordered.forEach((p) => {
        const item = document.createElement("span");
        item.className = "gantt-legend-item";
        const swatch = document.createElement("span");
        swatch.className = "gantt-legend-swatch";
        swatch.style.background = phaseColor(p, 0.85);
        item.appendChild(swatch);
        const label = document.createElement("span");
        label.textContent = p.toLowerCase();
        item.appendChild(label);
        legend.appendChild(item);
      });
    }

    // Group blocks by round. ``round=null`` blocks (DEFINE_TARGET,
    // PREPARE_ENVIRONMENT, …) collapse into a single "init" lane that
    // sorts before round 1.
    const groups = new Map();
    for (const b of panel.blocks) {
      const key = b.round == null ? "init" : `r${b.round}`;
      if (!groups.has(key)) {
        groups.set(key, { key, round: b.round, blocks: [] });
      }
      groups.get(key).blocks.push(b);
    }
    const rows = Array.from(groups.values());
    rows.sort((a, b) => {
      if (a.round == null && b.round != null) return -1;
      if (a.round != null && b.round == null) return 1;
      if (a.round == null && b.round == null) return 0;
      return a.round - b.round;
    });
    // Pre-compute per-row aggregates for label + tooltip use.
    rows.forEach((row) => {
      row.blocks.sort((a, b) => Date.parse(a.start_ts) - Date.parse(b.start_ts));
      const t0 = Math.min(...row.blocks.map((b) => Date.parse(b.start_ts)));
      const t1 = Math.max(...row.blocks.map((b) => Date.parse(b.end_ts)));
      row.t0 = t0;
      row.t1 = t1;
      row.dur_s = (t1 - t0) / 1000;
      row.cost = row.blocks.reduce((s, b) => s + (b.cost_usd || 0), 0);
    });

    paginate(footer, rows, 10, (page) => {
      wrap.innerHTML = "";
      page.forEach((row) => wrap.appendChild(renderGanttRow(row, panel.events || [])));
    });
  }

  // Map cost.md status string to a colour class. Phases are coloured
  // by status rather than by phase so the user can scan the lane for
  // "did this round time-out anywhere".
  function statusClass(status) {
    const s = (status || "").toLowerCase();
    if (s === "ok") return "status-ok";
    if (s === "cached") return "status-cached";
    if (s.startsWith("idle_timeout_retry")) return "status-warn";
    if (s.startsWith("idle_timeout") || s.startsWith("wall_timeout")) return "status-bad";
    if (s.startsWith("error") || s === "interrupted") return "status-bad";
    return "status-other";
  }

  function renderGanttRow(group, events) {
    const row = document.createElement("div");
    row.className = "gantt-row";

    const labelCell = document.createElement("div");
    labelCell.className = "gantt-phase";
    labelCell.textContent = group.key;
    row.appendChild(labelCell);

    const meta = document.createElement("div");
    meta.className = "gantt-round";
    meta.textContent = `${group.dur_s.toFixed(0)}s`;
    if (group.cost) meta.title = `cost $${group.cost.toFixed(4)}`;
    row.appendChild(meta);

    const track = document.createElement("div");
    track.className = "gantt-bar-track";

    const span = Math.max(group.t1 - group.t0, 1);
    group.blocks.forEach((b) => {
      const left = ((Date.parse(b.start_ts) - group.t0) / span) * 100;
      const right = ((Date.parse(b.end_ts)   - group.t0) / span) * 100;
      const bar = document.createElement("div");
      const cached = (b.status || "").toLowerCase() === "cached";
      bar.className = "gantt-bar"
                    + (cached ? " cached" : "")
                    + (b.abnormal ? " abnormal" : "");
      bar.style.backgroundColor = phaseColor(b.phase, 0.85);
      bar.style.left  = left + "%";
      bar.style.width = Math.max(right - left, 0.4) + "%";
      bar.title = `${b.phase}\nstatus=${b.status}  ${b.dur_s.toFixed(1)}s`
                + (b.cost_usd ? `  $${b.cost_usd.toFixed(4)}` : "");
      const abbr = abbreviatePhase(b.phase);
      if (abbr) bar.textContent = abbr;
      track.appendChild(bar);
    });

    // Debug events overlay — only for blocks in this row's window.
    events.filter((e) => e.ts).forEach((ev) => {
      const t = Date.parse(ev.ts);
      if (Number.isNaN(t) || t < group.t0 || t > group.t1) return;
      const x = ((t - group.t0) / span) * 100;
      const dot = document.createElement("span");
      dot.className = "gantt-event " + ev.kind;
      dot.style.position = "absolute";
      dot.style.top = "0";
      dot.style.bottom = "0";
      dot.style.left = x + "%";
      dot.style.width = "2px";
      dot.style.background = ev.kind === "retry_attempt" ? "#5aa8ff" : "#f17c8a";
      dot.title = ev.kind + (ev.fields ? "  " + JSON.stringify(ev.fields) : "");
      track.appendChild(dot);
    });

    row.appendChild(track);
    return row;
  }

  function abbreviatePhase(phase) {
    if (!phase) return "";
    // Strip parenthesised suffix; take first letters of words.
    const head = phase.split(/\s*\(/)[0];
    const words = head.split(/[_\s]+/).filter(Boolean);
    if (words.length === 1) return words[0].slice(0, 3);
    return words.map((w) => w[0]).join("").slice(0, 4);
  }

  // ---------------------------------------------------------------
  // Heatmap (panel 4)
  // ---------------------------------------------------------------
  function renderHeatmap(container, panel) {
    const wrap = container.querySelector('[data-role="heatmap-wrap"]');
    const empty = container.querySelector('[data-role="heatmap-empty"]');
    let footer = container.querySelector('[data-role="heatmap-paginator"]');
    if (!footer) {
      footer = document.createElement("div");
      footer.setAttribute("data-role", "heatmap-paginator");
      container.appendChild(footer);
    }
    if (!wrap) return;
    wrap.innerHTML = "";
    footer.innerHTML = "";
    if (!panel || !panel.shape_labels || panel.shape_labels.length === 0) {
      if (empty) empty.hidden = false;
      return;
    }
    if (empty) empty.hidden = true;

    let direction = "fwd";
    let currentPage = [];
    paginate(footer, panel.rows || [], 20, (page) => {
      currentPage = page;
      drawHeatmap(wrap, panel, direction, page);
    });
    container.querySelectorAll('input[name="heatmap-direction"]').forEach((inp) => {
      inp.addEventListener("change", (ev) => {
        direction = ev.target.value;
        drawHeatmap(wrap, panel, direction, currentPage);
      });
    });
  }

  function drawHeatmap(wrap, panel, direction, rows) {
    wrap.innerHTML = "";
    rows = rows || panel.rows;
    const table = document.createElement("table");
    table.className = "heatmap-table";

    const head = document.createElement("tr");
    const corner = document.createElement("th");
    corner.className = "row-head";
    corner.textContent = "round \\ shape";
    head.appendChild(corner);
    panel.shape_labels.forEach((lab) => {
      const th = document.createElement("th");
      th.textContent = lab;
      head.appendChild(th);
    });
    table.appendChild(head);

    rows.forEach((row) => {
      const tr = document.createElement("tr");
      const rowHead = document.createElement("th");
      rowHead.className = "row-head";
      rowHead.textContent = "round " + row.round;
      tr.appendChild(rowHead);

      const values = direction === "bwd" ? row.bwd : row.fwd;
      const deltas = direction === "bwd" ? row.delta_bwd : row.delta_fwd;
      const checks = row.check || [];
      values.forEach((val, i) => {
        const td = document.createElement("td");
        td.className = "heatmap-cell";
        if (val == null) {
          td.textContent = "—";
          tr.appendChild(td);
          return;
        }
        const delta = deltas ? deltas[i] : null;
        td.style.background = heatColor(delta);
        const lines = [val.toFixed(1)];
        if (delta != null) {
          lines.push(`(${delta >= 0 ? "+" : ""}${delta.toFixed(1)}%)`);
        }
        td.innerHTML = lines.map((l) => `<div>${l}</div>`).join("");
        if (checks[i] === "FAIL") td.classList.add("fail");
        tr.appendChild(td);
      });
      table.appendChild(tr);
    });

    wrap.appendChild(table);
  }

  function heatColor(delta) {
    if (delta == null) return "transparent";
    const clamp = Math.max(Math.min(delta, 12), -12) / 12;
    if (clamp >= 0) {
      const a = 0.10 + clamp * 0.55;
      return `rgba(93,212,158,${a.toFixed(2)})`;
    } else {
      const a = 0.10 + (-clamp) * 0.55;
      return `rgba(241,124,138,${a.toFixed(2)})`;
    }
  }

  // ---------------------------------------------------------------
  // Token / turn / wall (panel 8)
  // ---------------------------------------------------------------
  function renderTokenTurnWall(container, panel) {
    const empty = container.querySelector('[data-role="ttw-empty"]');
    if (!panel || !panel.rounds || panel.rounds.length === 0) {
      if (empty) empty.hidden = false;
      ["ttw-wall", "ttw-sdk", "ttw-turns"].forEach((role) => {
        const c = container.querySelector(`[data-role="${role}"]`);
        if (c) c.style.display = "none";
      });
      return;
    }
    if (empty) empty.hidden = true;

    const labels = panel.rounds.map((r) => "round " + r);
    const series = [
      ["ttw-wall",  "wall (s)",  panel.wall_s,  "rgba(90,168,255,0.85)"],
      ["ttw-sdk",   "sdk (s)",   panel.sdk_s,   "rgba(245,185,92,0.85)"],
      ["ttw-turns", "turns",     panel.turns,   "rgba(180,131,245,0.85)"],
    ];
    series.forEach(([role, label, data, color]) => {
      const canvas = container.querySelector(`[data-role="${role}"]`);
      if (!canvas || !window.Chart) return;
      if (canvas._chart) canvas._chart.destroy();
      canvas._chart = new Chart(canvas.getContext("2d"), {
        type: "line",
        data: {
          labels,
          datasets: [{
            label, data,
            borderColor: color,
            backgroundColor: color.replace("0.85", "0.18"),
            fill: true,
            tension: 0.2,
            pointRadius: 1.5,
            borderWidth: 1.5,
          }],
        },
        options: miniLineOptions(label),
      });
    });
  }

  // ---------------------------------------------------------------
  // Verified ineffective (panel 7)
  // ---------------------------------------------------------------
  function renderIneffective(container, items) {
    const ul = container.querySelector('[data-role="ineffective-list"]');
    const empty = container.querySelector('[data-role="ineffective-empty"]');
    let footer = container.querySelector('[data-role="ineffective-paginator"]');
    if (!footer) {
      footer = document.createElement("div");
      footer.setAttribute("data-role", "ineffective-paginator");
      container.appendChild(footer);
    }
    if (!ul) return;
    ul.innerHTML = "";
    footer.innerHTML = "";
    if (!items || items.length === 0) {
      if (empty) empty.hidden = false;
      return;
    }
    if (empty) empty.hidden = true;

    paginate(footer, items, 6, (page) => {
      ul.innerHTML = "";
      page.forEach((it) => ul.appendChild(buildIneffectiveItem(it)));
    });
  }

  // Direction strings are wall-of-text bullet points pushed straight
  // through from the agent; cap them at ~280 chars (≈ 3 lines on the
  // narrow sidebar) and let the user click to read the full text.
  // Reasons typically have one paragraph + a bench delta line — also
  // capped to keep the card compact.
  const _INEFFECTIVE_CAP_DIRECTION = 280;
  const _INEFFECTIVE_CAP_REASON = 220;

  function buildIneffectiveItem(it) {
    const li = document.createElement("li");

    const dir = document.createElement("div");
    dir.className = "direction";
    attachExpandable(dir, it.direction || "—", _INEFFECTIVE_CAP_DIRECTION);
    li.appendChild(dir);

    const meta = document.createElement("div");
    meta.className = "meta";
    const parts = [];
    if (it.round != null) parts.push("round " + it.round);
    if (it.at) parts.push(it.at);
    meta.textContent = parts.join(" · ");
    li.appendChild(meta);

    if (it.reason) {
      const reason = document.createElement("div");
      reason.className = "reason";
      attachExpandable(reason, it.reason, _INEFFECTIVE_CAP_REASON);
      li.appendChild(reason);
    }

    if (it.modified_files && it.modified_files.length) {
      const files = document.createElement("div");
      files.className = "modified-files";
      files.textContent = it.modified_files.join(", ");
      li.appendChild(files);
    }
    return li;
  }

  // Render ``text`` into ``el`` truncated at ``cap`` chars, with a
  // trailing "…  show more" affordance that swaps to the full text on
  // click (and back on a second click). Single-shot DOM, no listener
  // re-binding on toggle.
  function attachExpandable(el, text, cap) {
    if (text.length <= cap) {
      el.textContent = text;
      return;
    }
    const head = text.slice(0, cap).replace(/\s+\S*$/, "");
    el.textContent = head;
    const ellipsis = document.createElement("span");
    ellipsis.className = "ellipsis";
    ellipsis.textContent = "… ";
    el.appendChild(ellipsis);
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "expand-toggle";
    toggle.textContent = "show more";
    let expanded = false;
    toggle.addEventListener("click", (ev) => {
      ev.stopPropagation();
      expanded = !expanded;
      if (expanded) {
        el.textContent = text;
        ellipsis.textContent = " ";
        toggle.textContent = "show less";
      } else {
        el.textContent = head;
        ellipsis.textContent = "… ";
        toggle.textContent = "show more";
      }
      el.appendChild(ellipsis);
      el.appendChild(toggle);
    });
    el.appendChild(toggle);
  }

  // ---------------------------------------------------------------
  // Chart-options helpers
  // ---------------------------------------------------------------
  function lineChartOptions(yLabel, opts) {
    const isLinearTime = opts && opts.isLinearTime;
    const xScale = isLinearTime ? {
      type: "linear",
      ticks: {
        color: CHART.text,
        callback: (val) => fmtClock(val),
        maxRotation: 0, autoSkip: true, maxTicksLimit: 6,
      },
      grid: { color: CHART.grid },
    } : {
      ticks: { color: CHART.text },
      grid:  { color: CHART.grid },
    };
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { labels: { color: CHART.text, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            title: (items) => isLinearTime ? fmtClock(items[0].parsed.x) : items[0].label,
            label: (item) => `${item.dataset.label}: ${typeof item.parsed.y === "number" ? item.parsed.y.toFixed(3) : item.parsed.y}`,
          },
        },
      },
      scales: {
        x: xScale,
        y: {
          title: { display: true, text: yLabel, color: CHART.text },
          ticks: { color: CHART.text },
          grid:  { color: CHART.grid },
        },
      },
    };
  }

  // Compact line-chart options for 90px-tall sidebar mini charts.
  // Drops the legend + axis titles entirely; the panel header
  // "Token / Turn / Wall" already names the series.
  function miniLineOptions(label) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => items[0].label,
            label: (item) => `${label}: ${typeof item.parsed.y === "number" ? item.parsed.y.toFixed(2) : item.parsed.y}`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: CHART.text, font: { size: 9 }, maxTicksLimit: 4, maxRotation: 0 },
          grid:  { display: false },
          border: { display: false },
        },
        y: {
          ticks: { color: CHART.text, font: { size: 9 }, maxTicksLimit: 3 },
          grid:  { color: CHART.grid },
          border: { display: false },
          title: { display: false },
        },
      },
    };
  }

  function fmtClock(ms) {
    if (ms == null || Number.isNaN(ms)) return "";
    const d = new Date(ms);
    const pad = (n) => (n < 10 ? "0" + n : "" + n);
    return pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds());
  }

  function barChartOptions(yLabel) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (i) => `${i.label}: ${typeof i.parsed.y === "number" ? i.parsed.y.toFixed(4) : i.parsed.y}` } },
      },
      scales: {
        x: { ticks: { color: CHART.text }, grid: { color: CHART.grid } },
        y: {
          title: { display: true, text: yLabel, color: CHART.text },
          beginAtZero: true,
          ticks: { color: CHART.text },
          grid:  { color: CHART.grid },
        },
      },
    };
  }

  // Per-phase palette. Each phase gets a distinct, light-mode-friendly
  // hue so a Gantt row reads at a glance: define / prepare / baseline /
  // analyze / optimize / validate / decide / report / done.
  function phaseColor(phase, alpha) {
    const a = alpha == null ? 1 : alpha;
    const root = (phase || "").split(/\s|\(/)[0].toUpperCase();
    switch (root) {
      case "DEFINE_TARGET":      return `rgba(107,114,128,${a})`;  // slate
      case "PREPARE_ENVIRONMENT":return `rgba(75,85,99,${a})`;     // dark slate
      case "BASELINE":           return `rgba(20,184,166,${a})`;   // teal
      case "ANALYZE":            return `rgba(245,158,11,${a})`;   // amber
      case "OPTIMIZE":           return `rgba(59,130,246,${a})`;   // blue
      case "VALIDATE":           return `rgba(168,85,247,${a})`;   // purple
      case "DECIDE":             return `rgba(16,185,129,${a})`;   // emerald
      case "REPORT":             return `rgba(236,72,153,${a})`;   // pink
      case "DONE":               return `rgba(34,197,94,${a})`;    // green
      default:                   return `rgba(107,114,128,${a})`;
    }
  }

  // ---------------------------------------------------------------
  // Formatting helpers
  // ---------------------------------------------------------------
  function fmtTflops(v) {
    if (v == null || Number.isNaN(v)) return "—";
    return Number(v).toFixed(3);
  }

  // ---------------------------------------------------------------
  // Overview (multi-campaign) renderers
  // ---------------------------------------------------------------
  function renderKpi(container, kpi) {
    if (!kpi) return;
    container.innerHTML = "";
    const cells = [
      ["active campaigns",  String(kpi.active_campaigns ?? 0)],
      ["cost USD",          "$" + Number(kpi.total_cost_usd || 0).toFixed(2)],
      ["wall hours",        Number(kpi.total_wall_hours || 0).toFixed(1)],
      ["active 24h",        String(kpi.campaigns_active_24h ?? 0)],
    ];
    for (const [label, value] of cells) {
      container.appendChild(cell(label, value));
    }
  }

  function renderFilters(container, rows) {
    container.innerHTML = "";

    const op = filterSelect("op", distinct(rows.map((r) => r.op)));
    const backend = filterSelect("backend", distinct(rows.map((r) => r.backend)));
    const gpu = filterSelect("gpu", distinct(rows.map((r) => r.gpu)));
    const status = filterSelect("status", distinct(rows.map((r) => r.status)));
    const search = document.createElement("input");
    search.type = "search";
    search.placeholder = "search campaign id…";
    search.dataset.role = "search";

    [op, backend, gpu, status, search].forEach((el) => container.appendChild(el));

    return { op, backend, gpu, status, search };
  }

  function filterSelect(label, options) {
    const sel = document.createElement("select");
    sel.dataset.role = "filter-" + label;
    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = label + " (all)";
    sel.appendChild(blank);
    options.filter(Boolean).forEach((v) => {
      const opt = document.createElement("option");
      opt.value = String(v);
      opt.textContent = String(v);
      sel.appendChild(opt);
    });
    return sel;
  }

  function distinct(values) {
    return Array.from(new Set(values.filter((v) => v != null && v !== "")));
  }

  function applyFilters(rows, controls) {
    const q = (controls.search.value || "").toLowerCase().trim();
    return rows.filter((r) => {
      if (controls.op.value && r.op !== controls.op.value) return false;
      if (controls.backend.value && r.backend !== controls.backend.value) return false;
      if (controls.gpu.value && r.gpu !== controls.gpu.value) return false;
      if (controls.status.value && r.status !== controls.status.value) return false;
      if (q && !(r.campaign_id || "").toLowerCase().includes(q)) return false;
      return true;
    });
  }

  function renderOverviewTable(container, rows) {
    const tbody = container.querySelector('[data-role="overview-table"] tbody');
    const empty = container.querySelector('[data-role="overview-empty"]');
    if (!tbody) return;
    tbody.innerHTML = "";
    if (!rows || rows.length === 0) {
      if (empty) empty.hidden = false;
      return;
    }
    if (empty) empty.hidden = true;
    rows.forEach((r) => {
      const tr = document.createElement("tr");

      const idCell = document.createElement("td");
      const a = document.createElement("a");
      a.href = r.href || "#";
      a.textContent = r.campaign_id;
      idCell.appendChild(a);
      tr.appendChild(idCell);

      tr.appendChild(td(r.op || "—"));
      tr.appendChild(td(r.backend || "—"));
      tr.appendChild(td(r.gpu || "—"));

      const statusCell = document.createElement("td");
      const badge = document.createElement("span");
      const phase = (r.status || "").split(" ")[0];
      badge.className = "decision-badge decision-" + (phase || "PENDING");
      badge.textContent = r.status || "—";
      statusCell.appendChild(badge);
      tr.appendChild(statusCell);

      const round = r.current_round != null ? String(r.current_round) : "—";
      const best = r.best_round != null ? String(r.best_round) : "—";
      tr.appendChild(td(`${round} / ${best}`, "tabular"));

      const dpct = r.best_delta_pct != null
        ? (r.best_delta_pct >= 0 ? "+" : "") + r.best_delta_pct.toFixed(2) + "%"
        : "—";
      tr.appendChild(td(dpct, "tabular"));

      tr.appendChild(td(r.cost_usd != null ? "$" + Number(r.cost_usd).toFixed(4) : "—", "tabular"));
      tr.appendChild(td(r.started_at || "—", "tabular"));
      tr.appendChild(td(r.last_update || "—", "tabular"));

      tbody.appendChild(tr);
    });
  }

  // ---------------------------------------------------------------
  // Bootstrap
  // ---------------------------------------------------------------
  function bootstrap(payload) {
    const overviewKpi = document.getElementById("kpi");
    if (overviewKpi) {
      bootstrapOverview(payload);
      return;
    }

    const stickyEl = document.getElementById("p6");
    const perfEl   = document.getElementById("p1");
    const costEl   = document.getElementById("p2");
    const ganttEl  = document.getElementById("p3");
    const heatEl   = document.getElementById("p4");
    const roundsEl = document.getElementById("p5");
    const profileGlobalEl = document.getElementById("p-profile-global");
    const ineffEl  = document.getElementById("p7");
    const ttwEl    = document.getElementById("p8");
    if (stickyEl) renderStickyBar(stickyEl, payload.state, payload.kpi_summary);
    if (perfEl)   renderPerfTrend(perfEl, payload.perf_panel);
    if (costEl)   renderCostPanel(costEl, payload.cost_panel);
    if (ganttEl)  renderGantt(ganttEl, payload.gantt_panel);
    if (heatEl)   renderHeatmap(heatEl, payload.heatmap_panel);
    if (roundsEl) renderRoundsTable(roundsEl, payload.rounds || [],
                                     payload.profile_panels || {},
                                     payload.profile_global || null,
                                     payload.profile_diffs || null);
    if (profileGlobalEl) renderProfileGlobalPanel(profileGlobalEl, payload.profile_global || null);
    if (ineffEl)  renderIneffective(ineffEl, payload.ineffective || []);
    if (ttwEl)    renderTokenTurnWall(ttwEl, payload.token_turn_wall_panel);
  }

  function bootstrapOverview(payload) {
    const kpiEl = document.getElementById("kpi");
    const filtersEl = document.getElementById("filters");
    const tableEl = document.getElementById("campaigns");
    const rows = payload.campaigns || [];

    if (kpiEl) renderKpi(kpiEl, payload.kpi);
    const controls = filtersEl ? renderFilters(filtersEl, rows) : null;

    function repaint() {
      const filtered = controls ? applyFilters(rows, controls) : rows;
      renderOverviewTable(tableEl, filtered);
    }

    if (controls) {
      ["op", "backend", "gpu", "status", "search"].forEach((k) => {
        controls[k].addEventListener("input", repaint);
        controls[k].addEventListener("change", repaint);
      });
    }
    repaint();
  }

  // ---------------------------------------------------------------
  // Watch mode (PR-5): EventSource + live-tail panel
  // ---------------------------------------------------------------
  function isWatchMode() {
    return !!document.querySelector('meta[name="turbo-view-watch"]');
  }

  function dataUrlForReload() {
    return new URL("data.json", window.location.href).toString();
  }

  function detectCampaignId() {
    const m = window.location.pathname.match(/\/c\/([^/]+)\//);
    if (m) return m[1];
    if (PAYLOAD && PAYLOAD.state && PAYLOAD.state.campaign_id) {
      return PAYLOAD.state.campaign_id;
    }
    return "";
  }

  async function refetchAndRebootstrap() {
    let payload;
    try {
      const resp = await fetch(dataUrlForReload(), {cache: "no-store"});
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      payload = await resp.json();
    } catch (err) {
      console.warn("turbo-view: reload fetch failed", err);
      return;
    }
    bootstrap(payload);
    Object.assign(window.__primusTurboView, {PAYLOAD: payload});
    refreshLiveTail();
  }

  // Live tail panel (panel 9). Visible only in watch mode; refreshes
  // on every SSE reload plus a slow background poll so a non-restart
  // file update is still picked up if SSE drops.
  let _liveTailTimer = null;
  let _liveTailInflight = false;

  async function setupLiveTail() {
    const panel = document.getElementById("live-tail");
    if (!panel) return;
    panel.hidden = false;
    const phaseSel = panel.querySelector('[data-role="tail-phase"]');
    const nInp = panel.querySelector('[data-role="tail-n"]');
    const refresh = panel.querySelector('[data-role="tail-refresh"]');
    const status = panel.querySelector('[data-role="tail-status"]');

    if (refresh) refresh.addEventListener("click", refreshLiveTail);
    if (phaseSel) phaseSel.addEventListener("change", refreshLiveTail);
    if (nInp) nInp.addEventListener("change", refreshLiveTail);

    await populatePhases(phaseSel, status);
    refreshLiveTail();
    if (_liveTailTimer) clearInterval(_liveTailTimer);
    _liveTailTimer = setInterval(refreshLiveTail, 5000);
  }

  async function populatePhases(phaseSel, status) {
    if (!phaseSel) return;
    const campaign = detectCampaignId();
    const url = new URL("/phases", window.location.origin);
    if (campaign) url.searchParams.set("campaign", campaign);
    let phases = [];
    try {
      const resp = await fetch(url.toString(), {cache: "no-store"});
      if (resp.ok) {
        const data = await resp.json();
        phases = data.phases || [];
      }
    } catch (err) {
      if (status) status.textContent = "phase discovery failed: " + err.message;
    }

    const prev = phaseSel.value;
    phaseSel.innerHTML = "";

    if (phases.length === 0) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "(no transcript files)";
      phaseSel.appendChild(opt);
      phaseSel.disabled = true;
      return;
    }
    phaseSel.disabled = false;

    // Newest-on-top, but the user may have an existing selection
    // they want to keep across reloads.
    let preferred = prev;
    if (!preferred && PAYLOAD && PAYLOAD.state && PAYLOAD.state.current_phase) {
      preferred = String(PAYLOAD.state.current_phase).split(" ")[0];
    }
    let preferredMatch = "";
    for (const p of phases) {
      const opt = document.createElement("option");
      opt.value = p.phase;
      opt.textContent = p.phase;
      phaseSel.appendChild(opt);
      if (preferred && p.phase.toLowerCase() === preferred.toLowerCase()) {
        preferredMatch = p.phase;
      }
    }
    phaseSel.value = preferredMatch || phases[0].phase;
  }

  async function refreshLiveTail() {
    if (_liveTailInflight) return;
    const panel = document.getElementById("live-tail");
    if (!panel) return;
    const phaseSel = panel.querySelector('[data-role="tail-phase"]');
    const nInp = panel.querySelector('[data-role="tail-n"]');
    const status = panel.querySelector('[data-role="tail-status"]');
    const body = panel.querySelector('[data-role="tail-body"]');
    if (!phaseSel || !body) return;

    const phase = phaseSel.value;
    if (!phase) return;  // nothing to tail
    const n = Math.max(10, Math.min(500, parseInt(nInp && nInp.value, 10) || 50));
    const campaign = detectCampaignId();

    const url = new URL("/tail", window.location.origin);
    url.searchParams.set("phase", phase);
    url.searchParams.set("n", String(n));
    if (campaign) url.searchParams.set("campaign", campaign);

    _liveTailInflight = true;
    try {
      const resp = await fetch(url.toString(), {cache: "no-store"});
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      const data = await resp.json();
      const lines = data.lines || [];
      // Auto-scroll only when the user is already at the bottom,
      // so manual scrollback isn't yanked away on every poll.
      const atBottom = body.scrollTop + body.clientHeight >= body.scrollHeight - 4;
      body.textContent = lines.join("\n");
      if (atBottom) body.scrollTop = body.scrollHeight;
      if (status) {
        status.textContent = data.path
          ? `${lines.length} lines · ${data.path}`
          : "transcript not found";
      }
    } catch (err) {
      if (status) status.textContent = "fetch error: " + (err && err.message);
    } finally {
      _liveTailInflight = false;
    }
  }

  function setupEventSource() {
    if (typeof EventSource === "undefined") return;
    const es = new EventSource("/events");
    es.addEventListener("reload", refetchAndRebootstrap);
    es.addEventListener("error", () => {
      // Browser will auto-reconnect; nothing to do but log once.
      // The keepalive comment from the server keeps idle proxies
      // from eagerly closing the channel.
    });
    window.addEventListener("beforeunload", () => es.close());
  }

  bootstrap(PAYLOAD);
  window.__primusTurboView = { bootstrap, PAYLOAD };

  if (isWatchMode()) {
    setupLiveTail();
    setupEventSource();
  }
})();
