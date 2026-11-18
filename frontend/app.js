const $ = s => document.querySelector(s);
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
};

// Example buttons
$("#ex1").onclick = () => {
  $("#text").value = "Scientists confirm hot water cures all cancers, doctors hate this secret!";
  $("#url").value = "";
};
$("#ex2").onclick = () => {
  $("#text").value = "Central bank keeps interest rates unchanged, citing stable inflation and steady economic growth";
  $("#url").value = "https://www.thehindu.com/business/";
};

// Analysis Form Submission
$("#form").addEventListener("submit", async e => {
  e.preventDefault();
  const btn = $("#go");
  btn.disabled = true;
  btn.textContent = "Analyzing credibility...";
  try {
    const r = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: $("#text").value, url: $("#url").value })
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || "Analysis request failed");
    render(d);
    loadHistory();
  } catch (err) {
    const box = $("#result");
    box.hidden = false;
    box.replaceChildren(el("p", "err", err.message));
  } finally {
    btn.disabled = false;
    btn.textContent = "Check credibility";
  }
});

function renderEvidencePanel(d) {
  const panel = el("div", "panel");
  panel.append(el("h3", null, "Live Cross-Referencing & Stance Verification"));

  if (d.uncorroborated_flag) {
    panel.append(el("p", "err", "No corroborating reports found from trusted news outlets for this claim. Credibility capped at Unverified."));
  } else if (!d.confidence_flag) {
    panel.append(el("p", "err", "Low external source coverage — credibility is primarily derived from the calibrated ML classifier."));
  }

  if (d.source_evidence && d.source_evidence.length > 0) {
    const ul = el("ul", "plain");
    d.source_evidence.forEach(a => {
      const li = el("li");
      
      // Accessible badge with shape and color
      const isSupport = a.stance === "SUPPORTS";
      const isRefute = a.stance === "REFUTES";
      const badgeCls = isSupport ? "chip chip-real" : (isRefute ? "chip chip-fake" : "chip");
      const badgeText = isSupport ? "✔ Corroborates" : (isRefute ? "✖ Debunks / Refutes" : "— Related");
      const badge = el("span", badgeCls, badgeText);

      const link = el("a", null, ` ${a.title}`);
      link.href = a.link;
      link.target = "_blank";
      link.rel = "noopener";

      const meta = el("span", "muted",
        ` [${a.source_domain} | Reliability: ${Math.round((a.reliability || 0) * 100)}% | Confidence: ${Math.round((a.confidence || 0) * 100)}%]`
      );
      li.append(badge, link, meta);
      ul.append(li);
    });
    panel.append(ul);
  } else {
    panel.append(el("p", "muted", "No live coverage found from trusted news outlets for this specific claim."));
  }
  return panel;
}

function render(d) {
  const box = $("#result");
  box.hidden = false;
  box.replaceChildren();

  // 1. Verdict & Accessible Meter
  const v = el("div", "verdict");
  const verdictClass = d.verdict.replace(/ /g, "-");
  
  // Accessible icon indicator
  const icon = d.verdict === "Likely real" ? "✔ " : (d.verdict === "Likely fake" ? "✖ " : "? ");
  const strong = el("strong", verdictClass, `${icon}${d.verdict}`);
  const scoreBadge = el("span", "score-badge", `Credibility: ${d.credibility}/100`);
  const mlDetail = el("span", "score-meta", `(ML Prior: ${Math.round(d.p_real * 100)}% | Fused: ${d.credibility}%)`);
  v.append(strong, scoreBadge, mlDetail);

  // Semantic HTML5 <meter> for screen readers & a11y compliance
  const meter = document.createElement("meter");
  meter.min = 0; meter.max = 100; meter.low = 40; meter.high = 60; meter.optimum = 80;
  meter.value = d.credibility;
  meter.className = "sr-only";
  meter.setAttribute("role", "meter");
  meter.setAttribute("aria-valuenow", d.credibility);
  meter.setAttribute("aria-valuemin", "0");
  meter.setAttribute("aria-valuemax", "100");
  meter.setAttribute("aria-label", `Credibility score: ${d.credibility} out of 100`);
  v.append(meter);

  // Visual Gauge Bar with explicit boundary lines and prominent needle
  const rule = el("div", "rule");
  
  // Tier threshold boundary divider lines at 40% and 60%
  const div40 = el("div", "rule-divider");
  div40.style.left = "40%";
  const div60 = el("div", "rule-divider");
  div60.style.left = "60%";
  
  // Visible needle indicator line
  const needle = el("div", "needle");
  needle.setAttribute("title", `Credibility: ${d.credibility}%`);
  needle.style.left = "0%";
  rule.append(div40, div60, needle);
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      needle.style.left = `${Math.min(99.5, Math.max(0.5, d.credibility))}%`;
    });
  });

  // Ticks and scale markers with dedicated height
  const ticks = el("div", "ticks");
  const tickDefs = [
    { label: "Likely Fake (< 40%)", pos: "0%", align: "left" },
    { label: "40%", pos: "40%", align: "center", isDivider: true },
    { label: "Unverified (40–60%)", pos: "50%", align: "center" },
    { label: "60%", pos: "60%", align: "center", isDivider: true },
    { label: "Likely Real (> 60%)", pos: "100%", align: "right" }
  ];
  tickDefs.forEach(t => {
    const s = el("span", t.isDivider ? "tick-div" : null, t.label);
    s.style.position = "absolute";
    s.style.left = t.pos;
    if (t.align === "left") {
      s.style.transform = "none";
    } else if (t.align === "right") {
      s.style.right = "0";
      s.style.left = "auto";
      s.style.transform = "none";
    } else {
      s.style.transform = "translateX(-50%)";
    }
    ticks.append(s);
  });

  // Clean separating divider line under the credibility result section
  const sectionDivider = el("hr", "result-divider");
  box.append(v, rule, ticks, sectionDivider);

  // If uncorroborated ceiling applied, display explanation alert
  if (d.uncorroborated_flag) {
    const alert = el("div", "alert-uncorroborated");
    alert.innerHTML = "<strong>Notice:</strong> Zero corroborating news reports were found from trusted media outlets. Despite formal journalistic vocabulary, this unverified claim cannot be classified as real and is capped at <strong>Unverified (50%)</strong>.";
    box.append(alert);
  }

  // 2. Token Attribution (Explainability)
  box.append(el("h3", null, "Evidence Attribution: Words pushing the prediction"));
  const art = el("p", "article");
  d.explanation.tokens.forEach(tk => {
    const s = el("span", null, tk.t);
    const abs = Math.abs(tk.s);
    if (abs > 0.12) {
      if (tk.s > 0) {
        s.className = "token-real";
        s.style.background = `color-mix(in srgb, var(--real) ${Math.round(abs * 45)}%, transparent)`;
      } else {
        s.className = "token-fake";
        s.style.background = `color-mix(in srgb, var(--fake) ${Math.round(abs * 45)}%, transparent)`;
      }
    }
    art.append(s);
  });
  box.append(art, el("p", "legend", "Solid underline indicates support for authenticity; dashed underline indicates signal toward fake."));

  // 3. Stance Evidence Panel
  box.append(renderEvidencePanel(d));

  // 4. Characteristics & Source Grid
  const grid = el("div", "grid");
  
  // Descriptive Markers
  const m = el("div");
  m.append(el("h3", null, "Text Characteristics"));
  const ul = el("ul", "flags");
  const flags = d.markers.flags && d.markers.flags.length ? d.markers.flags : ["Standard formatting and vocabulary"];
  flags.forEach(f => ul.append(el("li", null, f)));
  m.append(ul);

  // Source Reputation
  const s = el("div");
  s.append(el("h3", null, "Source Domain"));
  const src = d.source;
  s.append(el("p", null, !src.domain ? "No URL provided — source domain did not affect score."
    : src.score == null ? `${src.domain}: Unregistered in known source index.`
    : `${src.domain}: Verified reliability ${Math.round(src.score * 100)}/100.`));
  if (src.watchlist) {
    s.append(el("p", "err", `Watchlist alert: domain previously flagged ${src.flagged} times.`));
  }
  s.append(el("p", "muted", `Active Model: ${d.model} (${d.latency_ms} ms latency)`));

  // Top Attributed Terms
  const terms = el("div");
  terms.append(el("h3", null, "Salient Linguistic Features"));
  d.explanation.fake_terms.forEach(t => terms.append(el("span", "chip chip-fake", t)));
  d.explanation.real_terms.forEach(t => terms.append(el("span", "chip chip-real", t)));

  grid.append(m, s, terms);
  box.append(grid);

  // 5. Related Authentic Coverage & Fact Checks
  const ev = el("div", "grid");
  const rel = el("div");
  rel.append(el("h3", null, "Corroborating Coverage"));
  const rl = el("ul", "plain");
  if (d.related && d.related.length) {
    d.related.forEach(a => {
      const li = el("li");
      const l = el("a", null, a.title);
      l.href = a.link;
      l.target = "_blank";
      l.rel = "noopener";
      li.append(l, el("span", "muted", a.source ? ` — ${a.source}` : ""));
      rl.append(li);
    });
  } else {
    rl.append(el("li", "muted", "No additional news links available."));
  }
  rel.append(rl);

  const fc = el("div");
  fc.append(el("h3", null, "Independent Fact-Checkers"));
  const fl = el("ul", "plain");
  d.factchecks.forEach(f => {
    const li = el("li");
    const l = el("a", null, f.name);
    l.href = f.link;
    l.target = "_blank";
    l.rel = "noopener";
    li.append(l);
    fl.append(li);
  });
  fc.append(fl);
  ev.append(rel, fc);
  box.append(ev);

  // 6. Safe Feedback Staging
  const fb = el("div", "row"), sent = $("#text").value;
  fb.append(el("span", "muted", "Submit verification feedback: "));
  [["Verified Real", 1], ["Verified Fake", 0]].forEach(([name, label]) => {
    const b = el("button", "ghost", name);
    b.type = "button";
    b.onclick = async () => {
      b.disabled = true;
      try {
        const r = await fetch("/api/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: sent, label })
        });
        const res = await r.json();
        fb.replaceChildren(el("span", "muted", res.saved ? "Feedback staged for administrator review." : "Could not stage feedback."));
        loadLearn();
      } catch (e) {
        fb.replaceChildren(el("span", "err", "Failed to submit feedback."));
      }
    };
    fb.append(b);
  });
  box.append(fb);
  box.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function loadHistory() {
  try {
    const r = await fetch("/api/history");
    if (!r.ok) return;
    const rows = await r.json();
    const ul = $("#history");
    if (!rows.length) return;
    ul.replaceChildren(...rows.map(r => {
      const li = el("li");
      const icon = r.verdict === "Likely real" ? "✔ " : (r.verdict === "Likely fake" ? "✖ " : "? ");
      li.append(
        el("strong", r.verdict.replace(/ /g, "-"), `${icon}${r.verdict}`),
        el("span", null, ` ${r.credibility}/100 — `),
        el("span", "muted", r.snippet)
      );
      return li;
    }));
  } catch (err) {
    console.warn("Failed to load history:", err);
  }
}

async function loadStats() {
  try {
    const r = await fetch("/api/stats");
    if (!r.ok) return;
    const d = await r.json();
    $("#meta").textContent = `Dataset: ${d.dataset} (${d.rows} rows + ${d.live_rows || 0} live; test set ${d.test_rows || "?"} rows, static only). Best model: ${d.best}.`;
    const t = $("#stats"), head = el("tr");
    ["Model", "Accuracy", "Precision", "Recall", "F1", "Confusion Matrix"].forEach(h => head.append(el("th", null, h)));
    t.replaceChildren(head);
    d.results.forEach(r => {
      const tr = el("tr");
      [r.model, r.accuracy, r.precision, r.recall, r.f1, JSON.stringify(r.confusion)].forEach(c => tr.append(el("td", null, c)));
      t.append(tr);
    });
  } catch (err) {
    console.warn("Failed to load model stats:", err);
  }
}

// Resilient Exponential Backoff Polling
async function pollRetrainStatus(maxAttempts = 15) {
  let attempt = 0;
  let delay = 1500;
  const learnP = $("#learn");

  while (attempt < maxAttempts) {
    attempt++;
    try {
      const r = await fetch("/api/learn/status");
      if (r.ok) {
        const d = await r.json();
        const msg = `Staged feedback: ${d.pending_feedback || 0} | Harvested RSS: ${d.rss_rows || 0} | Last trained: ${d.last}.` +
          (d.running ? ` [Retraining in progress... attempt ${attempt}/${maxAttempts}]` : "") +
          (d.error ? ` [Error: ${d.error}]` : "");
        learnP.textContent = msg;

        if (!d.running) {
          loadStats();
          return;
        }
      }
    } catch (err) {
      console.warn("Polling error:", err);
    }
    await new Promise(resolve => setTimeout(resolve, delay));
    delay = Math.min(8000, Math.round(delay * 1.4));
  }
  learnP.textContent = "Retraining job is still processing in background. Check back shortly.";
}

async function loadLearn() {
  try {
    const r = await fetch("/api/learn/status");
    if (!r.ok) return;
    const d = await r.json();
    $("#learn").textContent = `Staged feedback: ${d.pending_feedback || 0} | Harvested RSS: ${d.rss_rows || 0} | Last trained: ${d.last}.` +
      (d.running ? " [Retraining...]" : "");
    if (d.running) {
      pollRetrainStatus();
    }
  } catch (err) {
    console.warn("Failed to load learning status:", err);
  }
}

$("#retrain").onclick = async () => {
  const adminKey = prompt("Enter Administrator API Key to trigger retraining:", "veritas-admin-secret-2026");
  if (!adminKey) return;

  const btn = $("#retrain");
  btn.disabled = true;
  try {
    const r = await fetch("/api/retrain", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Admin-Key": adminKey
      },
      body: JSON.stringify({ harvest: true })
    });
    const d = await r.json();
    if (!r.ok) {
      alert(d.error || "Retraining request failed");
      return;
    }
    pollRetrainStatus();
  } catch (e) {
    alert("Error triggering retraining.");
  } finally {
    btn.disabled = false;
  }
};

loadHistory();
loadStats();
loadLearn();
