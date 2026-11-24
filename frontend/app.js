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
  $("#text").value = "Supreme Court stays Allahabad HC order on survey of Shahi Idgah complex";
  $("#url").value = "https://indianexpress.com/article/india/supreme-court-stays-allahabad-hc-order-survey-shahi-idgah/";
};
const ex3Btn = $("#ex3");
if (ex3Btn) {
  ex3Btn.onclick = () => {
    $("#text").value = "Will it rain today in Hyderabad?";
    $("#url").value = "";
  };
}

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
      
      const st = String(a.stance || "NEUTRAL").toUpperCase();
      const isSupport = st === "SUPPORTS" || st === "SUPPORTED";
      const isRefute = st === "REFUTES" || st === "REFUTED";
      const badgeCls = isSupport ? "chip chip-real" : (isRefute ? "chip chip-fake" : "chip");
      const badgeText = isSupport ? "✔ Corroborates" : (isRefute ? "✖ Debunks / Refutes" : "— Related");
      const badge = el("span", badgeCls, badgeText);

      const link = el("a", null, ` ${a.title || a.link}`);
      link.href = a.link;
      link.target = "_blank";
      link.rel = "noopener noreferrer";

      const relPct = Math.round((a.reliability || 0.8) * 100);
      const confPct = Math.round((a.confidence || 0.8) * 100);
      const dateStr = a.published_date && a.published_date !== "unknown" ? ` | Published: ${a.published_date}` : "";
      const meta = el("span", "muted",
        ` [${a.source_domain || "Source"} | Reliability: ${relPct}% | Confidence: ${confPct}%${dateStr}]`
      );
      li.append(badge, link, meta);

      if (a.key_sentence && a.key_sentence !== a.title) {
        const quote = el("blockquote", "key-sentence", `“${a.key_sentence}”`);
        li.append(quote);
      }

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

  // 1. Verdict & Accessible Meter Header
  const v = el("div", "verdict");
  const verdictClass = String(d.verdict_display || d.verdict).replace(/ /g, "-");
  
  const isReal = d.verdict === "Likely real" || d.verdict === "SUPPORTED";
  const isFake = d.verdict === "Likely fake" || d.verdict === "REFUTED";
  const icon = isReal ? "✔ " : (isFake ? "✖ " : (d.verdict === "NOT_A_CLAIM" ? "ℹ " : "? "));
  
  const label = d.verdict_display ? `${d.verdict_display} (${d.verdict.replace(/_/g, " ")})` : d.verdict.replace(/_/g, " ");
  const strong = el("strong", verdictClass, `${icon}${label}`);
  const scoreBadge = el("span", "score-badge", `Credibility: ${d.credibility}/100`);
  
  const mlDetail = el("span", "score-meta", `(ML Prior: ${Math.round(d.p_real * 100)}% | Fused: ${d.credibility}%)`);
  v.append(strong, scoreBadge, mlDetail);

  if (d.verdict_badge && d.verdict_badge !== d.verdict) {
    const badgeChip = el("span", "chip", `Grounding: ${d.verdict_badge}`);
    v.append(badgeChip);
  }
  if (d.category && d.category.label) {
    const catChip = el("span", "chip chip-category", `${d.category.icon || "📌"} ${d.category.label}`);
    v.append(catChip);
  }
  if (d.confidence) {
    const confPill = el("span", `confidence-badge conf-${d.confidence.toLowerCase()}`, `${d.confidence.toUpperCase()} CONFIDENCE`);
    v.append(confPill);
  }

  // Semantic HTML5 <meter> for screen readers & a11y
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
  
  const div40 = el("div", "rule-divider");
  div40.style.left = "40%";
  const div60 = el("div", "rule-divider");
  div60.style.left = "60%";
  
  const needle = el("div", "needle");
  needle.setAttribute("title", `Credibility: ${d.credibility}%`);
  needle.style.left = "0%";
  rule.append(div40, div60, needle);
  
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      needle.style.left = `${Math.min(99.5, Math.max(0.5, d.credibility))}%`;
    });
  });

  // Ticks and scale markers
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

  const sectionDivider = el("hr", "result-divider");
  box.append(v, rule, ticks, sectionDivider);

  // Non-claim Guidance Alert
  if (d.input_type && d.input_type !== "claim") {
    const ncBox = el("div", "alert-uncorroborated");
    let tip = "State a factual assertion that can be corroborated with news evidence.";
    if (d.input_type === "question") {
      tip = "Tip: Rephrase questions as declarative statements of fact (e.g., 'It rained in Hyderabad today' instead of 'Will it rain today?').";
    } else if (d.input_type === "prediction") {
      tip = "Tip: Future forecasts cannot be verified as existing facts. Test retrospective statements or past events.";
    } else if (d.input_type === "opinion") {
      tip = "Tip: Personal opinions and editorials cannot be fact-checked. State objective factual assertions.";
    }
    ncBox.innerHTML = `<strong>Input Gate Notice:</strong> This input was classified as a <strong>${d.input_type.toUpperCase()}</strong>. ${d.reasoning || ""}<br><span class="muted">${tip}</span>`;
    box.append(ncBox);
  }

  // If uncorroborated ceiling applied, display explanation alert
  if (d.uncorroborated_flag) {
    const alert = el("div", "alert-uncorroborated");
    const catLabel = d.category && d.category.label ? d.category.label : "Daily Activity";
    alert.innerHTML = `<strong>⚠️ Uncorroborated ${catLabel} Notice:</strong> Zero corroborating news reports were found from trusted media outlets for this claim. Despite formal grammatical style, this uncorroborated claim cannot be classified as real and is capped at <strong>Unverified (50%)</strong> to prevent false reporting.`;
    box.append(alert);
  }

  // 2. Token Attribution (Explainability)
  if (d.explanation && d.explanation.tokens) {
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
  }

  // 3. Stance Evidence Panel
  box.append(renderEvidencePanel(d));

  // 4. Characteristics & Source Grid
  const grid = el("div", "grid");
  
  // Descriptive Markers
  const m = el("div");
  m.append(el("h3", null, "Text Characteristics"));
  const ul = el("ul", "flags");
  const flags = d.markers && d.markers.flags && d.markers.flags.length ? d.markers.flags : ["Standard formatting and vocabulary"];
  flags.forEach(f => ul.append(el("li", null, f)));
  m.append(ul);

  // Source Reputation
  const s = el("div");
  s.append(el("h3", null, "Source Domain"));
  const src = d.source || {};
  s.append(el("p", null, !src.domain ? "No URL provided — source domain did not affect score."
    : src.score == null ? `${src.domain}: Unregistered in known source index.`
    : `${src.domain}: Verified reliability ${Math.round(src.score * 100)}/100.`));
  if (src.watchlist) {
    s.append(el("p", "err", `Watchlist alert: domain previously flagged ${src.flagged} times.`));
  }
  s.append(el("p", "muted", `Active Model: ${d.model || "Linear SVM"} (${d.latency_ms || 0} ms latency)`));

  // Top Attributed Terms
  if (d.explanation) {
    const terms = el("div");
    terms.append(el("h3", null, "Salient Linguistic Features"));
    (d.explanation.fake_terms || []).forEach(t => terms.append(el("span", "chip chip-fake", t)));
    (d.explanation.real_terms || []).forEach(t => terms.append(el("span", "chip chip-real", t)));
    grid.append(m, s, terms);
  } else {
    grid.append(m, s);
  }
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
      l.rel = "noopener noreferrer";
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
  (d.factchecks || []).forEach(f => {
    const li = el("li");
    const l = el("a", null, f.name);
    l.href = f.link;
    l.target = "_blank";
    l.rel = "noopener noreferrer";
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
      const isReal = r.verdict === "Likely real" || r.verdict === "SUPPORTED";
      const isFake = r.verdict === "Likely fake" || r.verdict === "REFUTED";
      const icon = isReal ? "✔ " : (isFake ? "✖ " : (r.verdict === "NOT_A_CLAIM" ? "ℹ " : "? "));
      const cls = r.verdict.replace(/ /g, "-");
      li.append(
        el("strong", cls, `${icon}${r.verdict}`),
        el("span", null, ` ${r.credibility}/100 — `),
        el("span", "muted", r.snippet)
      );
      return li;
    }));
  } catch (e) {}
}

async function loadLearn() {
  try {
    const r = await fetch("/api/learn/status");
    if (!r.ok) return;
    const s = await r.json();
    const elP = $("#learn");
    if (elP) {
      elP.textContent = `Model status: ${s.model_best} | Total trained samples: ${s.total_samples} | Pending feedback: ${s.pending_feedback}`;
    }
  } catch (e) {}
}

async function loadStats() {
  try {
    const r = await fetch("/api/stats");
    if (!r.ok) return;
    const data = await r.json();
    const table = $("#stats");
    const meta = $("#meta");
    if (meta && data.dataset) {
      meta.textContent = `Baseline model: ${data.best_model} on ${data.dataset.name} (${data.dataset.samples} samples)`;
    }
    if (table && data.models) {
      table.innerHTML = "<tr><th>Model</th><th>Accuracy</th><th>F1</th><th>ROC-AUC</th><th>Train time</th></tr>";
      for (const [mName, m] of Object.entries(data.models)) {
        const tr = el("tr");
        tr.append(
          el("td", null, mName),
          el("td", null, `${(m.accuracy * 100).toFixed(1)}%`),
          el("td", null, `${(m.f1 * 100).toFixed(1)}%`),
          el("td", null, `${(m.roc_auc * 100).toFixed(1)}%`),
          el("td", null, `${m.train_time_sec}s`)
        );
        table.append(tr);
      }
    }
  } catch (e) {}
}

const retrainBtn = $("#retrain");
if (retrainBtn) {
  retrainBtn.onclick = async () => {
    const adminKey = prompt("Enter ADMIN_API_KEY to trigger retraining:");
    if (!adminKey) return;
    try {
      const r = await fetch("/api/retrain", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Admin-Key": adminKey
        },
        body: JSON.stringify({ harvest: false })
      });
      const res = await r.json();
      if (!r.ok) {
        alert(res.error || "Retraining request failed");
      } else {
        alert("Retraining triggered successfully!");
      }
    } catch (e) {
      alert("Error contacting retraining endpoint: " + e.message);
    }
  };
}

loadHistory();
loadStats();
loadLearn();
