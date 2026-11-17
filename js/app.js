// ============================================================
// Full-Stack Robotics — app
// Routes: #/  roadmap · #/p/<project> · #/k[/<section>] knowledge
//         #/n/<section>/<note>
// All data comes from js/journal.js; prose from content/*.md.
// ============================================================

(function () {
  "use strict";

  const $ = (sel, el = document) => el.querySelector(sel);
  const $$ = (sel, el = document) => Array.from(el.querySelectorAll(sel));
  const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ESC[c]);
  const app = $("#app");
  const params = new URLSearchParams(location.search);
  if (params.get("theme") === "dark") document.documentElement.dataset.theme = "dark";
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches || params.has("still");

  // ---------- indexes ----------
  const LAYER = {};
  LAYERS.forEach((l, i) => { l.i = i; LAYER[l.id] = l; });
  const SEC = {};
  SECTIONS.forEach((s, i) => {
    s.i = i; SEC[s.id] = s;
    if (!LAYER[s.layer]) console.warn(`[journal] section "${s.id}": unknown layer "${s.layer}"`);
  });
  SECTIONS.forEach((s, i) => { s.groupStart = i === 0 || SECTIONS[i - 1].layer !== s.layer; });

  const PROJ_LIST = PROJECTS.slice().sort((a, b) => String(a.date).localeCompare(String(b.date)));
  const PROJ = {};
  PROJ_LIST.forEach((p, i) => { p.i = i; PROJ[p.id] = p; });

  const NOTE = {};
  NOTES.forEach((n, i) => {
    n.sec = n.id.split("/")[0];
    n.order = i;
    n.remember = n.remember || [];
    NOTE[n.id] = n;
    if (!SEC[n.sec]) console.warn(`[journal] note "${n.id}": unknown section "${n.sec}"`);
    if (!PROJ[n.from]) console.warn(`[journal] note "${n.id}": unknown project "${n.from}"`);
  });
  const VALID = NOTES.filter((n) => SEC[n.sec] && PROJ[n.from]);
  const bySection = (a, b) => SEC[a.sec].i - SEC[b.sec].i || a.order - b.order;

  PROJ_LIST.forEach((p) => {
    p.notes = VALID.filter((n) => n.from === p.id).sort(bySection);
    p.used = (p.uses || []).filter((id) => {
      if (!NOTE[id]) console.warn(`[journal] project "${p.id}" uses unknown note "${id}"`);
      return NOTE[id] && VALID.includes(NOTE[id]) && NOTE[id].from !== p.id;
    }).map((id) => NOTE[id]).sort(bySection);
    p.secIds = [...new Set([...p.notes, ...p.used].map((n) => n.sec))].sort((a, b) => SEC[a].i - SEC[b].i);
    p.lastRow = p.secIds.length ? SEC[p.secIds[p.secIds.length - 1]].i : -1;
  });
  VALID.forEach((n) => { n.usedBy = PROJ_LIST.filter((p) => p.used.includes(n)); });
  SECTIONS.forEach((s) => {
    s.notes = VALID.filter((n) => n.sec === s.id).sort((a, b) => PROJ[a.from].i - PROJ[b.from].i || a.order - b.order);
    const first = PROJ_LIST.find((p) => p.secIds.includes(s.id));
    s.first = first ? first.i : Infinity;
    s.lit = s.first < Infinity;
  });
  const LIT = SECTIONS.filter((s) => s.lit);
  const REMEMBER_N = VALID.reduce((k, n) => k + n.remember.length, 0);
  const LATEST = PROJ_LIST[PROJ_LIST.length - 1];

  const projHref = (p) => `#/p/${p.id}`;
  const noteHref = (n) => `#/n/${n.id}`;
  const secHref = (s) => `#/k/${s.id}`;
  const notesIn = (list, secId) => list.filter((n) => n.sec === secId);
  const gv = (s) => `--g: var(--g-${s.layer})`;          // layer colour for an element

  // ---------- markdown + math ----------
  const CALLOUT = { note: "补充", tip: "提示", pitfall: "坑", idea: "想法" };
  if (window.hljs) hljs.configure({ ignoreUnescapedHTML: true });

  function tex(src, display) {
    if (!window.katex) return `<code>${esc(src)}</code>`;
    try { return katex.renderToString(src, { displayMode: display, throwOnError: false, strict: "ignore" }); }
    catch (e) { return `<code>${esc(src)}</code>`; }
  }

  // Pull math out before Markdown sees it (so _ * | \\ survive), but never inside code.
  function mdHtml(src, inline) {
    const math = [];
    const stash = (t, display) => { math.push(tex(t.trim(), display)); return `@@M${math.length - 1}@@`; };
    const parts = String(src).split(/(```[\s\S]*?```|`[^`\n]*`)/g);
    for (let i = 0; i < parts.length; i += 2) {
      parts[i] = parts[i]
        .replace(/\$\$([\s\S]+?)\$\$/g, (_, t) => stash(t.replace(/^[ \t]*>[ \t]?/gm, ""), true))
        .replace(/\\\$/g, "@@DOLLAR@@")
        .replace(/\$([^\s$](?:[^$\n]*?[^\s$\\])?)\$/g, (_, t) => stash(t, false));
    }
    const text = parts.join("");
    const html = !window.marked ? esc(text) : inline ? marked.parseInline(text, { gfm: true }) : marked.parse(text, { gfm: true });
    return html.replace(/@@M(\d+)@@/g, (_, i) => math[+i]).replace(/@@DOLLAR@@/g, "$");
  }
  const inline = (src) => mdHtml(src, true);

  function renderMarkdown(src) {
    const box = document.createElement("div");
    box.innerHTML = mdHtml(src, false);
    enhance(box);
    return box;
  }

  function enhance(root) {
    $$("blockquote", root).forEach((bq) => {
      const first = bq.firstElementChild;
      if (!first || first.tagName !== "P") return;
      const m = first.innerHTML.match(/^\s*\[!(\w+)\][ \t]*/);
      if (!m) return;
      const type = m[1].toLowerCase();
      first.innerHTML = first.innerHTML.slice(m[0].length);
      if (!first.innerHTML.trim()) first.remove();
      const box = document.createElement("aside");
      box.className = `callout callout-${type}`;
      box.innerHTML = `<div class="callout-label">${esc(CALLOUT[type] || type)}</div>`;
      while (bq.firstChild) box.appendChild(bq.firstChild);
      bq.replaceWith(box);
    });
    $$("table", root).forEach((t) => {
      const w = document.createElement("div");
      w.className = "table-wrap";
      t.replaceWith(w);
      w.appendChild(t);
    });
    $$("a[href]", root).forEach((a) => {
      if (/^https?:/.test(a.getAttribute("href"))) { a.target = "_blank"; a.rel = "noopener"; }
    });
    if (window.hljs) $$("pre code", root).forEach((c) => { try { hljs.highlightElement(c); } catch (e) {} });
  }

  const mdCache = new Map();
  async function loadMd(path) {
    if (mdCache.has(path)) return mdCache.get(path);
    let text = null;
    try {
      const res = await fetch(`content/${path}.md`, { cache: "no-cache" });
      if (res.ok) text = await res.text();
    } catch (e) {}
    mdCache.set(path, text);
    return text;
  }
  const missing = (path) => `<p class="missing">还没写：<code>content/${esc(path)}.md</code></p>`;

  // ---------- small shared pieces ----------
  // a project's "fingerprint": one cell per section, in stack order
  const fingerprint = (p) => `<span class="fp" aria-label="${p.secIds.length} 个领域">${LAYERS.map((l) =>
    `<span class="fp-layer">${SECTIONS.filter((s) => s.layer === l.id).map((s) => {
      const k = notesIn(p.notes, s.id).length, u = notesIn(p.used, s.id).length;
      return `<i class="${k ? "on" : u ? "used" : ""}" style="${gv(s)}" title="${esc(s.t)}${k ? ` · ${k} 篇` : u ? " · 用到旧笔记" : ""}"></i>`;
    }).join("")}</span>`).join("")}</span>`;

  const secChip = (s, k) => `<a class="sec-chip" href="${secHref(s)}" style="${gv(s)}"><i></i>${esc(s.t)}${k ? ` <b>${k}</b>` : ""}</a>`;
  const secChips = (p) => p.secIds.map((id) => secChip(SEC[id], notesIn(p.notes, id).length)).join("");

  const rememberList = (n, cls = "") => n.remember.length
    ? `<ul class="remember ${cls}">${n.remember.map((r) => `<li>${inline(r)}</li>`).join("")}</ul>` : "";

  // ---------- the stack: every section as a brick, lit once a project used it ----------
  function stackFigure() {
    return `<figure class="stack" aria-label="Sections lit so far">
      ${LAYERS.map((l) => `<div class="stack-layer" style="--g: var(--g-${l.id})">
        <span class="stack-name">${esc(l.t)}<small>${esc(l.en)}</small></span>
        <div class="bricks">${SECTIONS.filter((s) => s.layer === l.id).map((s) =>
          `<a class="brick ${s.lit ? "on" : ""}" href="${secHref(s)}" title="${esc(s.en)}">${esc(s.t)}${s.notes.length ? `<b class="mono">${s.notes.length}</b>` : ""}</a>`).join("")}</div>
      </div>`).join("")}
      <figcaption>点亮 <b>${LIT.length}</b> / ${SECTIONS.length} 个领域</figcaption>
    </figure>`;
  }

  // ---------- roadmap ----------
  // One column per project (in time order), one rail per section.
  // A rail is dashed until the first project that touches its section, then solid in the layer's colour.
  // Each project is a station: a spine from its header down to its lowest node.
  function roadmap() {
    const P = PROJ_LIST;
    const head = `<div class="m-corner"></div>` +
      P.map((p) => `<a class="m-head ${p === LATEST ? "latest" : ""}" href="${projHref(p)}" data-p="${esc(p.id)}" style="--c:${p.i}">
          <span class="m-date mono">${esc(p.date)}${p === LATEST ? '<em>最新</em>' : ""}</span>
          <b>${esc(p.t)}</b>
          <small>${p.notes.length} 篇笔记 · ${p.secIds.length} 个领域</small>
          ${p.lastRow >= 0 ? '<i class="m-spine stub"></i>' : ""}
        </a>`).join("") +
      `<div class="m-head m-future"><span>下一站 →</span></div>`;

    const rows = SECTIONS.map((s) => {
      const row = `${gv(s)}${s.groupStart ? "; --gpad: 1.7rem" : ""}`;
      const label = `<a class="m-sec ${s.lit ? "on" : ""} ${s.groupStart ? "g-first" : ""}" href="${secHref(s)}" style="${row}">
          ${s.groupStart ? `<span class="m-layer">${esc(LAYER[s.layer].t)} · ${esc(LAYER[s.layer].en)}</span>` : ""}
          <span class="m-name"><i></i>${esc(s.t)}</span>${s.notes.length ? `<b class="mono">${s.notes.length}</b>` : ""}
        </a>`;
      const cells = P.map((p) => {
        const fresh = notesIn(p.notes, s.id), used = notesIn(p.used, s.id);
        const touched = fresh.length + used.length > 0;
        const spine = s.i < p.lastRow ? "full" : s.i === p.lastRow ? "top" : "";
        return `<div class="m-cell" data-p="${esc(p.id)}" style="${row}; --c:${p.i}">
          <i class="m-rail l ${s.first < p.i ? "on" : ""}"></i><i class="m-rail r ${s.first <= p.i ? "on" : ""}"></i>
          ${spine ? `<i class="m-spine ${spine}"></i>` : ""}
          ${touched ? `<i class="m-node ${fresh.length ? "new" : "used"}" title="${esc(s.t)} · ${fresh.length ? `${fresh.length} 篇新笔记` : "用到以前的笔记"}"></i>
          <ul class="m-notes">
            ${fresh.map((n) => `<li><a href="${noteHref(n)}">${esc(n.t)}</a></li>`).join("")}
            ${used.map((n) => `<li class="used"><a href="${noteHref(n)}">${esc(n.t)}</a></li>`).join("")}
          </ul>` : ""}
        </div>`;
      }).join("");
      const future = `<div class="m-cell m-future" style="${row}; --c:${P.length}"><i class="m-rail full ${s.lit ? "on" : ""}"></i></div>`;
      return label + cells + future;
    }).join("");

    return `<div class="map-panel">
        <div class="map-scroll" id="mapScroll">
          <div class="map ${reducedMotion ? "" : "anim"}" id="map" style="--cols:${P.length}">${head}${rows}</div>
        </div>
        <p class="map-legend">
          <span><i class="lg-node new"></i>新笔记</span>
          <span><i class="lg-node used"></i>复用</span>
          <span><i class="lg-rail on"></i>已点亮</span>
          <span><i class="lg-rail"></i>未涉足</span>
          <span class="lg-layers">${LAYERS.map((l) => `<span style="--g: var(--g-${l.id})"><i class="lg-dot"></i>${esc(l.t)}</span>`).join("")}</span>
        </p>
      </div>`;
  }

  function wireMap() {
    const map = $("#map"), sc = $("#mapScroll");
    if (!map) return;
    // bring the newest station into view, just right of the sticky section labels
    const heads = $$(".m-head[data-p]", map), last = heads[heads.length - 1];
    if (last && sc.scrollWidth > sc.clientWidth) {
      const labels = $(".m-corner", map).getBoundingClientRect().width;
      const want = last.getBoundingClientRect().right - sc.getBoundingClientRect().right + 140;
      sc.scrollLeft += Math.max(0, Math.min(want, last.getBoundingClientRect().left - sc.getBoundingClientRect().left - labels));
    }
    const highlight = (id) => {
      if ((map.dataset.hl || "") === (id || "")) return;
      map.dataset.hl = id || "";
      map.classList.toggle("hovering", !!id);
      $$("[data-p]", map).forEach((c) => c.classList.toggle("hl", c.dataset.p === id));
    };
    map.addEventListener("mouseover", (e) => { const el = e.target.closest("[data-p]"); highlight(el ? el.dataset.p : null); });
    map.addEventListener("mouseleave", () => highlight(null));
  }

  const projCard = (p) => `<li><a class="proj-card ${p === LATEST ? "latest" : ""}" href="${projHref(p)}">
      <span class="pc-top"><time class="mono">${esc(p.date)}</time>${fingerprint(p)}</span>
      <span class="pc-title">${esc(p.t)}</span>
      ${p.sub ? `<span class="pc-sub">${esc(p.sub)}</span>` : ""}
      ${p.one ? `<span class="pc-one">${esc(p.one)}</span>` : ""}
      <span class="pc-foot">${p.notes.length} 篇笔记 · ${p.notes.reduce((k, n) => k + n.remember.length, 0)} 条要记住<span class="arrow" aria-hidden="true">→</span></span>
    </a></li>`;

  // ---------- views ----------
  function viewHome() {
    const first = PROJ_LIST[0];
    app.innerHTML = `
    <section class="hero container wide">
      <div class="hero-text">
        <h1 class="display">Becoming a full&#8209;stack roboticist</h1>
        <p class="motto"><span class="motto-quote">How can we build a scalable robot-learning flywheel that unifies perception, whole-body control and dexterous manipulation?</span><span class="motto-credit">— the question that drives me</span></p>
        <dl class="counters">
          <div><dd>${PROJ_LIST.length}</dd><dt>Projects</dt></div>
          <div><dd>${VALID.length}</dd><dt>Notes</dt></div>
          <div><dd>${REMEMBER_N}</dd><dt>To remember</dt></div>
          <div><dd class="mono since">${first ? esc(first.date.replace("-", ".")) : "—"}</dd><dt>Since</dt></div>
        </dl>
      </div>
      ${stackFigure()}
    </section>

    <section class="container wide section" id="roadmap">
      <h2 class="sec-title">Roadmap</h2>
      ${roadmap()}
    </section>

    <section class="container wide section" id="projects">
      <h2 class="sec-title">Projects</h2>
      ${PROJ_LIST.length ? `<ol class="proj-grid">${PROJ_LIST.slice().reverse().map(projCard).join("")}</ol>` : `<p class="empty">还没有项目。</p>`}
    </section>`;
    wireMap();
  }

  async function viewProject(p, token) {
    const prev = PROJ_LIST[p.i - 1], next = PROJ_LIST[p.i + 1];
    app.innerHTML = `
    <div class="page container narrow">
      <nav class="crumb"><a href="#/">Roadmap</a><span>/</span><span>项目</span></nav>
      <header class="doc-head">
        <p class="eyebrow mono">${esc(p.date)}</p>
        <h1 class="page-title">${esc(p.t)}</h1>
        ${p.full ? `<p class="doc-en">${esc(p.full)}</p>` : p.sub ? `<p class="doc-en">${esc(p.sub)}</p>` : ""}
        <div class="doc-secs">${fingerprint(p)}<span class="doc-chips">${secChips(p)}</span></div>
        ${p.one ? `<p class="doc-one">${esc(p.one)}</p>` : ""}
      </header>
      <div class="md" id="md"><p class="loading">Loading…</p></div>
      ${p.secIds.length ? `<section class="learned">
        <h2 class="block-title">学到了什么 · 按领域</h2>
        ${p.secIds.map((id) => `<div class="ln-sec" style="${gv(SEC[id])}">
          <h3><a href="${secHref(SEC[id])}"><i></i>${esc(SEC[id].t)}</a><small>${esc(SEC[id].en)}</small></h3>
          <ul class="ln-notes">
            ${notesIn(p.notes, id).map((n) => `<li>
              <a class="ln-title" href="${noteHref(n)}">${esc(n.t)}<span class="arrow" aria-hidden="true">→</span></a>
              ${rememberList(n)}
            </li>`).join("")}
            ${notesIn(p.used, id).map((n) => `<li class="used">
              <a class="ln-title" href="${noteHref(n)}">${esc(n.t)}<span class="arrow" aria-hidden="true">→</span></a>
              <span class="tag">用到 · 来自 ${esc(PROJ[n.from].t)}</span>
            </li>`).join("")}
          </ul>
        </div>`).join("")}
      </section>` : ""}
      <nav class="pn">
        ${prev ? `<a class="pn-prev" href="${projHref(prev)}"><small>← 上一个项目 · ${esc(prev.date)}</small><span>${esc(prev.t)}</span></a>` : "<span></span>"}
        ${next ? `<a class="pn-next" href="${projHref(next)}"><small>下一个项目 · ${esc(next.date)} →</small><span>${esc(next.t)}</span></a>` : "<span></span>"}
      </nav>
    </div>`;
    const path = `projects/${p.id}`;
    const text = await loadMd(path);
    if (token !== routeToken) return;
    const mdEl = $("#md");
    if (text == null) { mdEl.innerHTML = missing(path); return; }
    mdEl.replaceChildren(...renderMarkdown(text).childNodes);
    debugScroll();
  }

  function viewKnowledge(focus) {
    const off = SECTIONS.filter((s) => !s.lit);
    app.innerHTML = `
    <div class="page container">
      <header class="page-head">
        <h1 class="page-title">Knowledge</h1>
        <p class="tally">${VALID.length} notes · ${REMEMBER_N} to remember · ${LIT.length} / ${SECTIONS.length} fields</p>
      </header>
      <div class="k-grid">
        <nav class="k-nav" aria-label="Sections">
          ${LAYERS.filter((l) => LIT.some((s) => s.layer === l.id)).map((l) => `
            <p class="k-nav-layer" style="--g: var(--g-${l.id})">${esc(l.t)}</p>
            ${LIT.filter((s) => s.layer === l.id).map((s) => `<a href="${secHref(s)}" style="${gv(s)}"><span><i></i>${esc(s.t)}</span><small class="mono">${s.notes.length}</small></a>`).join("")}`).join("")}
        </nav>
        <div class="k-secs">
          ${LAYERS.filter((l) => LIT.some((s) => s.layer === l.id)).map((l) => `
            <h2 class="k-layer" style="--g: var(--g-${l.id})">${esc(l.t)}<small>${esc(l.en)}</small></h2>
            ${LIT.filter((s) => s.layer === l.id).map((s) => `<section class="k-sec" id="sec-${s.id}" style="${gv(s)}">
              <header class="k-head">
                <h3>${esc(s.t)}<small>${esc(s.en)}</small></h3>
                <p>${s.notes.length} 篇 · 从 <a href="${projHref(PROJ_LIST[s.first])}">${esc(PROJ_LIST[s.first].t)}</a> 开始</p>
              </header>
              <ol class="k-notes">
                ${s.notes.map((n) => `<li>
                  <div class="k-line"><a class="k-title" href="${noteHref(n)}">${esc(n.t)}</a><span class="k-en">${esc(n.en || "")}</span>
                    <a class="k-from" href="${projHref(PROJ[n.from])}">${esc(PROJ[n.from].t)} · ${esc(PROJ[n.from].date)}</a></div>
                  ${rememberList(n, "small")}
                </li>`).join("")}
              </ol>
            </section>`).join("")}`).join("")}
          ${off.length ? `<section class="k-off" id="sec-off">
            <h2>还没碰过</h2>
            <p>${off.map((s) => `<span class="sec-chip off" id="sec-${s.id}" style="${gv(s)}"><i></i>${esc(s.t)}</span>`).join("")}</p>
          </section>` : ""}
        </div>
      </div>
    </div>`;
    if (focus && SEC[focus]) {
      const el = document.getElementById("sec-" + focus);
      if (el) requestAnimationFrame(() => el.scrollIntoView({ block: "start" }));
    }
  }

  function tocHtml(cur) {
    return `<div class="toc-inner">
      <a class="toc-home" href="#/k">← 全部知识</a>
      ${LIT.map((s) => `<details class="toc-sec" style="${gv(s)}" ${s.id === cur.sec ? "open" : ""}>
        <summary><span class="toc-t"><i></i>${esc(s.t)}</span><small class="mono">${s.notes.length}</small></summary>
        <ol>${s.notes.map((x) => `<li><a class="toc-link ${x === cur ? "active" : ""}" href="${noteHref(x)}"><i class="dot"></i>${esc(x.t)}</a></li>`).join("")}</ol>
      </details>`).join("")}
    </div>`;
  }

  async function viewNote(n, token) {
    const p = PROJ[n.from], s = SEC[n.sec];
    const sib = p.notes, j = sib.indexOf(n), prev = sib[j - 1], next = sib[j + 1];
    app.innerHTML = `
    <div class="note-layout">
      <aside class="toc" id="toc" aria-label="Notes">${tocHtml(n)}</aside>
      <article class="note" style="${gv(s)}">
        <button class="toc-btn" id="tocBtn" type="button">☰ 目录</button>
        <nav class="crumb"><a href="#/k">Knowledge</a><span>/</span><a href="${secHref(s)}">${esc(LAYER[s.layer].t)} · ${esc(s.t)}</a></nav>
        <h1 class="note-title">${esc(n.t)}</h1>
        ${n.en ? `<p class="note-en">${esc(n.en)}</p>` : ""}
        <p class="note-from">来自 <a href="${projHref(p)}">${esc(p.t)}</a> · ${esc(p.date)}${n.usedBy.length ? ` · 后来用在 ${n.usedBy.map((q) => `<a href="${projHref(q)}">${esc(q.t)}</a>`).join("、")}` : ""}</p>
        ${n.remember.length ? `<aside class="remember-box"><div class="label">要记住</div>${rememberList(n)}</aside>` : ""}
        <div class="md" id="md"><p class="loading">Loading…</p></div>
        <nav class="pn">
          ${prev ? `<a class="pn-prev" href="${noteHref(prev)}"><small>← 同一个项目 · ${esc(SEC[prev.sec].t)}</small><span>${esc(prev.t)}</span></a>` : "<span></span>"}
          ${next ? `<a class="pn-next" href="${noteHref(next)}"><small>同一个项目 · ${esc(SEC[next.sec].t)} →</small><span>${esc(next.t)}</span></a>` : "<span></span>"}
        </nav>
      </article>
      <aside class="outline" id="outline" aria-label="On this page"></aside>
    </div>`;
    $("#tocBtn").addEventListener("click", () => document.body.classList.toggle("toc-open"));
    const toc = $("#toc"), active = $(".toc-link.active");
    if (toc && active) toc.scrollTop += active.getBoundingClientRect().top - toc.getBoundingClientRect().top - toc.clientHeight / 2;

    const text = await loadMd(n.id);
    if (token !== routeToken) return;
    const mdEl = $("#md");
    if (text == null) { mdEl.innerHTML = missing(n.id); return; }
    mdEl.replaceChildren(...renderMarkdown(text).childNodes);
    buildOutline(mdEl);
    debugScroll();
  }

  let outlineObserver = null;
  function buildOutline(mdEl) {
    const outline = $("#outline");
    const hs = $$("h2", mdEl);
    if (!outline || hs.length < 2) return;
    hs.forEach((h, i) => { h.id = "s" + (i + 1); });
    outline.innerHTML = `<p class="label">On this page</p><ol>${hs.map((h) => `<li><a href="#" data-sec="${h.id}">${esc(h.textContent)}</a></li>`).join("")}</ol>`;
    outline.addEventListener("click", (e) => {
      const a = e.target.closest("[data-sec]");
      if (!a) return;
      e.preventDefault();
      document.getElementById(a.dataset.sec).scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "start" });
    });
    if (outlineObserver) outlineObserver.disconnect();
    outlineObserver = new IntersectionObserver((entries) => {
      entries.forEach((en) => {
        if (!en.isIntersecting) return;
        $$("a", outline).forEach((a) => a.classList.toggle("active", a.dataset.sec === en.target.id));
      });
    }, { rootMargin: "-80px 0px -70% 0px" });
    hs.forEach((h) => outlineObserver.observe(h));
  }

  function viewNotFound() {
    app.innerHTML = `<div class="page container narrow"><header class="page-head">
      <p class="eyebrow">404</p><h1 class="page-title">这里还什么都没有</h1>
      <p class="page-lead"><a href="#/">回到路线图</a> 或者 <a href="#/k">看看知识</a>。</p></header></div>`;
  }

  // screenshot helper: ?at=<css selector> hides everything above that element
  function debugScroll() {
    const at = params.get("at");
    let el = at && $(at);
    if (!el) return;
    for (; el && el !== app && !el.parentElement.classList.contains("note-layout"); el = el.parentElement) {
      for (let s = el.previousElementSibling; s; s = s.previousElementSibling) s.style.display = "none";
    }
  }

  // ---------- router ----------
  let routeToken = 0;
  function render() {
    const token = ++routeToken;
    const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
    const view = parts[0] || "";
    document.body.classList.remove("toc-open");
    let title = "", nav = "";
    if (!view) { nav = "map"; viewHome(); }
    else if (view === "p" && PROJ[parts[1]]) { nav = "map"; title = PROJ[parts[1]].t; viewProject(PROJ[parts[1]], token); }
    else if (view === "k") { nav = "k"; title = SEC[parts[1]] ? SEC[parts[1]].t : "知识"; viewKnowledge(parts[1]); }
    else if (view === "n" && NOTE[parts[1] + "/" + parts[2]] && VALID.includes(NOTE[parts[1] + "/" + parts[2]])) {
      nav = "k"; const n = NOTE[parts[1] + "/" + parts[2]]; title = n.t; viewNote(n, token);
    } else viewNotFound();
    $$("[data-nav]").forEach((a) => a.classList.toggle("active", a.dataset.nav === nav));
    document.title = (title ? title + " — " : "") + "Full-Stack Robotics · Shijie Zhou";
    if (!(view === "k" && parts[1])) window.scrollTo(0, 0);
    onScroll();
    debugScroll();
  }
  window.addEventListener("hashchange", render);

  // ---------- topbar ----------
  const topbar = $("#topbar"), progressEl = $("#progress");
  function onScroll() {
    topbar.classList.toggle("scrolled", scrollY > 4);
    const h = document.documentElement.scrollHeight - innerHeight;
    progressEl.style.transform = `scaleX(${h > 0 ? Math.min(1, scrollY / h) : 0})`;
  }
  addEventListener("scroll", onScroll, { passive: true });
  addEventListener("resize", onScroll);

  // ---------- theme toggle (circular reveal) ----------
  const themeToggle = $("#themeToggle");
  const rootEl = document.documentElement;
  const isDark = () => rootEl.dataset.theme === "dark";
  const syncToggleIcon = () => document.body.classList.toggle("dark-active", isDark());
  function applyTheme(next) {
    rootEl.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (e) {}
    syncToggleIcon();
  }
  themeToggle.addEventListener("click", () => {
    const next = isDark() ? "light" : "dark";
    if (!document.startViewTransition || reducedMotion) { applyTheme(next); return; }
    const r = themeToggle.getBoundingClientRect();
    const x = r.left + r.width / 2, y = r.top + r.height / 2;
    const radius = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y));
    const vt = document.startViewTransition(() => applyTheme(next));
    vt.ready.then(() => rootEl.animate(
      { clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${radius}px at ${x}px ${y}px)`] },
      { duration: 650, easing: "cubic-bezier(0.4, 0, 0.2, 1)", pseudoElement: "::view-transition-new(root)" }
    ));
  });
  syncToggleIcon();

  // ---------- search ----------
  const searchEl = $("#search"), sInput = $("#searchInput"), sResults = $("#searchResults");
  const plain = (s) => String(s).replace(/\$[^$]*\$/g, " ").replace(/[*_`\\]/g, "");
  const INDEX = [
    ...PROJ_LIST.map((p) => ({ kind: "项目", title: p.t, sub: p.sub || p.one || "", href: projHref(p), rank: 3,
      hay: `${p.t} ${p.sub} ${p.full} ${p.one}`.toLowerCase() })),
    ...LIT.map((s) => ({ kind: "领域", title: s.t, sub: `${s.en} · ${s.notes.length} 篇`, href: secHref(s), rank: 2,
      hay: `${s.t} ${s.en} ${s.id}`.toLowerCase() })),
    ...VALID.map((n) => ({ kind: "笔记", title: n.t, sub: `${SEC[n.sec].t} · ${n.en || ""}`, href: noteHref(n), rank: 2.5,
      hay: `${n.t} ${n.en} ${n.id} ${SEC[n.sec].t} ${plain(n.remember.join(" "))}`.toLowerCase() })),
    ...VALID.flatMap((n) => n.remember.map((r) => ({ kind: "要记住", title: plain(r), sub: n.t, href: noteHref(n), rank: 1,
      hay: plain(r).toLowerCase() }))),
  ];
  let sHits = [], sSel = 0;
  function runSearch() {
    const terms = sInput.value.toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) sHits = INDEX.filter((x) => x.kind === "项目" || x.kind === "领域");
    else sHits = INDEX.map((x) => {
      if (!terms.every((t) => x.hay.includes(t))) return null;
      const tl = x.title.toLowerCase();
      return [x.rank + terms.reduce((acc, t) => acc + (tl.includes(t) ? 3 : 0), 0), x];
    }).filter(Boolean).sort((a, b) => b[0] - a[0]).slice(0, 14).map((a) => a[1]);
    sSel = 0;
    sResults.innerHTML = sHits.length
      ? sHits.map((x, i) => `<li><a class="sr ${i === sSel ? "sel" : ""}" href="${esc(x.href)}" data-i="${i}">
          <span class="sr-kind">${x.kind}</span>
          <span class="sr-main"><span class="sr-title">${esc(x.title)}</span><span class="sr-sub">${esc(x.sub)}</span></span></a></li>`).join("")
      : `<li class="sr-empty">没找到。换个关键词试试？</li>`;
  }
  function moveSel(d) {
    if (!sHits.length) return;
    sSel = (sSel + d + sHits.length) % sHits.length;
    $$(".sr", sResults).forEach((a, i) => a.classList.toggle("sel", i === sSel));
    const cur = $(".sr.sel", sResults);
    if (cur) cur.scrollIntoView({ block: "nearest" });
  }
  function openSearch() { searchEl.hidden = false; document.body.classList.add("no-scroll"); sInput.value = ""; runSearch(); sInput.focus(); }
  function closeSearch() { searchEl.hidden = true; document.body.classList.remove("no-scroll"); }
  $("#searchBtn").addEventListener("click", openSearch);
  sInput.addEventListener("input", runSearch);
  sInput.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); moveSel(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); moveSel(-1); }
    else if (e.key === "Enter") { const a = $(".sr.sel", sResults); if (a) { e.preventDefault(); a.click(); } }
  });
  searchEl.addEventListener("click", (e) => {
    if (e.target === searchEl || e.target.closest(".sr")) closeSearch();
  });
  document.addEventListener("keydown", (e) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
    if (e.key === "Escape") { if (!searchEl.hidden) closeSearch(); document.body.classList.remove("toc-open"); }
    else if ((e.key === "k" && (e.metaKey || e.ctrlKey)) || (e.key === "/" && !typing)) { e.preventDefault(); searchEl.hidden ? openSearch() : closeSearch(); }
  });

  // close the mobile TOC when a note link is chosen
  document.addEventListener("click", (e) => {
    if (e.target.closest(".toc-link, .toc-home")) document.body.classList.remove("toc-open");
  });

  // ---------- footer ----------
  $("#footerMotto").textContent = SITE.motto;
  $("#footerLinks").innerHTML = SITE.footerLinks.map((l) => `<a href="${esc(l.url)}" ${/^https?:/.test(l.url) ? 'target="_blank" rel="noopener"' : ""}>${esc(l.label)}</a>`).join("");
  $("#footerCopy").textContent = SITE.copyright;

  render();
})();
