// ============================================================
// Full-Stack Robotics — app
// Hash router (#/, #/learn, #/learn/<track>/<lesson>, #/papers, #/log),
// a small Markdown + KaTeX renderer with callouts, progress kept in
// localStorage, search (/ or ⌘K) and the circular-reveal theme toggle.
// ============================================================

(function () {
  "use strict";

  const $ = (sel, el = document) => el.querySelector(sel);
  const $$ = (sel, el = document) => Array.from(el.querySelectorAll(sel));
  const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ESC[c]);
  const app = $("#app");
  if (new URLSearchParams(location.search).get("theme") === "dark") document.documentElement.dataset.theme = "dark";
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---------- indexes ----------
  const TRACK = {};
  const LESSONS = [];
  TRACKS.forEach((tr) => {
    TRACK[tr.id] = tr;
    tr.lessons.forEach((l, i) => LESSONS.push(Object.assign({}, l, { key: tr.id + "/" + l.id, track: tr, idx: i })));
  });
  const LESSON = Object.fromEntries(LESSONS.map((l) => [l.key, l]));
  const PAPER = Object.fromEntries(PAPERS.map((p) => [p.id, p]));
  const PAPER_LESSONS = {};
  LESSONS.forEach((l) => (l.papers || []).forEach((id) => {
    if (!PAPER[id]) console.warn(`[curriculum] ${l.key} cites unknown paper "${id}"`);
    (PAPER_LESSONS[id] = PAPER_LESSONS[id] || []).push(l);
  }));
  const trackLessons = (tr) => LESSONS.filter((l) => l.track === tr);
  const lessonNo = (l) => `${l.track.num}.${l.idx + 1}`;
  const lessonHref = (l) => `#/learn/${l.key}`;
  const paperUrl = (p) => p.url || "https://scholar.google.com/scholar?q=" + encodeURIComponent(p.t);

  // ---------- progress (localStorage) ----------
  const store = {
    get(k, d) { try { const v = JSON.parse(localStorage.getItem(k)); return v ?? d; } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} },
  };
  const K = { done: "fsr-done-v1", read: "fsr-read-v1", last: "fsr-last-v1" };
  const done = new Set(store.get(K.done, []));
  const read = new Set(store.get(K.read, []));
  function toggleIn(set, storeKey, id) {
    if (set.has(id)) set.delete(id); else set.add(id);
    store.set(storeKey, [...set]);
    return set.has(id);
  }
  function progress(lessons) {
    const n = lessons.length;
    const d = lessons.filter((l) => done.has(l.key)).length;
    return { n, d, pct: n ? Math.round((100 * d) / n) : 0, ready: lessons.filter((l) => l.ready).length };
  }
  const meter = (pct, cls = "") => `<i class="meter ${cls}" aria-hidden="true"><b style="width:${pct}%"></b></i>`;

  // ---------- markdown + math ----------
  const CALLOUT = { one: "一句话", tip: "提示", note: "补充", robot: "机器人视角", pitfall: "常见坑", quiz: "自测", mine: "我的笔记" };
  if (window.hljs) hljs.configure({ ignoreUnescapedHTML: true });

  function tex(src, display) {
    if (!window.katex) return `<code>${esc(src)}</code>`;
    try { return katex.renderToString(src, { displayMode: display, throwOnError: false, strict: "ignore" }); }
    catch (e) { return `<code>${esc(src)}</code>`; }
  }

  function renderMarkdown(src) {
    // Pull math out before Markdown sees it (so _ * | \\ survive), but never inside code.
    const math = [];
    const stash = (t, display) => { math.push(tex(t.trim(), display)); return `@@M${math.length - 1}@@`; };
    const parts = String(src).split(/(```[\s\S]*?```|`[^`\n]*`)/g);
    for (let i = 0; i < parts.length; i += 2) {
      parts[i] = parts[i]
        .replace(/\$\$([\s\S]+?)\$\$/g, (_, t) => stash(t.replace(/^[ \t]*>[ \t]?/gm, ""), true))
        .replace(/\\\$/g, "@@DOLLAR@@")
        .replace(/\$([^\s$](?:[^$\n]*?[^\s$\\])?)\$/g, (_, t) => stash(t, false));
    }
    let html = window.marked ? marked.parse(parts.join(""), { gfm: true }) : `<pre>${esc(src)}</pre>`;
    html = html.replace(/@@M(\d+)@@/g, (_, i) => math[+i]).replace(/@@DOLLAR@@/g, "$");
    const box = document.createElement("div");
    box.innerHTML = html;
    enhance(box);
    return box;
  }

  function enhance(root) {
    // > [!type] callouts; > [!quiz] becomes a <details>
    $$("blockquote", root).forEach((bq) => {
      const first = bq.firstElementChild;
      if (!first || first.tagName !== "P") return;
      const m = first.innerHTML.match(/^\s*\[!(\w+)\][ \t]*/);
      if (!m) return;
      const type = m[1].toLowerCase();
      first.innerHTML = first.innerHTML.slice(m[0].length);
      if (type === "quiz") {
        const html = first.innerHTML;
        const nl = html.indexOf("\n");
        const q = nl < 0 ? html : html.slice(0, nl);
        first.innerHTML = nl < 0 ? "" : html.slice(nl + 1);
        if (!first.innerHTML.trim()) first.remove();
        const det = document.createElement("details");
        det.className = "quiz";
        det.innerHTML = `<summary><span class="quiz-q">${q}</span><span class="quiz-toggle" aria-hidden="true"></span></summary><div class="quiz-a"></div>`;
        const a = det.querySelector(".quiz-a");
        while (bq.firstChild) a.appendChild(bq.firstChild);
        bq.replaceWith(det);
        return;
      }
      if (type === "one") first.innerHTML = first.innerHTML.replace(/^一句话[:：]\s*/, "");
      if (type === "mine") first.innerHTML = first.innerHTML.replace(/^我的笔记[:：]\s*/, "");
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
  async function loadLesson(l) {
    if (mdCache.has(l.key)) return mdCache.get(l.key);
    let text = null;
    try {
      const res = await fetch(`content/${l.key}.md`, { cache: "no-cache" });
      if (res.ok) text = await res.text();
    } catch (e) {}
    mdCache.set(l.key, text);
    return text;
  }

  // ---------- shared pieces ----------
  function paperCard(p, opts = {}) {
    if (!p) return "";
    const isRead = read.has(p.id);
    const lessons = opts.compact ? [] : (PAPER_LESSONS[p.id] || []).slice(0, 3);
    return `<article class="paper ${isRead ? "is-read" : ""}">
      <div class="p-meta"><span class="p-year">${p.y || "—"}</span><span class="p-venue">${esc(p.v)}</span>${p.kind === "book" ? '<span class="p-flag">Book</span>' : p.kind === "tool" ? '<span class="p-flag">Tool</span>' : ""}${p.must ? '<span class="p-flag must">必读</span>' : ""}</div>
      <h3 class="p-title"><a href="${esc(paperUrl(p))}" target="_blank" rel="noopener">${esc(p.t)}</a></h3>
      <p class="p-authors">${esc(p.a)}</p>
      <p class="p-why">${esc(p.why)}</p>
      <div class="p-foot">
        ${lessons.map((l) => `<a class="p-lesson" href="${lessonHref(l)}">${lessonNo(l)} ${esc(l.t)}</a>`).join("")}
        <button class="read-btn" type="button" data-read="${esc(p.id)}" aria-pressed="${isRead}">${isRead ? "✓ 读过" : "标记读过"}</button>
      </div>
    </article>`;
  }

  function flywheelSvg() {
    const C = 160, R = 108;
    const nodes = [
      { a: -90, en: "Data", zh: "数据" },
      { a: 0, en: "Learning", zh: "学习" },
      { a: 90, en: "Policy", zh: "策略" },
      { a: 180, en: "Deployment", zh: "部署" },
    ];
    const pt = (deg, r = R) => [C + r * Math.cos((deg * Math.PI) / 180), C + r * Math.sin((deg * Math.PI) / 180)];
    const arcs = nodes.map((n, i) => {
      const [x1, y1] = pt(n.a + 15), [x2, y2] = pt(n.a + 75);
      return `<path class="fw-arc" d="M${x1.toFixed(1)} ${y1.toFixed(1)} A${R} ${R} 0 0 1 ${x2.toFixed(1)} ${y2.toFixed(1)}" marker-end="url(#fwArrow)"/>`;
    }).join("");
    const dots = nodes.map((n) => {
      const [x, y] = pt(n.a);
      const [lx, ly] = pt(n.a, R + 30);
      const anchor = n.a === 0 ? "start" : n.a === 180 ? "end" : "middle";
      const dx = n.a === 0 ? -14 : n.a === 180 ? 14 : 0;
      return `<circle class="fw-halo" cx="${x}" cy="${y}" r="13"/><circle class="fw-node" cx="${x}" cy="${y}" r="5.5"/>
        <text class="fw-en" x="${lx + dx}" y="${ly - 3}" text-anchor="${anchor}">${n.en}</text>
        <text class="fw-zh" x="${lx + dx}" y="${ly + 13}" text-anchor="${anchor}">${n.zh}</text>`;
    }).join("");
    const comet = reducedMotion ? "" : `<circle class="fw-comet" r="3.2"><animateMotion dur="14s" repeatCount="indefinite" path="M${C} ${C - R} A${R} ${R} 0 1 1 ${C - 0.01} ${C - R}"/></circle>`;
    return `<svg class="flywheel" viewBox="-30 -10 380 340" role="img" aria-label="The robot-learning flywheel: data, learning, policy, deployment">
      <defs><marker id="fwArrow" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1 L8 5 L1 9" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
      <circle class="fw-ring" cx="${C}" cy="${C}" r="${R}"/>
      <g class="fw-spin"><circle class="fw-dash" cx="${C}" cy="${C}" r="${R - 16}"/></g>
      ${arcs}${comet}${dots}
      <text class="fw-center" x="${C}" y="${C - 16}" text-anchor="middle">Perception</text>
      <text class="fw-center" x="${C}" y="${C + 4}" text-anchor="middle">Whole-body control</text>
      <text class="fw-center" x="${C}" y="${C + 24}" text-anchor="middle">Dexterous manipulation</text>
    </svg>`;
  }

  function trackCard(tr) {
    const p = progress(trackLessons(tr));
    return `<a class="stack-card" href="#/learn/${tr.id}">
      <span class="sc-num">${tr.num}</span>
      <span class="sc-title">${esc(tr.t)}</span>
      <span class="sc-en">${esc(tr.en)}</span>
      <span class="sc-meta">${p.n} 课 · 已写 ${p.ready}${p.d ? ` · 学完 ${p.d}` : ""}</span>
      ${meter(p.pct)}
    </a>`;
  }

  // ---------- views ----------
  function viewHome() {
    const all = progress(LESSONS);
    const last = LESSON[store.get(K.last, "")];
    const start = LESSONS[0];
    const steps = ["一句话", "直觉", "形式化", "方法", "机器人视角", "常见坑", "自测", "我的笔记"];
    app.innerHTML = `
    <section class="hero" id="hero">
      <div class="container hero-grid">
        <div class="hero-text reveal">
          <p class="eyebrow">${esc(SITE.kicker)}</p>
          <h1 class="display">${esc(SITE.headline)}</h1>
          <figure class="hero-quote">
            <blockquote>“${esc(SITE.statement)}”</blockquote>
            <figcaption>— ${esc(SITE.statementBy)}</figcaption>
          </figure>
          <p class="lead">${esc(SITE.lead)}</p>
          <div class="cta-row">
            <a class="btn btn-primary" href="${lessonHref(last || start)}">${last ? "继续学习" : "从这里开始"} <span aria-hidden="true">→</span></a>
            <a class="btn" href="#/learn">全部路线</a>
            <a class="btn" href="#/papers">论文库</a>
          </div>
          ${last ? `<p class="resume">上次读到 <a href="${lessonHref(last)}">${lessonNo(last)} · ${esc(last.t)}</a></p>` : ""}
        </div>
        <figure class="hero-card glass reveal">
          ${flywheelSvg()}
          <figcaption>
            <span class="label">The question I'm chasing</span>
            <em>${esc(SITE.question)}</em>
          </figcaption>
        </figure>
      </div>
    </section>

    <section class="container stats reveal" aria-label="Site stats">
      <div class="stat"><b>${TRACKS.length}</b><span>路线 Tracks</span></div>
      <div class="stat"><b>${LESSONS.length}</b><span>课 Lessons</span></div>
      <div class="stat"><b>${LESSONS.filter((l) => l.ready).length}</b><span>已写 Written</span></div>
      <div class="stat"><b>${PAPERS.length}</b><span>论文 Papers</span></div>
      <div class="stat"><b>${all.pct}<small>%</small></b><span>我学完了 ${all.d} 课</span></div>
    </section>

    <section class="container section">
      <header class="section-head reveal">
        <p class="eyebrow">The Stack</p>
        <h2 class="section-title">一张全栈地图</h2>
        <p class="section-lead">自下而上，每一层都建立在下一层之上。点开任意一块开始；进度条会随着我学完的课慢慢填满。</p>
      </header>
      <a class="start-banner glass reveal" href="${lessonHref(start)}">
        <span class="sc-num">00</span><span><b>导论 · Orientation</b> —— 先读这张地图怎么用</span><span class="arrow" aria-hidden="true">→</span>
      </a>
      <div class="stack reveal">
        ${STACK.map((row) => `
          <div class="stack-row" data-layer="${row.layer}">
            <div class="stack-label"><span>${esc(row.label)}</span><small>${esc(row.zh)}</small></div>
            <div class="stack-cells">${row.tracks.map((id) => trackCard(TRACK[id])).join("")}</div>
          </div>`).join("")}
      </div>
    </section>

    <section class="container section">
      <header class="section-head reveal">
        <p class="eyebrow">How a lesson reads</p>
        <h2 class="section-title">每一课都走同一条路</h2>
        <p class="section-lead">先用一句话抓住核心，再从直觉走到公式，最后回到真机：会踩什么坑、自己能不能答上来。</p>
      </header>
      <ol class="steps reveal">${steps.map((s, i) => `<li><span class="mono">${String(i + 1).padStart(2, "0")}</span>${s}</li>`).join("")}</ol>
    </section>

    ${LOG.length ? `
    <section class="container section">
      <header class="section-head reveal">
        <p class="eyebrow">Log</p>
        <h2 class="section-title">最近在学</h2>
      </header>
      <div class="log-preview reveal">
        ${LOG.slice(0, 2).map((e) => `<a class="log-card glass" href="#/log"><time>${esc(e.date)}</time><h3>${esc(e.title)}</h3><span class="arrow" aria-hidden="true">→</span></a>`).join("")}
      </div>
    </section>` : ""}`;
  }

  function viewLearn(focus) {
    const all = progress(LESSONS);
    app.innerHTML = `
    <div class="page container wide">
      <header class="page-head reveal">
        <p class="eyebrow">Curriculum</p>
        <h1 class="page-title">学习路线</h1>
        <p class="page-lead">${TRACKS.length} 条路线、${LESSONS.length} 节课，自下而上排列。亮着的是已经写好的课，灰色的还在计划中——但每一课都先列好了核心问题和必读论文。</p>
        <div class="overall">${meter(all.pct, "big")}<span>学完 ${all.d} / ${all.n}</span></div>
      </header>
      <div class="learn-grid">
        <nav class="track-nav" aria-label="Tracks">
          ${TRACKS.map((tr) => {
            const p = progress(trackLessons(tr));
            return `<a href="#/learn/${tr.id}" data-track="${tr.id}"><span class="mono">${tr.num}</span>${esc(tr.t)}<small>${p.d}/${p.n}</small></a>`;
          }).join("")}
        </nav>
        <div class="tracks">
          ${TRACKS.map((tr) => {
            const p = progress(trackLessons(tr));
            return `<section class="track glass" id="track-${tr.id}">
              <header class="track-head">
                <span class="track-num">${tr.num}</span>
                <div class="track-title">
                  <h2>${esc(tr.t)} <small>${esc(tr.en)}</small></h2>
                  <p>${esc(tr.blurb)}</p>
                </div>
                <div class="track-prog">${meter(p.pct)}<span>${p.d}/${p.n}</span></div>
              </header>
              <ol class="lesson-list">
                ${trackLessons(tr).map((l) => `<li><a class="lesson-row ${l.ready ? "is-ready" : "is-planned"} ${done.has(l.key) ? "is-done" : ""}" href="${lessonHref(l)}">
                  <span class="lr-no mono">${lessonNo(l)}</span>
                  <span class="lr-main"><span class="lr-title">${esc(l.t)}</span><span class="lr-en">${esc(l.en)}</span></span>
                  <span class="lv lv-${l.lv}">${LEVELS[l.lv]}</span>
                  <span class="lr-min mono">${l.min}′</span>
                  <span class="lr-state" title="${done.has(l.key) ? "学完了" : l.ready ? "已写好" : "计划中"}">${done.has(l.key) ? "✓" : l.ready ? "●" : "○"}</span>
                </a></li>`).join("")}
              </ol>
            </section>`;
          }).join("")}
        </div>
      </div>
    </div>`;
    if (focus && TRACK[focus]) {
      const el = document.getElementById("track-" + focus);
      if (el) requestAnimationFrame(() => el.scrollIntoView({ block: "start" }));
    }
  }

  function tocHtml(cur) {
    return `<div class="toc-inner">
      <a class="toc-home" href="#/learn">← 全部路线</a>
      ${TRACKS.map((tr) => {
        const p = progress(trackLessons(tr));
        return `<details class="toc-track" ${tr === cur.track ? "open" : ""}>
          <summary><span class="mono">${tr.num}</span><span class="toc-t">${esc(tr.t)}</span><small>${p.d}/${p.n}</small></summary>
          <ol>${trackLessons(tr).map((x) => `<li><a class="toc-link ${x === cur ? "active" : ""} ${x.ready ? "" : "planned"} ${done.has(x.key) ? "done" : ""}" href="${lessonHref(x)}"><i class="dot"></i>${esc(x.t)}</a></li>`).join("")}</ol>
        </details>`;
      }).join("")}
    </div>`;
  }

  async function viewLesson(l, token) {
    store.set(K.last, l.key);
    const i = LESSONS.indexOf(l), prev = LESSONS[i - 1], next = LESSONS[i + 1];
    app.innerHTML = `
    <div class="lesson-layout">
      <aside class="toc" id="toc" aria-label="Contents">${tocHtml(l)}</aside>
      <article class="lesson">
        <button class="toc-btn" id="tocBtn" type="button">☰ 目录</button>
        <nav class="crumb"><a href="#/learn">Learn</a><span>/</span><a href="#/learn/${l.track.id}">${l.track.num} ${esc(l.track.t)}</a></nav>
        <h1 class="lesson-title">${esc(l.t)}</h1>
        <p class="lesson-en">${esc(l.en)}</p>
        <div class="meta-row">
          <span class="lv lv-${l.lv}">${LEVELS[l.lv]}</span>
          <span class="tag">${l.min} min</span>
          <span class="tag mono">${lessonNo(l)}</span>
          ${l.ready ? "" : '<span class="tag tag-planned">计划中</span>'}
        </div>
        <div class="core-q"><span class="label">核心问题</span><p>${esc(l.q)}</p></div>
        <div class="md" id="md"><p class="loading">Loading…</p></div>
        ${(l.papers || []).length ? `<section class="lesson-papers">
          <h2 class="lp-title">相关论文</h2>
          <div class="paper-grid">${l.papers.map((id) => paperCard(PAPER[id], { compact: true })).join("")}</div>
        </section>` : ""}
        <div class="done-row"><button class="btn done-btn" id="doneBtn" type="button"></button></div>
        <nav class="pn">
          ${prev ? `<a class="pn-prev" href="${lessonHref(prev)}"><small>← 上一课 · ${lessonNo(prev)}</small><span>${esc(prev.t)}</span></a>` : "<span></span>"}
          ${next ? `<a class="pn-next" href="${lessonHref(next)}"><small>下一课 · ${lessonNo(next)} →</small><span>${esc(next.t)}</span></a>` : "<span></span>"}
        </nav>
      </article>
      <aside class="outline" id="outline" aria-label="On this page"></aside>
    </div>`;

    const doneBtn = $("#doneBtn");
    const syncDone = () => {
      const d = done.has(l.key);
      doneBtn.classList.toggle("is-done", d);
      doneBtn.textContent = d ? "✓ 已学完 · 点击撤销" : "学完了，标记一下";
      const link = $(".toc-link.active");
      if (link) link.classList.toggle("done", d);
    };
    doneBtn.addEventListener("click", () => { toggleIn(done, K.done, l.key); syncDone(); });
    syncDone();
    $("#tocBtn").addEventListener("click", () => document.body.classList.toggle("toc-open"));
    // centre the current lesson inside the TOC only — never scroll the page itself
    const toc = $("#toc"), active = $(".toc-link.active");
    if (toc && active) toc.scrollTop += active.getBoundingClientRect().top - toc.getBoundingClientRect().top - toc.clientHeight / 2;

    const text = await loadLesson(l);
    if (token !== routeToken) return;
    const mdEl = $("#md");
    if (text == null) {
      mdEl.innerHTML = `<div class="wip glass">
        <p class="eyebrow">Work in progress</p>
        <h3>这一课还在路上</h3>
        <p>还没写。先带着上面的核心问题去读${(l.papers || []).length ? "下面的论文" : "相关资料"}，读完回来把笔记写进 <code>content/${esc(l.key)}.md</code>。</p>
      </div>`;
      return;
    }
    mdEl.replaceChildren(...renderMarkdown(text).childNodes);
    buildOutline(mdEl);
    debugScroll();
  }

  // debug flags for screenshots: ?at=<css selector> hides everything above that element
  // (so it sits at the top of the shot); ?theme=dark
  function debugScroll() {
    const at = new URLSearchParams(location.search).get("at");
    let el = at && $(at);
    if (!el) return;
    $$(".reveal").forEach((r) => r.classList.add("in"));
    for (; el && el !== app && !el.parentElement.classList.contains("lesson-layout"); el = el.parentElement) {
      for (let s = el.previousElementSibling; s; s = s.previousElementSibling) s.style.display = "none";
    }
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

  const paperState = { q: "", tr: "all", must: false, unread: false };
  function viewPapers(trackFilter) {
    if (trackFilter !== undefined) paperState.tr = TRACK[trackFilter] ? trackFilter : "all";
    const counts = {};
    PAPERS.forEach((p) => (p.tr || []).forEach((t) => { counts[t] = (counts[t] || 0) + 1; }));
    app.innerHTML = `
    <div class="page container">
      <header class="page-head reveal">
        <p class="eyebrow">Paper Library</p>
        <h1 class="page-title">论文库</h1>
        <p class="page-lead">${PAPERS.length} 篇手选的论文与教材，每篇附一句「为什么读它」。读过就点一下，进度只存在这台设备的浏览器里。</p>
      </header>
      <div class="paper-controls">
        <input class="paper-search" id="paperSearch" type="search" placeholder="搜标题、作者、会议…" value="${esc(paperState.q)}" autocomplete="off">
        <div class="chips" id="paperChips">
          <button class="chip" data-tr="all" type="button">全部 <small>${PAPERS.length}</small></button>
          ${TRACKS.filter((t) => counts[t.id]).map((t) => `<button class="chip" data-tr="${t.id}" type="button">${esc(t.en)} <small>${counts[t.id]}</small></button>`).join("")}
        </div>
        <div class="toggles">
          <label><input type="checkbox" id="mustOnly" ${paperState.must ? "checked" : ""}> 只看必读</label>
          <label><input type="checkbox" id="unreadOnly" ${paperState.unread ? "checked" : ""}> 只看未读</label>
          <span class="paper-count" id="paperCount"></span>
        </div>
      </div>
      <div class="paper-list" id="paperList"></div>
    </div>`;
    const draw = () => {
      const terms = paperState.q.toLowerCase().split(/\s+/).filter(Boolean);
      const list = PAPERS.filter((p) => {
        if (paperState.tr !== "all" && !(p.tr || []).includes(paperState.tr)) return false;
        if (paperState.must && !p.must) return false;
        if (paperState.unread && read.has(p.id)) return false;
        const hay = `${p.t} ${p.a} ${p.v} ${p.y} ${p.id} ${p.why}`.toLowerCase();
        return terms.every((t) => hay.includes(t));
      }).sort((a, b) => (b.y || 0) - (a.y || 0) || a.t.localeCompare(b.t));
      $("#paperList").innerHTML = list.length ? list.map((p) => paperCard(p)).join("") : `<p class="empty">没有匹配的论文。</p>`;
      const r = list.filter((p) => read.has(p.id)).length;
      $("#paperCount").textContent = `${list.length} 篇 · 读过 ${r}`;
      $$("#paperChips .chip").forEach((c) => c.classList.toggle("active", c.dataset.tr === paperState.tr));
    };
    $("#paperSearch").addEventListener("input", (e) => { paperState.q = e.target.value; draw(); });
    $("#paperChips").addEventListener("click", (e) => {
      const c = e.target.closest(".chip");
      if (!c) return;
      paperState.tr = c.dataset.tr;
      history.replaceState(null, "", paperState.tr === "all" ? "#/papers" : `#/papers/${paperState.tr}`);
      draw();
    });
    $("#mustOnly").addEventListener("change", (e) => { paperState.must = e.target.checked; draw(); });
    $("#unreadOnly").addEventListener("change", (e) => { paperState.unread = e.target.checked; draw(); });
    draw();
  }

  function viewLog() {
    app.innerHTML = `
    <div class="page container narrow">
      <header class="page-head reveal">
        <p class="eyebrow">Log</p>
        <h1 class="page-title">学习日志</h1>
        <p class="page-lead">学了什么、卡在哪、下一步做什么。写在 <code>js/site.js</code> 的 LOG 里，最新的在最上面。</p>
      </header>
      <ol class="log-list">
        ${LOG.map((e, i) => `<li class="log-entry">
          <time class="mono">${esc(e.date)}</time>
          <div class="log-body glass">
            <h2>${esc(e.title)}</h2>
            ${(e.tags || []).length ? `<div class="log-tags">${e.tags.map((t) => `<span class="tag">${esc(t)}</span>`).join("")}</div>` : ""}
            <div class="md" data-log="${i}"></div>
          </div>
        </li>`).join("")}
      </ol>
    </div>`;
    $$("[data-log]").forEach((el) => el.replaceChildren(...renderMarkdown(LOG[+el.dataset.log].body).childNodes));
  }

  function viewFrontier() {
    const F = window.FRONTIER || { areas: [] };
    app.innerHTML = `
    <div class="page container">
      <header class="page-head reveal">
        <p class="eyebrow">State of the field · ${esc(F.updated || "")}</p>
        <h1 class="page-title">前沿速览</h1>
        <p class="page-lead">${esc(F.lead || "")}</p>
      </header>
      <nav class="chips frontier-jump" aria-label="Areas">
        ${F.areas.map((a) => `<a class="chip" href="#" data-jump="fr-${esc(a.id)}">${esc(a.t)}</a>`).join("")}
      </nav>
      <div class="frontier">
        ${F.areas.map((a, i) => `<section class="fr-area glass reveal" id="fr-${esc(a.id)}">
          <header class="fr-head">
            <span class="track-num">${String(i + 1).padStart(2, "0")}</span>
            <div><h2>${esc(a.t)} <small>${esc(a.en)}</small></h2>
            ${(a.tracks || []).length ? `<p class="fr-tracks">${a.tracks.filter((id) => TRACK[id]).map((id) => `<a href="#/learn/${id}">${TRACK[id].num} ${esc(TRACK[id].t)} →</a>`).join("")}</p>` : ""}</div>
          </header>
          <div class="md fr-body" data-fr="${i}"></div>
        </section>`).join("")}
      </div>
    </div>`;
    $$("[data-fr]").forEach((el) => {
      const a = F.areas[+el.dataset.fr];
      el.replaceChildren(...renderMarkdown(a.points.map((p) => "- " + p).join("\n")).childNodes);
    });
    $(".frontier-jump").addEventListener("click", (e) => {
      const c = e.target.closest("[data-jump]");
      if (!c) return;
      e.preventDefault();
      document.getElementById(c.dataset.jump).scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "start" });
    });
  }

  function viewNotFound() {
    app.innerHTML = `<div class="page container narrow"><header class="page-head">
      <p class="eyebrow">404</p><h1 class="page-title">这里还什么都没有</h1>
      <p class="page-lead"><a href="#/">回到首页</a> 或者 <a href="#/learn">看看学习路线</a>。</p></header></div>`;
  }

  // ---------- router ----------
  let routeToken = 0;
  function render() {
    const token = ++routeToken;
    const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
    const view = parts[0] || "home";
    document.body.classList.remove("toc-open");
    $$("[data-nav]").forEach((a) => a.classList.toggle("active", a.dataset.nav === view));
    let title = "";
    if (view === "home") viewHome();
    else if (view === "learn" && parts.length >= 3) {
      const l = LESSON[parts[1] + "/" + parts[2]];
      if (l) { title = l.t; viewLesson(l, token); } else viewNotFound();
    } else if (view === "learn") { title = parts[1] && TRACK[parts[1]] ? TRACK[parts[1]].t : "学习路线"; viewLearn(parts[1]); }
    else if (view === "papers") { title = "论文库"; viewPapers(parts[1] || "all"); }
    else if (view === "log") { title = "学习日志"; viewLog(); }
    else if (view === "frontier") { title = "前沿速览"; viewFrontier(); }
    else viewNotFound();
    document.title = (title ? title + " — " : "") + "Full-Stack Robotics · Shijie Zhou";
    if (!(view === "learn" && parts.length === 2)) window.scrollTo(0, 0);
    reveal();
    onScroll();
    debugScroll();
  }
  window.addEventListener("hashchange", render);

  // read toggles anywhere (paper cards)
  app.addEventListener("click", (e) => {
    const b = e.target.closest("[data-read]");
    if (!b) return;
    const on = toggleIn(read, K.read, b.dataset.read);
    b.setAttribute("aria-pressed", String(on));
    b.textContent = on ? "✓ 读过" : "标记读过";
    b.closest(".paper").classList.toggle("is-read", on);
    const count = $("#paperCount");
    if (count && location.hash.startsWith("#/papers")) {
      const n = $$(".paper", $("#paperList")).length;
      count.textContent = `${n} 篇 · 读过 ${$$(".paper.is-read", $("#paperList")).length}`;
    }
  });

  // ---------- scroll reveal ----------
  const io = "IntersectionObserver" in window && !reducedMotion
    ? new IntersectionObserver((es) => es.forEach((en) => {
        if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); }
      }), { rootMargin: "0px 0px -6% 0px" })
    : null;
  function reveal() { $$(".reveal", app).forEach((el) => (io ? io.observe(el) : el.classList.add("in"))); }

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
  const INDEX = [
    ...TRACKS.map((tr) => ({ kind: "路线", no: tr.num, title: tr.t, sub: tr.en, href: `#/learn/${tr.id}`,
      hay: `${tr.t} ${tr.en} ${tr.id} ${tr.blurb}`.toLowerCase() })),
    ...LESSONS.map((l) => ({ kind: "课", no: lessonNo(l), title: l.t, sub: `${l.en} · ${l.q}`, href: lessonHref(l), ready: l.ready,
      hay: `${l.t} ${l.en} ${l.q} ${l.key} ${l.track.t} ${l.track.en}`.toLowerCase() })),
    ...PAPERS.map((p) => ({ kind: "论文", no: String(p.y || "Book"), title: p.t, sub: `${p.a} · ${p.v}`, href: paperUrl(p), external: true,
      hay: `${p.t} ${p.a} ${p.v} ${p.id} ${p.why}`.toLowerCase() })),
  ];
  let sHits = [], sSel = 0;
  function runSearch() {
    const terms = sInput.value.toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) sHits = INDEX.filter((x) => x.kind === "路线");
    else sHits = INDEX.map((x) => {
      if (!terms.every((t) => x.hay.includes(t))) return null;
      const tl = x.title.toLowerCase();
      let s = terms.reduce((acc, t) => acc + (tl.includes(t) ? 3 : 0), 0);
      if (x.kind === "课") s += 1 + (x.ready ? 0.5 : 0);
      return [s, x];
    }).filter(Boolean).sort((a, b) => b[0] - a[0]).slice(0, 12).map((a) => a[1]);
    sSel = 0;
    sResults.innerHTML = sHits.length
      ? sHits.map((x, i) => `<li><a class="sr ${i === sSel ? "sel" : ""}" href="${esc(x.href)}" ${x.external ? 'target="_blank" rel="noopener"' : ""} data-i="${i}">
          <span class="sr-kind">${x.kind}</span><span class="sr-no mono">${esc(x.no)}</span>
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
    if (e.target === searchEl) closeSearch();
    else if (e.target.closest(".sr")) closeSearch();
  });
  document.addEventListener("keydown", (e) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
    if (e.key === "Escape") { if (!searchEl.hidden) closeSearch(); document.body.classList.remove("toc-open"); }
    else if ((e.key === "k" && (e.metaKey || e.ctrlKey)) || (e.key === "/" && !typing)) { e.preventDefault(); searchEl.hidden ? openSearch() : closeSearch(); }
  });

  // close the mobile TOC when a lesson link is chosen
  document.addEventListener("click", (e) => {
    if (e.target.closest(".toc-link, .toc-home")) document.body.classList.remove("toc-open");
  });

  // ---------- footer ----------
  $("#footerMotto").textContent = SITE.motto;
  $("#footerLinks").innerHTML = SITE.footerLinks.map((l) => `<a href="${esc(l.url)}" ${/^https?:/.test(l.url) ? 'target="_blank" rel="noopener"' : ""}>${esc(l.label)}</a>`).join("");
  $("#footerCopy").textContent = SITE.copyright;

  render();
})();
