# Full-Stack Robotics

Shijie Zhou's learning record: after every project, the knowledge it needed is written up as notes,
filed by field, with the fundamentals worth remembering pulled out. A roadmap generated from those
records shows how the fields light up, one project at a time.

**Live:** https://jerryplain.github.io/fullstack_robotics/

Pure static HTML/CSS/JS, no build step. Same "paper & glass" design family as
[the homepage](https://jerryplain.github.io/) and [the photography site](https://jerryplain.github.io/photography/).

## Structure

```
index.html                  page shell (topbar, footer, search overlay)
css/style.css               all styles; light + dark tokens at the top
js/journal.js               LAYERS, SECTIONS, PROJECTS, NOTES: the data everything is built from (edit by hand)
js/site.js                  footer copy
js/app.js                   hash router, roadmap, Markdown + KaTeX renderer, search
content/projects/<id>.md    one summary per project
content/<section>/<note>.md one file per note
CONTENT_GUIDE.md            how to add a project and its notes
```

Routes: `#/` roadmap · `#/p/<project>` · `#/k` knowledge (`#/k/<section>`) · `#/n/<section>/<note>`.

## Local preview

```bash
python3 -m http.server 8000
# → http://localhost:8000
```

Screenshot helpers: `?theme=dark` forces dark mode; `?still` skips the roadmap's draw-in animation;
`?at=<css selector>` hides everything above an element, e.g. `/?at=%23s3#/n/ctrl/cem`.

Libraries (via jsDelivr): [marked](https://marked.js.org), [KaTeX](https://katex.org),
[highlight.js](https://highlightjs.org).
