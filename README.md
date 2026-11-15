# Full-Stack Robotics

Shijie Zhou's field notes on becoming a full-stack roboticist — a personal learning site that runs
from motors and joints, through control, RL, locomotion and manipulation, up to VLA models and
world models.

**Live:** https://jerryplain.github.io/fullstack_robotics/

Pure static HTML/CSS/JS, no build step. Same "paper & glass" design family as
[the homepage](https://jerryplain.github.io/) and [the photography site](https://jerryplain.github.io/photography/).

## Structure

```
index.html            page shell (topbar, footer, search overlay)
css/style.css         all styles; light + dark tokens at the top
js/site.js            site copy + the learning LOG (edit by hand)
js/curriculum.js      TRACKS → lessons: the whole syllabus (edit by hand)
js/papers.js          the paper library (edit by hand)
js/frontier.js        "what's new" snapshot of the field (edit by hand)
js/app.js             hash router, Markdown + KaTeX renderer, views, search, progress
js/stars.js           night-sky canvas for dark mode
content/<track>/<lesson>.md   one Markdown file per lesson
CONTENT_GUIDE.md      how to write a lesson (callouts, quizzes, math)
```

Routes: `#/` · `#/learn` · `#/learn/<track>` · `#/learn/<track>/<lesson>` · `#/papers` · `#/frontier` · `#/log`.

## Writing a lesson

1. Write `content/<track>/<lesson>.md` following [CONTENT_GUIDE.md](CONTENT_GUIDE.md).
2. Set `ready: true` on that lesson in `js/curriculum.js`.
3. Link papers by id in the lesson's `papers: [...]`; add new ones to `js/papers.js`.

Progress ("学完了" / "读过") is stored in the browser's localStorage, per device.

## Local preview

Lessons are fetched at runtime, so open it through a local server rather than `file://`:

```bash
python3 -m http.server 8000
# → http://localhost:8000
```

Screenshot helpers: `?theme=dark` forces dark mode; `?at=<css selector>` hides everything above an
element so it sits at the top of a headless screenshot, e.g. `/?at=%23s6#/learn/rl/ppo`.

Libraries (via jsDelivr): [marked](https://marked.js.org), [KaTeX](https://katex.org),
[highlight.js](https://highlightjs.org).
