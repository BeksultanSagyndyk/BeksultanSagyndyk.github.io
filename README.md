# [beksultan's blog](https://BeksultanSagyndyk.github.io)

Minimal, text-first blog for paper reviews and notes. Built with Jekyll on GitHub
Pages. Old-school [Hacker News](https://news.ycombinator.com/) look — no
JavaScript, no external CSS, one small stylesheet.

## Adding a new post

Each post is one Markdown file in `_posts/`, named `YYYY-MM-DD-title.md`.

**Easiest way (no local setup):** open the repo on GitHub → `_posts/` →
**Add file → Create new file** → name it `2026-07-25-my-review.md` → paste the
template below → **Commit changes**. GitHub Pages rebuilds the site in ~1 minute.

Copy this template:

```markdown
---
layout: post
title: "Paper Review N: <paper name>"
subtitle: "one-line summary"
tags: [efficiency, transformers]
---

[PAPER](https://arxiv.org/abs/xxxx.xxxxx)

Your review in Markdown. Headings, **bold**, lists, and images all work.

![alt text](https://link-to-image.png)
```

Only `title` is strictly required; `subtitle` and `tags` are optional but show up
on the home list and tag pages. To upload an image, drag it into a GitHub issue
or the file editor and paste the URL it gives you (that's how existing posts do
it), or drop it in `assets/img/` and link `/assets/img/name.png`.

## Adding a materials entry

The sidebar's **materials** menu has four sections, each its own list of Markdown
files (works just like paper reviews):

| Section | Folder  | Page URL              |
|---------|---------|-----------------------|
| ML/DS   | `_mlds/`  | `/materials/ml-ds/` |
| Math    | `_math/`  | `/materials/math/`  |
| Algos   | `_algos/` | `/materials/algos/` |
| Other   | `_other/` | `/materials/other/` |

To add an entry, create a Markdown file in the matching folder (e.g.
`_mlds/gradient-boosting.md`) with:

```markdown
---
title: "Gradient boosting notes"
subtitle: "one-line summary (optional)"
date: 2026-07-25
---

Your content in Markdown.
```

It appears automatically on that section's page, newest first. No layout line
needed — the folder sets it. (To rename a section or add another, edit the
`collections:` block in `_config.yml` and the links in `_includes/sidebar.html`.)

## Editing locally (optional)

```
bundle install
bundle exec jekyll serve   # http://localhost:4000
```

## Where things live

- `assets/css/minimal.css` — the entire look. Edit the variables at the top
  (`--accent`, `--side-bg`, `--bg`, `--sidebar-w`, fonts…) to restyle.
- `_includes/sidebar.html` — the dark left rail: nav, materials menu, projects,
  and social links.
- `_layouts/` — `home.html` (paper-reviews list), `post.html`,
  `materials-list.html` (a section list), `material.html` (one entry),
  `page.html`, `base.html` (page shell).
- `_config.yml` — site title, sidebar tagline, `project-links`,
  `social-network-links`, and the materials `collections`.
