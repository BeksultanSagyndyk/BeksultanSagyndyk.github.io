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

## Editing locally (optional)

```
bundle install
bundle exec jekyll serve   # http://localhost:4000
```

## Where things live

- `assets/css/minimal.css` — the entire look. Edit the variables at the top
  (`--bg`, `--bar`, `--maxw`, fonts…) to restyle.
- `_layouts/` — `home.html` (post list), `post.html`, `page.html`, `base.html`
  (page shell).
- `_includes/nav.html`, `footer.html`, `head.html` — top bar, footer, `<head>`.
- `_config.yml` — site title, nav links, social links.
