# website

Minimal Jekyll site, same stack as teddykoker.com / ryleymcconkey.com.

## Run locally

Needs Ruby plus the development headers (`sudo apt install ruby-dev` on Ubuntu):

    bundle install
    bundle exec jekyll serve --livereload
    # http://localhost:4000

Or without touching the system Ruby, in Docker:

    docker run --rm -p 4000:4000 -p 35729:35729 -v "$PWD":/srv/jekyll \
      -v jekyll-gems:/usr/local/bundle -w /srv/jekyll ruby:3.2 \
      sh -c "bundle install && bundle exec jekyll serve --host 0.0.0.0 --livereload --force_polling --destination /srv/site"

`--destination /srv/site` keeps the generated site inside the container, so the container's root
user never writes into your working tree.

## Deploy

Push to `main`. `.github/workflows/jekyll.yml` builds with Jekyll 4 and deploys to GitHub Pages.
In the repo settings set Pages -> Source to **GitHub Actions** (GitHub's built-in builder is Jekyll 3
and will fail on this Gemfile).

This is a project page, so the repository name is the path segment: renaming the repo moves the
site, and no config change is needed. `actions/configure-pages` reports the base path and the
workflow passes it to the build, which is why `baseurl` stays empty in `_config.yml` and local
serving stays at `/`. Paths in the templates go through `relative_url`, and the hero resolves its
data file relative to its own module URL, so any base path works.

Rename the repo to `Nailujj.github.io` to drop the path segment entirely. For a custom domain, add a
`CNAME` file containing the domain and set `url` in `_config.yml`.

## Layout
- `index.md` — home page (bio, links, interactive hero, news, research)
- `writing/index.md` + `_posts/` — blog; add posts as `_posts/YYYY-MM-DD-slug.md` with `layout: post` (`math: true` enables MathJax)
- `_layouts/` — `default.html` (nav + main) and `post.html`
- `assets/css/style.css` — all styling
- `assets/js/hero.js` — interactive hero (Three.js from a CDN via the import map in `index.md`). Loads one real realization from the thermo-mechanics training set (`assets/data/sample.json`). *Heat flow* solves steady conduction on the grain graph in the browser with the simulation's own material constants and animates the heat crossing each boundary; dragging a crystal or picking a texture re-solves. *Energy* and *Dissipation* show the finite-element fields, an OLS baseline and the residual. The heat-flow field is computed in the page, not read from the dataset; it is checked against the simulated interface dissipation. Regenerate the sample with `tools/ols/`.
- `docs/` — drop talk slides here

## To do
- Fill in GitHub and Google Scholar URLs in `index.md`
