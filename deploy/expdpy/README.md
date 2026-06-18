# Deploy the expdpy Shiny app to shinyapps.io

This folder is a self-contained [rsconnect-python](https://docs.posit.co/rsconnect-python/) bundle
that publishes the **Shiny** app (`expdpy.app.ExPdPy`) to [shinyapps.io](https://www.shinyapps.io/)
in **upload mode** — visitors bring their own data; no bundled dataset is shown.

| File | Purpose |
|------|---------|
| `app.py` | Module-level `app = ExPdPy(run=False)` — the entrypoint shinyapps.io serves (`app:app`). |
| `requirements.txt` | Installs `expdpy[app]` from the public GitHub repo (tracks `main`). |

## One-time: install the deploy CLI

```bash
uv tool install rsconnect-python      # or: pixi global install rsconnect-python
```

## One-time: register your shinyapps.io account

Get the token/secret from shinyapps.io → **Account → Tokens**. Run this yourself so the secret stays
private (it is not stored in this repo):

```bash
rsconnect add --account <ACCOUNT> --name shinyapps --token <TOKEN> --secret <SECRET>
rsconnect list                        # verify
```

> `--name shinyapps` here is the **server/account nickname** (a local alias for your shinyapps.io
> account) — *not* the app name. The published app's URL slug comes from the **deploy folder basename**
> (`deploy/expdpy` → `…/expdpy/`).

## Deploy (and redeploy)

Run from the repo root. `--python` points at the Pixi 3.12 interpreter so the recorded Python version
satisfies `requires-python>=3.10`. The bundle's `requirements.txt` is used as-is.

```bash
rsconnect deploy shiny deploy/expdpy \
  --name shinyapps \
  --entrypoint app:app \
  --title expdpy \
  --python /Users/carlos/GitHub/expdpy/.pixi/envs/default/bin/python
```

rsconnect prints the live URL (`https://<account>.shinyapps.io/expdpy/`) when the build finishes (the
first build is slow — heavy deps). Subsequent runs of this command update the same app in place; add
`--new` only to force the creation of a brand-new app.

## Local smoke test (same entrypoint shinyapps.io uses)

```bash
pixi run shiny run deploy/expdpy/app.py   # then open http://127.0.0.1:8000
```

## Notes

- **Tracking `main`:** shinyapps.io caches the Python env by the `requirements.txt` hash. With the
  unpinned git URL, pushing new commits to `main` won't necessarily reinstall `expdpy` on a later
  redeploy. To force an update, pin a commit in `requirements.txt`
  (`...expdpy.git@<sha>`) — that changes the hash and triggers a rebuild.
- **Free tier:** ~1 GB RAM / limited active hours / max 5 apps. The dependency set
  (pyfixest/statsmodels/scipy/plotly) is memory-heavy; move to a paid instance if the build or
  runtime struggles.
