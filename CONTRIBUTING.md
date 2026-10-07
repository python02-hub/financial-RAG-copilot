# Contributing

Thanks for considering a contribution — this is a small demonstration project, so the process is intentionally lightweight.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your own ANTHROPIC_API_KEY for testing
```

## Making changes

- **Python backend** (`src/`, `cli.py`, `app.py`): standard PEP 8-ish style, type hints where it helps readability. Run `python -m py_compile src/*.py cli.py app.py` before opening a PR as a basic sanity check.
- **Web app** (`web/template.html`): this is the *source* — never edit `docs/index.html` directly, it's a generated build artifact. After editing the template or anything under `data/`, regenerate it:
  ```bash
  python web/build.py
  ```
  Commit both the template change and the regenerated `docs/index.html` in the same PR.
- **Data**: if you add a new bundled sample filing or price series, keep it real (sourced from SEC EDGAR or a real market data provider) and cite the source in a comment at the top of the file, consistent with the existing `data/filings/AAPL_10K_FY2025.txt`.

## Pull requests

1. Fork the repo and create a branch from `main`.
2. Keep PRs focused — one logical change per PR is easier to review.
3. Describe what changed and why in the PR description. For UI changes to the web app, a screenshot or short clip helps.
4. CI (`.github/workflows/ci.yml`) checks that everything compiles/parses and that `docs/index.html` matches what `web/build.py` would currently produce. If your PR touches the template or sample data, make sure you've rebuilt before pushing.

## Reporting issues

Open a GitHub issue with:
- What you expected vs. what happened
- Whether you were using the Python backend or the web app
- Browser/OS if it's a web app issue, Python version if it's a backend issue

## Code of conduct

Be respectful, assume good faith, keep feedback constructive. Standard open-source etiquette.
