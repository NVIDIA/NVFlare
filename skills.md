# NVFlare Repo Skills

## Main Branch Versioning

- Treat `main` as the development branch for the next NVFlare release.
- On `main`, example `requirements.txt` files may intentionally pin the first upcoming NVFlare version that supports a new feature, even if that package is not published on PyPI yet.
- Do not change those pins back to the latest stable release just to make `pip install -r requirements.txt` succeed.
- When a requirement points to an unreleased NVFlare version, add a prominent note in the example docs telling users to install NVFlare from this repo until that package is published.
