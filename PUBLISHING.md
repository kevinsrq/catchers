# Publishing to PyPI Guide 🚀

Since `catchers` is a Rust-based project using `maturin`, you need to build wheels for different platforms (Linux, macOS, Windows) to make it easily installable for everyone.

## 1. Local Pre-check

First, verify that your package builds correctly in release mode:

```bash
uv run maturin build --release
```

## 2. Authentication

You will need a PyPI account and an API Token.

1. Create an account on [PyPI](https://pypi.org/).
2. Go to **Account Settings** -> **API Tokens** and generate a new token.
3. (Optional but recommended) Save this token in your environment as `MATURIN_PASSWORD`.

## 3. Manual Publishing (Not Recommended for Production)

If you only want to publish for your current platform (e.g., Linux):

```bash
uv run maturin publish
```

*Note: This will only upload the wheel for your current OS.*

## 4. Recommended: GitHub Actions (Automated)

The best way to publish is using a GitHub Action that builds wheels for all platforms automatically when you create a new release tag.

Create a file at `.github/workflows/upload.yml`:

```yaml
name: Publish to PyPI

on:
  push:
    tags:
      - 'v*'

jobs:
  build_wheels:
    name: Build wheels on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Build wheels
        uses: PyO3/maturin-action@v1
        with:
          command: build
          args: --release --out dist --find-interpreter
          sccache: 'true'
      - name: Upload wheels
        uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}
          path: dist

  publish:
    name: Publish to PyPI
    runs-on: ubuntu-latest
    needs: [build_wheels]
    steps:
      - uses: actions/download-artifact@v4
        with:
          merge-multiple: true
          path: dist
      - name: Publish to PyPI
        uses: PyO3/maturin-action@v1
        with:
          command: upload
          args: --non-interactive --skip-existing dist/*
        env:
          MATURIN_PYPI_TOKEN: ${{ secrets.PYPI_API_TOKEN }}
```

### Steps to set up

1. Add `PYPI_API_TOKEN` to your GitHub Repository Secrets.
2. Push your code to GitHub.
3. Tag a version: `git tag v0.1.0 && git push origin v0.1.0`.
4. GitHub Actions will handle the rest!
