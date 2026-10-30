# Development

## Tests

```bash
uv run pytest            # tests marked gpu are skipped without CuPy, pycolmap and a CUDA device
uv run pytest -m gpu     # only the GPU tests
```

The tests generate synthetic streams with known ground truth: a procedural artwork revealed over time, shown through a moving, flipping, rotating view with overlays, a cursor, a scene cut and a layer transform. They check registration accuracy and canvas reconstruction, and the setup page's server.

The browser version has its own tests, in headless Chrome: SIFT against OpenCV's, register, composite and render, and the page end to end. [web/README.md](../web/README.md#tests) has them.

## The figures

`scripts/figures.py` draws the diagrams in `docs/figures/` and the setup page's example; its docstring has the command that renders the PNGs. The video and screenshots in `docs/media/` are listed, with how each was made, in [docs/media/README.md](media/README.md).

## The website

`web/` is the browser version, deployed to Cloudflare Pages by `.github/workflows/pages.yaml` on each push to main that changes it. It is a port of the engine and follows it.
