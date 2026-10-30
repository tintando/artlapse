# Command line

Running Artlapse on your own computer. It needs Python 3.12 or later through [uv](https://docs.astral.sh/uv/), and [FFmpeg](https://ffmpeg.org/) on the `PATH`.

```bash
uv sync
uv run artlapse init stream.mp4 work/mystream    # or: artlapse fetch <twitchtracker url> work/mystream
uv run artlapse setup work/mystream              # the setup page: draw the boxes, pick the reference frame
uv run artlapse run work/mystream                # register + composite + render -> work/mystream/timelapse.mp4
```

Each stream gets a work directory. All its settings live in `work/<name>/config.yaml` (`src/artlapse/config.py` says what each does), and every stage writes its outputs there.

## The setup page

`setup` opens the page in the browser. It also takes a work directory that doesn't exist yet; the page then starts with **Load video**, which picks the recording from the files on this computer (it stays where it is). Each change saves itself: the boxes to `config.yaml`, the names given to exclusions to `setup.json`.

**Export setup** saves `config.yaml` with the exclusions' names as one file, and **Import setup** reads it back, into this work directory or another: a setup made on another recording brings its boxes but not its start, end and cuts.

**Make timelapse** at the bottom of the sidebar runs `artlapse run`, shows how far it has got, and plays the result. Its log goes to `setup-run.log`.

- **Workspace**: the part of the screen where the art program shows the canvas. Leave out its panels, and any area where a second view of the canvas sometimes appears.
- **Exclude**: overlays drawn over the workspace that stay there (avatar, chat, name tickers). Nothing excluded is ever tracked or painted onto the canvas. Brief things (alerts, the cursor) need no rectangle, and tracking copes with a missed panel, but a long-sitting one may leave a mark on the canvas.
- **Target**: the canvas area to rebuild. Draw it on a frame where the canvas is unrotated and unflipped. That frame becomes the reference, and its orientation is the output's.
- **Start/End** and **cuts**: optionally limit processing to part of the video, and leave out stretches inside it (`cuts: [[from, to], ...]`). Only for speed: what doesn't show the canvas is skipped anyway. The reference frame can't be in a cut.

## Running again

`run` again after changing settings redoes only the stages the change reaches: a render setting re-renders, an `accumulate` setting composites and renders, a region or `register` setting starts from register. When the register settings are back to those of an earlier run, its track comes back from `history/` instead of being recomputed. `run --force` runs every stage.

Every stage appends a line to `work/<name>/runs.jsonl`: when it ran, which engine (version, git commit, source hash), config (hash) and device, how long it took and what it found. It also keeps its small outputs (`track.npz`, `final.png`, `timelapse.mp4`) with the config under `work/<name>/history/<engine>-<config>/`, so trying another setting or engine version does not lose the earlier result.

## Checking and tuning

```bash
uv run artlapse preview work/mystream              # the boxes drawn on the reference frame
uv run artlapse overlay work/mystream 25:00 30:00  # video of that range with the tracked canvas outline
uv run artlapse render work/mystream               # re-render after changing render settings
uv run artlapse sidebyside work/mystream           # the stream next to the timelapse, in sync
uv run artlapse check work/mystream                # where the track may have slipped
```

`sidebyside` shows each timelapse frame next to the stream at the moment its canvas shows, with the stream time in the corner; the video at the top of the README is one. Given several work directories (streams continuing one drawing, say), it plays them one after another. `check` is explained in [troubleshooting.md](troubleshooting.md).

Times, in the config and on the command line, are seconds or `h:mm:ss`.

## A drawing over several streams

When the artist continues a drawing in a later stream, give that stream its own work directory and set `canvas.continues` to the earlier one (`continues: ../mystream_1`): its canvas starts from that work directory's final canvas instead of empty, placed where the picture sits in this stream's canvas. The placement comes from matching that canvas against this stream's first frames, so the two work directories may have their own reference frames and targets, even with the picture moved or rescaled in the document between them. `sidebyside work/mystream_1 work/mystream_2` plays them one after another.

## GPU

With an NVIDIA GPU, composite runs much faster on it:

```bash
uv sync --extra gpu                              # CuPy (CUDA 13) and pycolmap (COLMAP's SiftGPU)
```

`device` in the config chooses: `auto` (the default) uses the GPU when the `gpu` extra is installed and sees a device, `cpu` and `cuda` force one. On the GPU, composite keeps the canvas there and gives the same canvas as the CPU up to rounding.

Register stays on OpenCV's SIFT unless `register.gpu_features: true`: SiftGPU makes it faster, but its tracks are less reliable (on one test stream it loses the view for minutes that OpenCV follows). Each stage records its device in `runs.jsonl`, and keeps the outputs of a stage that ran on the GPU apart from a CPU run's (`history/<engine>-<config>-cuda/`).

## Sample cache

Decoding the stream recording is a large part of register and composite. The first pass over a stretch of the video keeps a lossless copy of its samples (2 per second) in `~/.cache/artlapse/`, and later runs on the same video, from any work directory, read that copy instead: the same pixels, read back 9–13× faster than decoding the source (1080p). On the test streams that makes register up to 2.9× faster (where decoding is what it waits for) and composite 1.3–1.9×; filling it makes the first run 5–16% slower. It takes about 2.5–3 GB per hour of 1080p stream.

```bash
uv run artlapse cache                            # what is cached, per video
uv run artlapse cache --clear [video ...]        # remove it (all, or for these videos)
```

`ARTLAPSE_CACHE=<dir>` puts the cache elsewhere, `ARTLAPSE_CACHE=off` turns it off.

## Twitch VODs

`artlapse fetch` runs twitch-recover, a separate tool that has to be checked out next to Artlapse (`--recover-dir`, default `~/projects/twitch-recover`). It downloads the VOD to `work/<name>/stream.mp4` and initialises the work directory.
