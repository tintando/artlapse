# Troubleshooting

Most problems in a timelapse come from the boxes; the README's [When something looks wrong](../README.md#when-something-looks-wrong) lists those. This page is for the rest: a canvas that jumps, smears or rebuilds itself for a stretch. That is usually a slip in the track: for a stretch of the stream, register placed the view in the wrong part of the canvas.

## Find it

Look at the stream at that time first: `artlapse sidebyside` shows which stream moment each timelapse frame comes from, and `artlapse overlay` draws the tracked canvas outline on a range of the stream.

```bash
uv run artlapse sidebyside work/mystream
uv run artlapse overlay work/mystream 1:20:00 1:25:00
uv run artlapse check work/mystream              # suspect jumps; stretches that disagree with the final canvas
zcat work/mystream/decisions.jsonl.gz | jq 'select(.time == "1:23:45.0")'   # how register decided there
```

`check` lists the jumps in the track that the frames do not support (the view moved, but the frame before explains the frame after better unmoved), and matches frames every 30 s directly against the final canvas: a slip shows as a stretch of samples that disagree with it, beginning or ending where it happened. Register's suspect jumps are also in its log and in `runs.jsonl`.

The decision log has, for every sample register did not find unmoved, the views it found (inliers, zoom, rotation, flip, where each puts the screen's centre on the canvas) and the pixel votes between them. Register seeds its random choices per sample, so a run over part of the range reproduces that part.

## Judge between two tracks

A canvas is a fair reference only for the track that built it, and only near the time it shows: content the artist moved later disagrees with it however right the track is, and a canvas built from another track (an older run's, say) carries that track's errors.

To tell two tracks apart, warp an early frame (from before they part) and a later one into the canvas with each track and overlay them: under the right track, strokes that did not change coincide. `scripts/residual.py` measures that for a series of later frames. `check --track` and `--canvas` compare other tracks or canvases framed the same way.

## Known problems

The problems seen on the test streams and not fixed yet are in the README, under [Scope and known gaps](../README.md#scope-and-known-gaps).
