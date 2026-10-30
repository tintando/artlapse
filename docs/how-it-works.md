# How it works

Artlapse makes a timelapse in three stages. Each reads the output of the one before, so a change only reruns the stages it reaches.

1. **register** finds the canvas in each sampled frame: where it is on screen, how large, how rotated, whether flipped.
2. **composite** paints each frame's view of the canvas into one full-resolution canvas, keeping what stays and dropping what passes.
3. **render** picks frames of that canvas filling in, paced by painting activity, and encodes the timelapse.

The setup tells them where to look: the **Workspace** (where the art program shows the canvas), the **Exclude** boxes (overlays that stay over it) and the **Target** (the part of the canvas to rebuild, drawn on a frame where it is upright and unflipped, the reference frame).

![A sketched stream screen with the Workspace, two Exclude boxes and the Target drawn on it](figures/target.png)

## register

Every sampled frame (2 a second by default) is matched against a map of SIFT features in canvas coordinates, built up from keyframes. The result is a screen-to-canvas similarity transform per frame: position, scale and rotation. Flips are handled by mirroring the frame first and matching both ways.

A frame that matches nothing, like a BRB scene or another document, is marked lost and skipped. One that has not changed from the previous frame is marked unmoved and keeps its view without matching.

The map keeps only the latest appearance of each canvas area, so it follows the painting as it changes. When the matches support more than one view (the artist transformed a layer, or the canvas zoomed under a chat box or a toolbar that stays put), the previous frame decides: the view that turns it into the current frame over most of the picture wins. That keeps the track on the document when a layer moves, and on the canvas when something fixed sits on top of it.

Processing runs from the reference frame backward to `start`, then forward to `end`, passing over the `cuts`, the stretches left out in between. Each sample's random choices are seeded by its index, so a run over part of the range reproduces that part.

## composite

Each registered frame is warped into the canvas. A pixel changes only after showing the same new value for 8 seconds (`stable_seconds`), so strokes get through while cursors, alerts and raid animations don't.

Zoomed-in views sharpen the pixels they agree with, which is how the canvas ends up sharper than the stream ever showed the whole of it. A view further out than the one a pixel came from shows that pixel softer, so it may make only clear changes there, or faint ones where the strokes' ink changed (a sketch layer faded before inking, `ink_threshold`).

While the view holds still, only the tiles that changed since they were last processed are updated. The canvas is written to `canvas.mp4` every time it changes enough (`min_changed_px`), with the sample it came from.

## render

Render picks frames from `canvas.mp4` paced by painting activity, mixed with stream time (`activity_weight`), so idle stretches shrink and busy ones get room, then holds on the result (`hold_end`). Cuts take no time in it.

## In the browser

The website runs the same three stages in the browser: SIFT, matching and the canvas are WebGPU compute shaders, and the browser's WebCodecs decode the recording and encode the timelapse. Where WebGPU gives no device, the same passes run on the processor instead, each shared out by rows across one worker per core. Its register agrees with this one on 98% of the samples of the vod test stream. [web/README.md](../web/README.md) has how it is built and how it compares.
