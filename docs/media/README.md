# Media

The video and images in the README, what belongs in each, and how each was made. Every file here is used by the README, and every picture the README shows is here or in `docs/figures/`.

The stream and the illustration are SlimyMana's ([x.com/SlimyMana](https://x.com/SlimyMana)): her stream in `work/slimy_2_whole`, from the run of 2026-10-29 (engine 3327dae141, config 9663a3b9be; register on the CPU, composite and render on CUDA).

## stream-and-painting.mp4

The README's opening: the chaos of the stream against the steadiness of the rebuilt painting. The stream on the left, at the moment each timelapse frame comes from, so it jumps through six hours of zooms, rotations and scenes; the timelapse on the right, filling in steadily. Labels above each half, in Carlito (metric-compatible with Lato, the setup page's font) on the dark theme's background.

1440×448, 32 s (the 63 s `sidebyside` video at double speed), H.264, no sound, 9.5 MB: under the 10 MB GitHub takes for a video in a README.

```bash
uv run artlapse sidebyside work/slimy_2_whole          # side_by_side.mp4, 3858x1080, 63 s
F=/usr/share/fonts/google-carlito-fonts/Carlito-Bold.ttf
ffmpeg -n -i work/slimy_2_whole/side_by_side.mp4 -an -vf "setpts=PTS/2,fps=30,scale=1440:-2:flags=lanczos,pad=iw:ih+44:0:44:color=0x15151c,\
drawtext=fontfile=$F:text='The stream, 6 hours':x=(w/2-text_w)/2:y=11:fontsize=26:fontcolor=0xe8e7ee,\
drawtext=fontfile=$F:text='The painting, rebuilt by Artlapse':x=w/2+(w/2-text_w)/2:y=11:fontsize=26:fontcolor=0xe8e7ee" \
  -c:v libx264 -preset slow -crf 27 -pix_fmt yuv420p -movflags +faststart docs/media/stream-and-painting.mp4
```

GitHub plays a video in a README only when it was uploaded through its web editor. Until then the README shows the still below, linking to this file. To play it in place: edit README.md on github.com, drag this file into the editor, and replace the `<a ...><img ...></a>` line with the `https://github.com/user-attachments/assets/...` link GitHub inserts, on a line of its own.

## stream-and-painting.jpg

The still the README shows in the video's place: frame 11.6 s of the video, stream time 3:03:22. The canvas is tilted about 40° and zoomed in on the stream, with the avatar and the chat over it; on the right the painting is upright and whole.

```bash
ffmpeg -n -ss 11.6 -i docs/media/stream-and-painting.mp4 -frames:v 1 -q:v 3 docs/media/stream-and-painting.jpg
```

## setup-page.png

The setup page, as an artist sees it once the boxes are drawn: the website version (web/, served by `npx vite`), light theme, 1440×860 at scale 2, no browser chrome. The setup imported is the one the run above used (`work/slimy_2_whole/history/3327dae141-9663a3b9be/config.yaml`), so the page opens on the target frame, 1:30:00, with the Workspace, three exclusions and the Target, and no problems listed. The tour is marked as seen, so it does not cover the page.

Taken with puppeteer-core in headless Chrome: import the setup, load the recording through the file chooser, wait for the frame, move the mouse to the corner, screenshot.

The stream's chat and the viewers' names in its bottom strip are in frame, as they were on the stream.
