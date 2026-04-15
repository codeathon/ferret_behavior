# Optical Flow

Calculate optical flow for pupil videos.

> Archived module under `src/old/`. Kept for reference, not part of the maintained pipeline.

## Run

Change the video file path in `__main__.py` and run it. If you have not processed that video before,
it prompts you to draw a crop around the eye. Press Enter to accept the crop, then it runs the flow
calculation. You can toggle whether the flow is displayed and whether it is recorded (saved to video).
There is also a `full_plot` option. When selected, it shows a full matplotlib plot with the flow video
and histograms, but it runs very slowly.