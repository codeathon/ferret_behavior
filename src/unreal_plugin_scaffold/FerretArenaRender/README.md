# FerretArenaRender Unreal Plugin

Offline arena wall-stimulus playback for Unreal Engine **5.7**. Consumes outputs from the Python `arena_render` pipeline:

- `unreal_arena_manifest.json`
- `stimulus_timeline.json`
- `wall_textures/wall_<id>/NNNNNN.jpg`

No ZMQ, no ferret POV — wall textures only.

## Python prep (lab machine)

After `extract-textures` completes:

```bash
export SESSION_ROOT=/home/scholl-lab/ferret_recordings/session_2026-03-19_psychopy_trial_1_ferret411

uv run python -m src.arena_render.export_unreal_bundle \
  --session "$SESSION_ROOT" \
  --geometry "$SESSION_ROOT/arena_geometry.json"
```

Writes:

`$SESSION_ROOT/full_recording/arena_render/unreal_arena_manifest.json`

## Install plugin

1. Create or open a **UE 5.7 C++** project.
2. Copy this folder to `<YourProject>/Plugins/FerretArenaRender`.
3. Regenerate project files from the `.uproject`.
4. Build the **Editor** target.
5. Enable **Ferret Arena Render (UE 5.7)** under **Edit → Plugins**.

## Scene setup (first milestone)

### 1. Arena box

Manifest `arena` and `wall_screens` use **centimeters** (Python converts mm → cm).

Example floor/wall layout:

- Arena half-size: 50 cm × 50 cm × 50 cm (1 m outer box)
- Place four wall planes at `wall_screens[].screen_center_cm` with scale from `screen_half_size_cm`

### 2. Wall materials

Create a material `M_WallStimulus` with a **Texture Sample Parameter** named `WallTexture` plugged into Base Color (or Emissive for screens).

For each wall mesh:

1. Add `UMaterialInstanceDynamic` from `M_WallStimulus`
2. Bind in `ArenaStimulusPlayerComponent` → `WallBindings`:
   - `WallId`: `north` / `south` / `east` / `west`
   - `WallMaterial`: the dynamic instance
   - `TextureParameterName`: `WallTexture`

### 3. Stimulus player actor

1. Create an empty actor `ArenaStimulusController`.
2. Add **Arena Stimulus Player Component**.
3. Set `ManifestPath` to the absolute path of `unreal_arena_manifest.json` on the lab machine.
4. Wire all four `WallBindings`.
5. PIE → call **Play** (Blueprint or Details panel).

### 4. Observer camera

Use `cameras[]` from the manifest to place debug cameras matching overhead calibration (optional).

## Component API

| Function | Purpose |
|----------|---------|
| `LoadManifest(path)` | Load manifest + timeline JSON |
| `SetFrameIndex(n)` | Jump to frame and update wall textures |
| `Play` / `Pause` / `Stop` | Timeline playback at `PlaybackFps` |

## Movie Render Queue (offline MP4)

1. Set `SetFrameIndex(0)` and verify walls update.
2. Add **Movie Render Queue** pipeline.
3. Sequence: for each frame `0..N-1`, call `SetFrameIndex` via Blueprint or Editor Utility Widget.
4. Render from a fixed observer camera.

A Blueprint sequencer automation utility can be added in a follow-up milestone.

## Files

- `ArenaStimulusPlayerComponent` — timeline playback
- `ArenaTextureLoader` — jpg/png → `UTexture2D`
- `ArenaWallMaterialBinding` — wall id → material instance

## Related repo modules

- Python: `src/arena_render/`
- Live gaze (separate): `src/unreal_plugin_scaffold/FerretGazeLive/`
