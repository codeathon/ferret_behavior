# FerretGazeLive Unreal Plugin Scaffold

This is a concrete Unreal C++ plugin layout for live gaze rendering.

**Target engine:** Unreal Engine **5.7** (see `EngineVersion` in `FerretGazeLive.uplugin`). The editor **Plugins** list shows this build as **Ferret Gaze Live (UE 5.7)** via `FriendlyName`; keep that string in sync when you retarget a new engine minor.  
UE 4.27 and other 4.x builds often fail on current Apple Clang; use UE5 for macOS development.

## Class layout

- `FFerretGazeSubscriberWorker` (`FRunnable`)
	- Runs on a background thread.
	- Receives transport payloads (`ZMQ SUB` integration point).
	- Parses payload into `FFerretGazePacket`.
	- Pushes packet to `TQueue` for game-thread consumption.

- `ULiveGazeReceiverComponent` (`UActorComponent`)
	- Runs on game thread (`TickComponent`).
	- Drains queue with a bounded budget (`MaxPacketsPerTick`).
	- Applies `latest-frame-wins` policy.
	- Broadcasts `OnGazePacket` to Blueprint/C++ listeners.
	- Broadcasts `OnGazeHealth` every tick for stream telemetry.

- `UGazeRenderApplierComponent` (`UActorComponent`)
	- Subscribes to `OnGazePacket`.
	- Applies skull pose to a target actor (or owner).
	- Draws left/right gaze debug rays.
	- Drops stale packets (`MaxPacketAgeMs`) and supports optional smoothing.

- `FFerretGazeLiveModule` (`IModuleInterface`)
	- Standard plugin startup/shutdown hook.

## Files

- `FerretGazeLive.uplugin`
- `Source/FerretGazeLive/FerretGazeLive.Build.cs`
- `Source/FerretGazeLive/Public/FerretGazePacket.h`
- `Source/FerretGazeLive/Public/FerretGazeSubscriberWorker.h`
- `Source/FerretGazeLive/Public/LiveGazeReceiverComponent.h`
- `Source/FerretGazeLive/Public/GazeRenderApplierComponent.h`
- `Source/FerretGazeLive/Private/FerretGazeSubscriberWorker.cpp`
- `Source/FerretGazeLive/Private/LiveGazeReceiverComponent.cpp`
- `Source/FerretGazeLive/Private/GazeRenderApplierComponent.cpp`
- `Source/FerretGazeLive/Private/FerretGazeLiveModule.cpp`
- `ThirdParty/README.md` (ZMQ / msgpack headers and libzmq layout)

## ZMQ and msgpack (live transport)

`FerretGazeLive.Build.cs` sets **`FERRET_GAZE_WITH_ZMQ_MSGPACK=1`** on **Win64, Mac, and Linux**. Populate **`Plugins/FerretGazeLive/ThirdParty/`** as described in **`ThirdParty/README.md`** (`include/` for headers, `lib/<Platform>/` for **libzmq**). Other platforms build with the define set to **0** (no link). Windows dynamic **zmq.dll** must be discoverable at runtime (see ThirdParty README).

## Integrate into Unreal project (UE 5.7)

1. Create or open a **UE 5.7** **C++** project.
2. Copy this `FerretGazeLive/` folder into `<YourProject>/Plugins/FerretGazeLive` (final path: `Plugins/FerretGazeLive/FerretGazeLive.uplugin`).
3. For **Win64 / Mac / Linux Editor** builds, populate **`Plugins/FerretGazeLive/ThirdParty/`** (`include/` + `lib/<Platform>/`) per **`ThirdParty/README.md`** so **libzmq** and headers are present before compiling.
4. Right-click the `.uproject` → **Generate Visual Studio / Xcode project files** (or use the UE context menu).
5. Build the **Editor** target once (from IDE or `RunUAT.sh BuildEditor` as per Epic docs).
6. Launch the editor; enable the plugin under **Edit → Plugins** if it is not already enabled.
7. Add `LiveGazeReceiverComponent` to an actor.
8. Add `GazeRenderApplierComponent` to the same actor (or another scene actor).
9. Set `ReceiverComponent` on the applier, or let it auto-discover on owner.
10. Optionally set `TargetHeadActor`, `MaxPacketAgeMs`, and smoothing settings.

### If the project fails to compile

- **Layout:** The `.uplugin` file must live at `YourProject/Plugins/FerretGazeLive/FerretGazeLive.uplugin`, with `Source/FerretGazeLive/` beside it. Putting only `Source` under Plugins or pointing an “extra plugins” directory at the wrong folder breaks UBT discovery.
- **C++ project:** Blueprint-only projects cannot compile this plugin until you add at least one C++ class (File → New C++ Class) so the game module and targets exist.
- **Regenerate:** After copying the plugin, always regenerate IDE project files from the `.uproject`, then do a clean build of the **Editor** target.
- **Engine version:** `EngineVersion` in the `.uplugin` must be compatible with your editor. Mismatches can disable the plugin or fail code generation.
- **UBT enum:** If C# fails with `Unreal5_7` missing from `EngineIncludeOrderVersion`, your install may be older than 5.7; set `IncludeOrderVersion = EngineIncludeOrderVersion.Latest` in `FerretGazeLive.Build.cs` or match the enum your `Engine/Source/Programs/UnrealBuildTool` defines.
- **Share the first error:** The first compiler or UHT line (not the tail of the log) pinpoints the issue quickly.
- **ZMQ ThirdParty:** On Win64/Mac/Linux, UBT fails fast if `ThirdParty/lib/<Platform>/` does not contain **libzmq** (see **`ThirdParty/README.md`**). Missing headers show as `msgpack.hpp` / `zmq.hpp` not found under `ThirdParty/include/`.

## Transport TODOs

- Structured logs / UE stats for connection churn and drop spikes (beyond current `LogTemp` counters).

## Step 8 validation loop

1. Add both components to one actor:
	- `LiveGazeReceiverComponent`
	- `GazeRenderApplierComponent`
2. Configure Unreal defaults:
	- `MaxPacketAgeMs = 80`
	- `SmoothingAlpha = 0.35`
	- `RollingWindowSize = 240`
3. Run smoke publisher from repo root (default **circle** gaze; use `--motion sine` for legacy wobble):
	- `uv run python src/ferret_gaze/realtime/smoke_publish_live.py --seconds 180 --hz 60`
	- Slower circle: `--circle-period 8` (seconds per full rotation)
4. Watch Unreal log stats:
	- `age_ms[p50=..., p95=...]`
	- `stale_drop=...`
	- `conf_drop=...`
5. Tune:
	- If stale drops rise: increase `MaxPacketAgeMs` slightly.
	- If motion feels laggy: increase `SmoothingAlpha`.
	- If motion is jittery: decrease `SmoothingAlpha`.

## Step 9 hardening

- Use centralized runtime config:
	- `src/ferret_gaze/realtime/runtime_config.py`
- Use checked-in config presets:
	- scaffold / synthetic: `configs/realtime.runtime.json`
	- live hardware baseline: `configs/realtime.runtime.live.json` (replace model + calibration paths before running)
- Run acceptance checklist:
	- `uv run python src/ferret_gaze/realtime/step9_acceptance_check.py`
	- optional config file: `--config /path/to/realtime_config.json`
- Acceptance now supports resilience counters in addition to latency:
	- `acceptance_max_dropped_count`
	- `acceptance_max_queue_overflow_count`
	- `acceptance_max_stage_error_count`
	- `acceptance_max_publish_error_count`
- Receiver now logs transport events with counters:
	- connection state transitions
	- reconnect attempts
	- receive errors
	- drop spikes (every +120 dropped packets)
