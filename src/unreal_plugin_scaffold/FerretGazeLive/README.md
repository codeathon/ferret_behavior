# FerretGazeLive Unreal Plugin Scaffold

This is a concrete Unreal C++ plugin layout for live gaze rendering.

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

## Integrate into Unreal project

1. Copy `FerretGazeLive/` into `<YourProject>/Plugins/`.
2. Regenerate project files.
3. Build project.
4. Add `LiveGazeReceiverComponent` to an actor.
5. Add `GazeRenderApplierComponent` to the same actor (or another scene actor).
6. Set `ReceiverComponent` on the applier, or let it auto-discover on owner.
7. Optionally set `TargetHeadActor`, `MaxPacketAgeMs`, and smoothing settings.

## Transport TODOs

- Wire external include/library paths and set `FERRET_GAZE_WITH_ZMQ_MSGPACK=1`.
- Add structured logs/UE stats for connection churn and drop spikes.

## Step 8 validation loop

1. Add both components to one actor:
	- `LiveGazeReceiverComponent`
	- `GazeRenderApplierComponent`
2. Configure Unreal defaults:
	- `MaxPacketAgeMs = 80`
	- `SmoothingAlpha = 0.35`
	- `RollingWindowSize = 240`
3. Run smoke publisher from repo root:
	- `uv run python src/ferret_gaze/realtime/smoke_publish_live.py --seconds 180 --hz 60`
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
- Run acceptance checklist:
	- `uv run python src/ferret_gaze/realtime/step9_acceptance_check.py`
	- optional config file: `--config /path/to/realtime_config.json`
- Receiver now logs transport events with counters:
	- connection state transitions
	- reconnect attempts
	- receive errors
	- drop spikes (every +120 dropped packets)
