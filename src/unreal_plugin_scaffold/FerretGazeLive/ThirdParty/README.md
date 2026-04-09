# ThirdParty: ZeroMQ, cppzmq, msgpack-c++

`FerretGazeLive.Build.cs` adds **`Plugins/FerretGazeLive/ThirdParty/include`** to the compile include path and links **libzmq** from **`ThirdParty/lib/<Platform>/`**.

- **msgpack-c++** and **cppzmq** are **header-only**; copy their headers under `ThirdParty/include` (see below).
- **libzmq** must be supplied as a **library** (and on Windows often a **DLL**) per platform folder.

Expected layout after you populate it:

```text
ThirdParty/
  include/
    msgpack.hpp          (and msgpack/… tree from msgpack-c++)
    zmq.h                (from libzmq)
    zmq.hpp              (from cppzmq; C++ wrapper)
  lib/
    Mac/
      libzmq.a           **universal (arm64+x86_64)** static lib preferred for UE Editor
      libsodium.a        optional; link if present (universal recommended)
      arm64/             source slices used to build the universal .a (optional in repo)
      x64/
    Win64/
      zmq.lib            (import or static lib; name must match Build.cs or rename)
      zmq.dll            (required at runtime for dynamic builds — see below)
      libsodium.lib      (optional, for static zmq+sodium)
    Linux/
      libzmq.a           (preferred) or libzmq.so
      libsodium.a        (optional)
```

`FerretGazeLive.Build.cs` enables **`FERRET_GAZE_WITH_ZMQ_MSGPACK=1`** on **Win64, Mac, and Linux** only. Other platforms compile with the define set to **0** (transport stubs). On supported platforms, if **no** libzmq file is found, UBT throws a **BuildException** with this path hint.

## macOS (Homebrew example)

Install:

```bash
brew install zeromq cppzmq msgpack-cxx
```

Then copy into the plugin (adjust `FERRET_ROOT` to your `Plugins/FerretGazeLive` path):

```bash
FERRET_ROOT="/path/to/YourProject/Plugins/FerretGazeLive"
mkdir -p "$FERRET_ROOT/ThirdParty/include" "$FERRET_ROOT/ThirdParty/lib/Mac"

cp -R "$(brew --prefix msgpack-cxx)/include/msgpack" "$FERRET_ROOT/ThirdParty/include/"
cp "$(brew --prefix msgpack-cxx)/include/msgpack.hpp" "$FERRET_ROOT/ThirdParty/include/"

cp "$(brew --prefix zeromq)/include/zmq.h" "$FERRET_ROOT/ThirdParty/include/"
cp "$(brew --prefix cppzmq)/include/zmq.hpp" "$FERRET_ROOT/ThirdParty/include/"

cp "$(brew --prefix zeromq)/lib/libzmq.a" "$FERRET_ROOT/ThirdParty/lib/Mac/"
# If your build only ships dylib:
# cp "$(brew --prefix zeromq)/lib/libzmq.dylib" "$FERRET_ROOT/ThirdParty/lib/Mac/"
```

If linking fails with missing **sodium** symbols, install `libsodium` and copy **`libsodium.a`** into `ThirdParty/lib/Mac/` (Build.cs links it when that file exists).

### macOS: msgpack headers without `msgpack-cxx` (Homebrew pulls Boost)

If `brew install msgpack-cxx` is slow or blocked, install only **`zeromq`** and **`cppzmq`**, then unpack **msgpack-c** C++ headers from the upstream tag (header-only; no Boost):

```bash
FERRET_ROOT="/path/to/YourProject/Plugins/FerretGazeLive"
curl -fsSL -o /tmp/msgpack-c-cpp.tar.gz "https://github.com/msgpack/msgpack-c/archive/refs/tags/cpp-6.1.0.tar.gz"
tar -xzf /tmp/msgpack-c-cpp.tar.gz -C /tmp
cp -f /tmp/msgpack-c-cpp-6.1.0/include/msgpack.hpp "$FERRET_ROOT/ThirdParty/include/"
cp -R /tmp/msgpack-c-cpp-6.1.0/include/msgpack "$FERRET_ROOT/ThirdParty/include/"
```

This repo’s **`ThirdParty/include/`** for Mac is populated using that tarball plus Homebrew **`zmq.h`** / **`zmq.hpp`**.

**Universal `libzmq.a` / `libsodium.a` (arm64 + x86_64):** Unreal Editor on Mac links **both** architectures. Homebrew on Apple Silicon usually ships **arm64-only** `.a` files, which breaks the Intel slice. This repo keeps:

- **`lib/Mac/arm64/`** — Homebrew **`libzmq.a`** + **`libsodium.a`**
- **`lib/Mac/x64/`** — static libs built for **x86_64** (e.g. official **zeromq-4.3.5** and **libsodium-1.0.20** source with `CC="clang -arch x86_64"` and `-mmacosx-version-min=14.0`)
- **`lib/Mac/libzmq.a`** / **`libsodium.a`** — merged with **`lipo -create arm64/libzmq.a x64/libzmq.a -output lib/Mac/libzmq.a`** (same for sodium)

`FerretGazeLive.Build.cs` defines **`MSGPACK_NO_BOOST=1`** so vendored msgpack uses **`msgpack/predef/other/endian.h`** instead of Boost (no Boost in `ThirdParty`).

## Windows (vcpkg example)

Install a **triplet** that matches your UE MSVC toolset (e.g. `x64-windows` or static `x64-windows-static`).

```bat
vcpkg install zeromq:x64-windows cppzmq:x64-windows msgpack-cxx:x64-windows
```

Copy from the vcpkg `installed/<triplet>/` tree:

- Headers: `include/msgpack.hpp`, `include/msgpack/`, `include/zmq.h`, `include/zmq.hpp` → `ThirdParty/include/`
- `lib/zmq.lib` → `ThirdParty/lib/Win64/zmq.lib` (rename if needed to match)
- For **dynamic** vcpkg builds: copy **`zmq.dll`** to **`ThirdParty/lib/Win64/zmq.dll`** and also place a copy next to the binary that loads the plugin (often **`YourProject/Binaries/Win64/`** for the Editor, or follow Epic’s guidance for packaged builds) so the loader can find it.

Import libraries from third parties must match the **same CRT and MSVC version** UE uses for your engine build.

## Linux

Use your distro or self-built **libzmq** / **msgpack-c++** / **cppzmq** packages; mirror the same **`include/`** layout and place **`libzmq.a`** or **`libzmq.so`** under **`ThirdParty/lib/Linux/`**.

## Version notes

- **cppzmq** should be new enough that **`zmq::recv`** returns **`std::optional`**-style results (as used in `FerretGazeSubscriberWorker.cpp`).
- If headers or ABIs drift, rebuild **libzmq** and recopy headers/libs together.
