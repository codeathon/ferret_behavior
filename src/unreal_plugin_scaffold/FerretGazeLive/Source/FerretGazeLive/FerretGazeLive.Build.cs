using System.IO;
using UnrealBuildTool;

public class FerretGazeLive : ModuleRules
{
	public FerretGazeLive(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
		// UE 5.x build settings (required for modern toolchains and IWYU).
		DefaultBuildSettings = BuildSettingsVersion.V5;
		// Pin to 5.7 so behavior matches EngineVersion in FerretGazeLive.uplugin.
		IncludeOrderVersion = EngineIncludeOrderVersion.Unreal5_7;

		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"CoreUObject",
				"Engine",
				"Json"
			}
		);

		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"Projects"
			}
		);

		// Plugin root: .../Plugins/FerretGazeLive (contains ThirdParty/, FerretGazeLive.uplugin).
		string PluginDirectory = Path.GetFullPath(Path.Combine(ModuleDirectory, "..", ".."));
		string ThirdPartyRoot = Path.Combine(PluginDirectory, "ThirdParty");
		string IncludeRoot = Path.Combine(ThirdPartyRoot, "include");

		PublicIncludePaths.Add(IncludeRoot);

		// Desktop targets only: link libzmq and enable transport. Other targets compile stubs (no ZMQ link).
		bool bZmqPlatform =
			Target.Platform == UnrealTargetPlatform.Win64
			|| Target.Platform == UnrealTargetPlatform.Mac
			|| Target.Platform == UnrealTargetPlatform.Linux;

		if (!bZmqPlatform)
		{
			PublicDefinitions.Add("FERRET_GAZE_WITH_ZMQ_MSGPACK=0");
			return;
		}

		PublicDefinitions.Add("FERRET_GAZE_WITH_ZMQ_MSGPACK=1");
		// Vendored msgpack-c headers use embedded predef instead of Boost (ThirdParty/include/msgpack/predef/...).
		PublicDefinitions.Add("MSGPACK_NO_BOOST=1");
		// Subscriber uses try/catch around ZMQ/msgpack when transport is enabled.
		bEnableExceptions = true;

		bool bLinked = TryAddZeromqLibrary(ThirdPartyRoot);
		if (!bLinked)
		{
			throw new BuildException(
				"FerretGazeLive: FERRET_GAZE_WITH_ZMQ_MSGPACK is on for this platform but no libzmq was found. "
				+ "Populate ThirdParty per Plugins/FerretGazeLive/ThirdParty/README.md (expected under "
				+ Path.Combine(ThirdPartyRoot, "lib", "<Platform>/") + ")."
			);
		}

		if (Target.Platform == UnrealTargetPlatform.Win64)
		{
			// libzmq on Windows typically needs these when linking the import library.
			PublicSystemLibraries.AddRange(new string[] { "ws2_32", "iphlpapi", "winmm" });
		}
		else if (Target.Platform == UnrealTargetPlatform.Mac)
		{
			PublicFrameworks.Add("CoreFoundation");
		}
	}

	// Resolves libzmq from ThirdParty/lib/<Platform>/; optional libsodium in the same folder for static builds.
	private bool TryAddZeromqLibrary(string ThirdPartyRoot)
	{
		string LibDir = Path.Combine(ThirdPartyRoot, "lib");
		if (Target.Platform == UnrealTargetPlatform.Mac)
		{
			// Prefer universal (fat) static libs under lib/Mac/ for arm64+x64 Editor builds (see ThirdParty/README.md).
			string MacDir = Path.Combine(LibDir, "Mac");
			TryAddOptionalLib(Path.Combine(MacDir, "libsodium.a"));
			string A = Path.Combine(MacDir, "libzmq.a");
			if (File.Exists(A))
			{
				PublicAdditionalLibraries.Add(A);
				return true;
			}
			string Dy = Path.Combine(MacDir, "libzmq.dylib");
			if (File.Exists(Dy))
			{
				PublicAdditionalLibraries.Add(Dy);
				return true;
			}
		}
		else if (Target.Platform == UnrealTargetPlatform.Win64)
		{
			string WinDir = Path.Combine(LibDir, "Win64");
			TryAddOptionalLib(Path.Combine(WinDir, "libsodium.lib"));
			string Lib = Path.Combine(WinDir, "zmq.lib");
			if (File.Exists(Lib))
			{
				PublicAdditionalLibraries.Add(Lib);
				return true;
			}
		}
		else if (Target.Platform == UnrealTargetPlatform.Linux)
		{
			string LnxDir = Path.Combine(LibDir, "Linux");
			TryAddOptionalLib(Path.Combine(LnxDir, "libsodium.a"));
			string A = Path.Combine(LnxDir, "libzmq.a");
			if (File.Exists(A))
			{
				PublicAdditionalLibraries.Add(A);
				return true;
			}
			string So = Path.Combine(LnxDir, "libzmq.so");
			if (File.Exists(So))
			{
				PublicAdditionalLibraries.Add(So);
				return true;
			}
		}

		return false;
	}

	private void TryAddOptionalLib(string AbsolutePath)
	{
		if (File.Exists(AbsolutePath))
		{
			PublicAdditionalLibraries.Add(AbsolutePath);
		}
	}
}
