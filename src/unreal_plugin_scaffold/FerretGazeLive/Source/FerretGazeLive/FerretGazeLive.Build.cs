using UnrealBuildTool;

public class FerretGazeLive : ModuleRules
{
	public FerretGazeLive(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"CoreUObject",
				"Engine",
				"Json",
				"JsonUtilities"
			}
		);

		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"Projects"
			}
		);

		// Default off: plugin still compiles without third-party deps.
		PublicDefinitions.Add("FERRET_GAZE_WITH_ZMQ_MSGPACK=0");
		// Set to 1 and add include/libs in your Unreal project to enable live transport.
		// PublicDefinitions.Add("FERRET_GAZE_WITH_ZMQ_MSGPACK=1");
	}
}
