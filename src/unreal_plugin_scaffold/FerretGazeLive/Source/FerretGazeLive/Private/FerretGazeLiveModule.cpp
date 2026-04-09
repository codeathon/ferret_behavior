#include "Modules/ModuleManager.h"

/**
 * Runtime plugin module for live gaze streaming.
 */
class FFerretGazeLiveModule : public IModuleInterface
{
public:
	virtual void StartupModule() override
	{
		// Module startup hook reserved for transport init if needed.
	}

	virtual void ShutdownModule() override
	{
		// Module shutdown hook reserved for transport cleanup if needed.
	}
};

IMPLEMENT_MODULE(FFerretGazeLiveModule, FerretGazeLive)
