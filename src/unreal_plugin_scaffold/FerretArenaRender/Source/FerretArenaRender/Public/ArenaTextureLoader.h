#pragma once

#include "CoreMinimal.h"

class UTexture2D;

namespace FerretArena
{
	/** Load a jpg/png from disk into a transient UTexture2D for wall materials. */
	bool LoadImageTextureFromFile(const FString& AbsolutePath, UTexture2D*& OutTexture);
}
