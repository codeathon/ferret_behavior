#include "ArenaTextureLoader.h"

#include "Engine/Texture2D.h"
#include "ImageUtils.h"

namespace FerretArena
{
	bool LoadImageTextureFromFile(const FString& AbsolutePath, UTexture2D*& OutTexture)
	{
		OutTexture = nullptr;
		// UE 5.7 path: builds valid transient platform data (manual mip lock often fails).
		OutTexture = FImageUtils::ImportFileAsTexture2D(AbsolutePath);
		if (OutTexture != nullptr)
		{
			// Runtime-loaded frames must be immediately sampleable by wall materials.
			OutTexture->NeverStream = true;
			OutTexture->SRGB = true;
			OutTexture->UpdateResource();
		}
		return OutTexture != nullptr;
	}
}
