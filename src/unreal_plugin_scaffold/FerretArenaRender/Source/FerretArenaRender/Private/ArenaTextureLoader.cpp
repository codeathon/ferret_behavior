#include "ArenaTextureLoader.h"

#include "IImageWrapper.h"
#include "IImageWrapperModule.h"
#include "Modules/ModuleManager.h"
#include "Engine/Texture2D.h"
#include "Misc/FileHelper.h"
#include "Rendering/Texture2DResource.h"
#include "TextureResource.h"

namespace FerretArena
{
	static EImageFormat ImageFormatFromExtension(const FString& Extension)
	{
		if (Extension.Equals(TEXT("png"), ESearchCase::IgnoreCase))
		{
			return EImageFormat::PNG;
		}
		return EImageFormat::JPEG;
	}

	bool LoadImageTextureFromFile(const FString& AbsolutePath, UTexture2D*& OutTexture)
	{
		TArray<uint8> RawFileData;
		if (!FFileHelper::LoadFileToArray(RawFileData, *AbsolutePath))
		{
			return false;
		}

		IImageWrapperModule& ImageWrapperModule =
			FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));
		const FString Extension = FPaths::GetExtension(AbsolutePath, true);
		const TSharedPtr<IImageWrapper> ImageWrapper =
			ImageWrapperModule.CreateImageWrapper(ImageFormatFromExtension(Extension));
		if (!ImageWrapper.IsValid() || !ImageWrapper->SetCompressed(RawFileData.GetData(), RawFileData.Num()))
		{
			return false;
		}

		TArray<uint8> UncompressedBGRA;
		if (!ImageWrapper->GetRaw(ERGBFormat::BGRA, 8, UncompressedBGRA))
		{
			return false;
		}

		const int32 Width = ImageWrapper->GetWidth();
		const int32 Height = ImageWrapper->GetHeight();
		if (Width <= 0 || Height <= 0)
		{
			return false;
		}

		OutTexture = UTexture2D::CreateTransient(Width, Height, PF_B8G8R8A8);
		if (OutTexture == nullptr)
		{
			return false;
		}

		void* TextureData = OutTexture->GetPlatformData()->Mips[0].BulkData.Lock(LOCK_READ_WRITE);
		FMemory::Memcpy(TextureData, UncompressedBGRA.GetData(), UncompressedBGRA.Num());
		OutTexture->GetPlatformData()->Mips[0].BulkData.Unlock();
		OutTexture->UpdateResource();
		return true;
	}
}
