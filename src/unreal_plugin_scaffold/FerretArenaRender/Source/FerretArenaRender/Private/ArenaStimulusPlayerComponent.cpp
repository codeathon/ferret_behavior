#include "ArenaStimulusPlayerComponent.h"

#include "ArenaTextureLoader.h"
#include "Dom/JsonObject.h"
#include "Engine/Texture2D.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

UArenaStimulusPlayerComponent::UArenaStimulusPlayerComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
	PrimaryComponentTick.TickGroup = TG_PrePhysics;
}

void UArenaStimulusPlayerComponent::BeginPlay()
{
	Super::BeginPlay();
	if (!ManifestPath.IsEmpty())
	{
		LoadManifest(ManifestPath);
	}
}

void UArenaStimulusPlayerComponent::TickComponent(
	float DeltaTime,
	ELevelTick TickType,
	FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
	if (!bIsPlaying || TotalFrameCount <= 0 || PlaybackFps <= 0.0f)
	{
		return;
	}

	PlaybackAccumulator += DeltaTime;
	const float FrameDuration = 1.0f / PlaybackFps;
	while (PlaybackAccumulator >= FrameDuration)
	{
		PlaybackAccumulator -= FrameDuration;
		const int32 NextFrame = CurrentFrameIndex + 1;
		if (NextFrame >= TotalFrameCount)
		{
			Pause();
			break;
		}
		SetFrameIndex(NextFrame);
	}
}

bool UArenaStimulusPlayerComponent::LoadManifest(const FString& ManifestAbsolutePath)
{
	FString JsonText;
	if (!FFileHelper::LoadFileToString(JsonText, *ManifestAbsolutePath))
	{
		UE_LOG(LogTemp, Error, TEXT("Arena: failed to read manifest %s"), *ManifestAbsolutePath);
		return false;
	}

	TSharedPtr<FJsonObject> Root;
	const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonText);
	if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
	{
		UE_LOG(LogTemp, Error, TEXT("Arena: invalid manifest JSON"));
		return false;
	}

	TextureRoot = Root->GetStringField(TEXT("texture_root"));
	TotalFrameCount = static_cast<int32>(Root->GetNumberField(TEXT("frame_count")));
	PlaybackFps = static_cast<float>(Root->GetNumberField(TEXT("playback_fps")));
	const FString TimelinePath = Root->GetStringField(TEXT("timeline_json"));
	CurrentFrameIndex = 0;
	return LoadTimelineJson(TimelinePath);
}

bool UArenaStimulusPlayerComponent::LoadTimelineJson(const FString& TimelineAbsolutePath)
{
	FString JsonText;
	if (!FFileHelper::LoadFileToString(JsonText, *TimelineAbsolutePath))
	{
		UE_LOG(LogTemp, Error, TEXT("Arena: failed to read timeline %s"), *TimelineAbsolutePath);
		return false;
	}

	TSharedPtr<FJsonObject> Root;
	const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonText);
	if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
	{
		return false;
	}

	const TArray<TSharedPtr<FJsonValue>>* Frames = nullptr;
	if (!Root->TryGetArrayField(TEXT("frames"), Frames))
	{
		return false;
	}

	TimelineFrames = *Frames;
	TotalFrameCount = TimelineFrames.Num();
	return SetFrameIndex(0);
}

bool UArenaStimulusPlayerComponent::SetFrameIndex(int32 FrameIndex)
{
	if (FrameIndex < 0 || FrameIndex >= TimelineFrames.Num())
	{
		return false;
	}
	CurrentFrameIndex = FrameIndex;
	return ApplyFrameTextures(FrameIndex);
}

bool UArenaStimulusPlayerComponent::ApplyFrameTextures(int32 FrameIndex)
{
	const TSharedPtr<FJsonObject> FrameObject = TimelineFrames[FrameIndex]->AsObject();
	if (!FrameObject.IsValid())
	{
		return false;
	}

	const TSharedPtr<FJsonObject>* WallTextures = nullptr;
	if (!FrameObject->TryGetObjectField(TEXT("wall_textures"), WallTextures))
	{
		return false;
	}

	for (FArenaWallMaterialBinding& Binding : WallBindings)
	{
		if (Binding.WallMaterial == nullptr || Binding.WallId.IsEmpty())
		{
			continue;
		}
		FString RelativePath;
		if (!(*WallTextures)->TryGetStringField(Binding.WallId, RelativePath))
		{
			continue;
		}

		UTexture2D* LoadedTexture = nullptr;
		const FString AbsolutePath = ResolveTexturePath(RelativePath);
		if (!FerretArena::LoadImageTextureFromFile(AbsolutePath, LoadedTexture))
		{
			UE_LOG(LogTemp, Warning, TEXT("Arena: texture load failed %s"), *AbsolutePath);
			continue;
		}
		Binding.WallMaterial->SetTextureParameterValue(Binding.TextureParameterName, LoadedTexture);
	}
	return true;
}

FString UArenaStimulusPlayerComponent::ResolveTexturePath(const FString& RelativePath) const
{
	if (FPaths::FileExists(RelativePath))
	{
		return RelativePath;
	}
	return FPaths::Combine(TextureRoot, RelativePath);
}

void UArenaStimulusPlayerComponent::Play()
{
	bIsPlaying = TotalFrameCount > 0;
	PlaybackAccumulator = 0.0f;
}

void UArenaStimulusPlayerComponent::Pause()
{
	bIsPlaying = false;
}

void UArenaStimulusPlayerComponent::Stop()
{
	bIsPlaying = false;
	PlaybackAccumulator = 0.0f;
	SetFrameIndex(0);
}
