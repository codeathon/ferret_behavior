#include "ArenaStimulusPlayerComponent.h"

#include "ArenaTextureLoader.h"
#include "Components/PrimitiveComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Dom/JsonObject.h"
#include "DrawDebugHelpers.h"
#include "Engine/Texture2D.h"
#include "Engine/World.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

namespace FerretArenaPoseJson
{
	static bool ReadVec3(const TSharedPtr<FJsonObject>& Object, const FString& Field, FVector& OutVector)
	{
		const TArray<TSharedPtr<FJsonValue>>* Values = nullptr;
		if (!Object->TryGetArrayField(Field, Values) || Values->Num() != 3)
		{
			return false;
		}
		OutVector.X = static_cast<float>((*Values)[0]->AsNumber());
		OutVector.Y = static_cast<float>((*Values)[1]->AsNumber());
		OutVector.Z = static_cast<float>((*Values)[2]->AsNumber());
		return true;
	}

	static bool ReadQuatWxyz(const TSharedPtr<FJsonObject>& Object, const FString& Field, FQuat& OutQuat)
	{
		const TArray<TSharedPtr<FJsonValue>>* Values = nullptr;
		if (!Object->TryGetArrayField(Field, Values) || Values->Num() != 4)
		{
			return false;
		}
		OutQuat.W = static_cast<float>((*Values)[0]->AsNumber());
		OutQuat.X = static_cast<float>((*Values)[1]->AsNumber());
		OutQuat.Y = static_cast<float>((*Values)[2]->AsNumber());
		OutQuat.Z = static_cast<float>((*Values)[3]->AsNumber());
		return true;
	}
}

UArenaStimulusPlayerComponent::UArenaStimulusPlayerComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
	PrimaryComponentTick.TickGroup = TG_PrePhysics;
}

void UArenaStimulusPlayerComponent::BeginPlay()
{
	Super::BeginPlay();
	UE_LOG(
		LogTemp,
		Warning,
		TEXT("Arena: BeginPlay on %s — bindings=%d manifest='%s'"),
		*GetNameSafe(GetOwner()),
		WallBindings.Num(),
		*ManifestPath);

	// Defer one frame so Blueprint BeginPlay can finish filling WallBindings/WallActor.
	bPendingManifestLoad = !ManifestPath.IsEmpty();
}

void UArenaStimulusPlayerComponent::TickComponent(
	float DeltaTime,
	ELevelTick TickType,
	FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	if (bPendingManifestLoad)
	{
		bPendingManifestLoad = false;
		TryAutoLoadManifest();
	}

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

void UArenaStimulusPlayerComponent::TryAutoLoadManifest()
{
	if (ManifestPath.IsEmpty())
	{
		UE_LOG(LogTemp, Error, TEXT("Arena: ManifestPath is empty on %s"), *GetNameSafe(GetOwner()));
		return;
	}
	if (!LoadManifest(ManifestPath))
	{
		return;
	}
	if (bAutoPlayOnLoad)
	{
		Play();
		UE_LOG(
			LogTemp,
			Warning,
			TEXT("Arena: auto-play ON — %d frames at %.1f fps (set bAutoPlayOnLoad=false to scrub manually)"),
			TotalFrameCount,
			PlaybackFps);
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("Arena: loaded frame 0 — call Play() to advance"));
	}
}

UPrimitiveComponent* UArenaStimulusPlayerComponent::ResolveWallMesh(const FArenaWallMaterialBinding& Binding) const
{
	if (Binding.WallMesh != nullptr)
	{
		return Binding.WallMesh;
	}
	if (Binding.WallActor != nullptr)
	{
		return Binding.WallActor->FindComponentByClass<UStaticMeshComponent>();
	}
	return nullptr;
}

bool UArenaStimulusPlayerComponent::LoadManifest(const FString& ManifestAbsolutePath)
{
	CachedWallMaterialDynamics.Reset();

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

	UE_LOG(
		LogTemp,
		Warning,
		TEXT("Arena: manifest ok — frames=%d texture_root='%s'"),
		TotalFrameCount,
		*TextureRoot);

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
		UE_LOG(LogTemp, Error, TEXT("Arena: invalid timeline JSON %s"), *TimelineAbsolutePath);
		return false;
	}

	const TArray<TSharedPtr<FJsonValue>>* Frames = nullptr;
	if (!Root->TryGetArrayField(TEXT("frames"), Frames))
	{
		UE_LOG(LogTemp, Error, TEXT("Arena: timeline missing frames[] %s"), *TimelineAbsolutePath);
		return false;
	}

	TimelineFrames = *Frames;
	TotalFrameCount = TimelineFrames.Num();
	UE_LOG(LogTemp, Warning, TEXT("Arena: timeline loaded %d frames"), TotalFrameCount);
	return SetFrameIndex(0);
}

bool UArenaStimulusPlayerComponent::SetFrameIndex(int32 FrameIndex)
{
	if (FrameIndex < 0 || FrameIndex >= TimelineFrames.Num())
	{
		UE_LOG(LogTemp, Error, TEXT("Arena: frame %d out of range (0..%d)"), FrameIndex, TimelineFrames.Num() - 1);
		return false;
	}
	CurrentFrameIndex = FrameIndex;
	const bool bTexturesApplied = ApplyFrameTextures(FrameIndex);
	const bool bPoseApplied = ApplyFramePose(FrameIndex);
	return bTexturesApplied && bPoseApplied;
}

bool UArenaStimulusPlayerComponent::ApplyFramePose(int32 FrameIndex)
{
	if (!bApplyPoseFromTimeline)
	{
		return true;
	}

	const TSharedPtr<FJsonObject> FrameObject = TimelineFrames[FrameIndex]->AsObject();
	if (!FrameObject.IsValid())
	{
		return false;
	}

	const TSharedPtr<FJsonObject>* SkullObject = nullptr;
	if (!FrameObject->TryGetObjectField(TEXT("skull"), SkullObject) || !SkullObject->IsValid())
	{
		return true;
	}

	if (SkullActor != nullptr)
	{
		FVector SkullPositionCm;
		FQuat SkullRotation;
		if (FerretArenaPoseJson::ReadVec3(*SkullObject, TEXT("position_cm"), SkullPositionCm)
			&& FerretArenaPoseJson::ReadQuatWxyz(*SkullObject, TEXT("quaternion_wxyz"), SkullRotation))
		{
			SkullActor->SetActorLocation(SkullPositionCm);
			SkullActor->SetActorRotation(SkullRotation);
		}
	}

	if (!bDrawGazeDebug)
	{
		return true;
	}

	const TSharedPtr<FJsonObject>* GazeObject = nullptr;
	if (!FrameObject->TryGetObjectField(TEXT("gaze"), GazeObject) || !GazeObject->IsValid())
	{
		return true;
	}

	UWorld* World = GetWorld();
	if (World == nullptr)
	{
		return true;
	}

	const float RayLength = GazeDebugLengthCm;
	const auto DrawEyeRay = [&](const FString& EyeKey, const FColor& Color)
	{
		const TSharedPtr<FJsonObject>* EyeObject = nullptr;
		if (!(*GazeObject)->TryGetObjectField(EyeKey, EyeObject) || !EyeObject->IsValid())
		{
			return;
		}
		FVector OriginCm;
		FVector Direction;
		if (!FerretArenaPoseJson::ReadVec3(*EyeObject, TEXT("origin_cm"), OriginCm)
			|| !FerretArenaPoseJson::ReadVec3(*EyeObject, TEXT("direction"), Direction))
		{
			return;
		}
		const FVector EndCm = OriginCm + Direction.GetSafeNormal() * RayLength;
		DrawDebugLine(World, OriginCm, EndCm, Color, false, 0.0f, 0, 1.5f);
	};

	DrawEyeRay(TEXT("left"), FColor::Green);
	DrawEyeRay(TEXT("right"), FColor::Cyan);
	return true;
}

UMaterialInstanceDynamic* UArenaStimulusPlayerComponent::ResolveWallMaterialDynamic(int32 BindingIndex)
{
	if (!WallBindings.IsValidIndex(BindingIndex))
	{
		return nullptr;
	}

	const FArenaWallMaterialBinding& Binding = WallBindings[BindingIndex];
	if (Binding.WallMaterial == nullptr)
	{
		return nullptr;
	}

	if (CachedWallMaterialDynamics.Num() != WallBindings.Num())
	{
		CachedWallMaterialDynamics.SetNum(WallBindings.Num());
	}

	if (UMaterialInstanceDynamic* CachedMID = CachedWallMaterialDynamics[BindingIndex].Get())
	{
		return CachedMID;
	}

	if (UMaterialInstanceDynamic* BindingMID = Cast<UMaterialInstanceDynamic>(Binding.WallMaterial))
	{
		CachedWallMaterialDynamics[BindingIndex] = BindingMID;
		return BindingMID;
	}

	UPrimitiveComponent* WallMesh = ResolveWallMesh(Binding);
	if (WallMesh != nullptr)
	{
		UMaterialInstanceDynamic* MeshMID = WallMesh->CreateDynamicMaterialInstance(0, Binding.WallMaterial);
		if (MeshMID != nullptr)
		{
			WallMesh->SetMaterial(0, MeshMID);
			WallMesh->MarkRenderStateDirty();
			CachedWallMaterialDynamics[BindingIndex] = MeshMID;
			UE_LOG(
				LogTemp,
				Warning,
				TEXT("Arena: binding %d (%s) MID on %s.%s"),
				BindingIndex,
				*Binding.WallId,
				*GetNameSafe(WallMesh->GetOwner()),
				*GetNameSafe(WallMesh));
			return MeshMID;
		}
	}

	UE_LOG(
		LogTemp,
		Error,
		TEXT("Arena: binding %d (%s) has no WallActor/WallMesh — assign a wall actor in the level"),
		BindingIndex,
		*Binding.WallId);
	return nullptr;
}

bool UArenaStimulusPlayerComponent::ApplyFrameTextures(int32 FrameIndex)
{
	const TSharedPtr<FJsonObject> FrameObject = TimelineFrames[FrameIndex]->AsObject();
	if (!FrameObject.IsValid())
	{
		UE_LOG(LogTemp, Error, TEXT("Arena: frame %d has no JSON object"), FrameIndex);
		return false;
	}

	const TSharedPtr<FJsonObject>* WallTextures = nullptr;
	if (!FrameObject->TryGetObjectField(TEXT("wall_textures"), WallTextures))
	{
		UE_LOG(LogTemp, Error, TEXT("Arena: frame %d missing wall_textures"), FrameIndex);
		return false;
	}

	if (WallBindings.Num() == 0)
	{
		UE_LOG(LogTemp, Error, TEXT("Arena: WallBindings is empty — assign 4 bindings on the level actor"));
		return false;
	}

	int32 AppliedCount = 0;
	for (int32 BindingIndex = 0; BindingIndex < WallBindings.Num(); ++BindingIndex)
	{
		const FArenaWallMaterialBinding& Binding = WallBindings[BindingIndex];
		if (Binding.WallMaterial == nullptr || Binding.WallId.IsEmpty())
		{
			UE_LOG(LogTemp, Warning, TEXT("Arena: binding %d missing WallMaterial or WallId"), BindingIndex);
			continue;
		}
		FString RelativePath;
		if (!(*WallTextures)->TryGetStringField(Binding.WallId, RelativePath))
		{
			UE_LOG(LogTemp, Warning, TEXT("Arena: frame %d has no texture for wall '%s'"), FrameIndex, *Binding.WallId);
			continue;
		}

		UTexture2D* LoadedTexture = nullptr;
		const FString AbsolutePath = ResolveTexturePath(RelativePath);
		if (!FerretArena::LoadImageTextureFromFile(AbsolutePath, LoadedTexture))
		{
			UE_LOG(LogTemp, Warning, TEXT("Arena: texture load failed %s"), *AbsolutePath);
			continue;
		}

		UMaterialInstanceDynamic* WallMID = ResolveWallMaterialDynamic(BindingIndex);
		if (WallMID == nullptr)
		{
			continue;
		}
		WallMID->SetTextureParameterValue(Binding.TextureParameterName, LoadedTexture);

		UTexture* VerifyTexture = nullptr;
		if (!WallMID->GetTextureParameterValue(Binding.TextureParameterName, VerifyTexture))
		{
			UE_LOG(
				LogTemp,
				Error,
				TEXT("Arena: material '%s' has no texture parameter '%s' — use Texture Sample Parameter 2D in M_WallStimulus"),
				*GetNameSafe(Binding.WallMaterial),
				*Binding.TextureParameterName.ToString());
			continue;
		}

		if (UPrimitiveComponent* WallMesh = ResolveWallMesh(Binding))
		{
			WallMesh->MarkRenderStateDirty();
		}

		++AppliedCount;
	}

	if (AppliedCount == 0)
	{
		UE_LOG(
			LogTemp,
			Error,
			TEXT("Arena: frame %d applied 0 textures — check WallActor + M_WallStimulus param '%s'"),
			FrameIndex,
			WallBindings.Num() > 0 ? *WallBindings[0].TextureParameterName.ToString() : TEXT("WallTexture"));
		return false;
	}

	// Log first frame and every 90 frames to avoid spamming Output Log during playback.
	if (FrameIndex == 0 || FrameIndex % 90 == 0)
	{
		UE_LOG(
			LogTemp,
			Warning,
			TEXT("Arena: frame %d applied %d/%d wall textures"),
			FrameIndex,
			AppliedCount,
			WallBindings.Num());
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
