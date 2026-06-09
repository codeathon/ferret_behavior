#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Dom/JsonValue.h"
#include "ArenaWallMaterialBinding.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "ArenaStimulusPlayerComponent.generated.h"

class UPrimitiveComponent;

/**
 * Offline wall-stimulus playback from stimulus_timeline.json + wall_textures/.
 * Offline arena playback: wall textures plus optional merged skull/gaze pose.
 */
UCLASS(ClassGroup = (FerretArena), meta = (BlueprintSpawnableComponent))
class FERRETARENARENDER_API UArenaStimulusPlayerComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UArenaStimulusPlayerComponent();

	virtual void BeginPlay() override;
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

	UFUNCTION(BlueprintCallable, Category = "FerretArena")
	bool LoadManifest(const FString& ManifestAbsolutePath);

	UFUNCTION(BlueprintCallable, Category = "FerretArena")
	bool SetFrameIndex(int32 FrameIndex);

	UFUNCTION(BlueprintCallable, Category = "FerretArena")
	void Play();

	UFUNCTION(BlueprintCallable, Category = "FerretArena")
	void Pause();

	UFUNCTION(BlueprintCallable, Category = "FerretArena")
	void Stop();

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FerretArena")
	FString ManifestPath;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FerretArena")
	TArray<FArenaWallMaterialBinding> WallBindings;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FerretArena", meta = (ClampMin = "1.0"))
	float PlaybackFps = 90.0f;

	// Advance frames automatically after manifest load (calls Play()).
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FerretArena")
	bool bAutoPlayOnLoad = true;

	// Offline pose from merged timeline (skull + gaze blocks per frame).
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FerretArena|Pose")
	TObjectPtr<AActor> SkullActor = nullptr;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FerretArena|Pose")
	bool bApplyPoseFromTimeline = true;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FerretArena|Pose")
	bool bDrawGazeDebug = true;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FerretArena|Pose", meta = (ClampMin = "1.0"))
	float GazeDebugLengthCm = 30.0f;

	UPROPERTY(BlueprintReadOnly, Category = "FerretArena")
	int32 CurrentFrameIndex = 0;

	UPROPERTY(BlueprintReadOnly, Category = "FerretArena")
	int32 TotalFrameCount = 0;

	UPROPERTY(BlueprintReadOnly, Category = "FerretArena")
	bool bIsPlaying = false;

private:
	bool LoadTimelineJson(const FString& TimelineAbsolutePath);
	bool ApplyFrameTextures(int32 FrameIndex);
	bool ApplyFramePose(int32 FrameIndex);
	FString ResolveTexturePath(const FString& RelativePath) const;
	UMaterialInstanceDynamic* ResolveWallMaterialDynamic(int32 BindingIndex);

private:
	FString TextureRoot;
	TArray<TSharedPtr<FJsonValue>> TimelineFrames;
	TArray<TObjectPtr<UMaterialInstanceDynamic>> CachedWallMaterialDynamics;
	float PlaybackAccumulator = 0.0f;
	bool bPendingManifestLoad = false;

	UPrimitiveComponent* ResolveWallMesh(const FArenaWallMaterialBinding& Binding) const;
	void TryAutoLoadManifest();
};
