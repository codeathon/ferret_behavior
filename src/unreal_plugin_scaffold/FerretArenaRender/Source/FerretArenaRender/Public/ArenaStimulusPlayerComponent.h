#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Dom/JsonValue.h"
#include "ArenaWallMaterialBinding.h"
#include "ArenaStimulusPlayerComponent.generated.h"

/**
 * Offline wall-stimulus playback from stimulus_timeline.json + wall_textures/.
 * Arena-only scope: no ferret pose or ZMQ.
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

	UPROPERTY(BlueprintReadOnly, Category = "FerretArena")
	int32 CurrentFrameIndex = 0;

	UPROPERTY(BlueprintReadOnly, Category = "FerretArena")
	int32 TotalFrameCount = 0;

	UPROPERTY(BlueprintReadOnly, Category = "FerretArena")
	bool bIsPlaying = false;

private:
	bool LoadTimelineJson(const FString& TimelineAbsolutePath);
	bool ApplyFrameTextures(int32 FrameIndex);
	FString ResolveTexturePath(const FString& RelativePath) const;

private:
	FString TextureRoot;
	TArray<TSharedPtr<FJsonValue>> TimelineFrames;
	float PlaybackAccumulator = 0.0f;
};
