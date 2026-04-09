#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "FerretGazePacket.h"
#include "GazeRenderApplierComponent.generated.h"

class ULiveGazeReceiverComponent;

/**
 * Applies live gaze packets to scene objects on the game thread.
 */
UCLASS(ClassGroup = (FerretGaze), meta = (BlueprintSpawnableComponent))
class FERRETGAZELIVE_API UGazeRenderApplierComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UGazeRenderApplierComponent();

	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

	// Receives live packets from ULiveGazeReceiverComponent.
	UFUNCTION()
	void HandleGazePacket(const FFerretGazePacket& Packet);

	// Explicit rebinding helper for runtime wiring.
	UFUNCTION(BlueprintCallable, Category = "FerretGaze|Render")
	void RebindReceiver();

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FerretGaze|Render")
	TObjectPtr<ULiveGazeReceiverComponent> ReceiverComponent = nullptr;

	// If unset, owner actor is used.
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FerretGaze|Render")
	TObjectPtr<AActor> TargetHeadActor = nullptr;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FerretGaze|Render")
	bool bApplyHeadPose = true;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FerretGaze|Render")
	bool bDrawGazeDebugRays = true;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FerretGaze|Render", meta = (ClampMin = "1.0"))
	float GazeRayLengthCm = 100.0f;

	// Drops packets too old to render plausibly in realtime.
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FerretGaze|Render", meta = (ClampMin = "1.0"))
	float MaxPacketAgeMs = 80.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FerretGaze|Render")
	bool bEnableSmoothing = false;

	// 1.0 = no smoothing; lower values add more temporal smoothing.
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FerretGaze|Render", meta = (ClampMin = "0.01", ClampMax = "1.0"))
	float SmoothingAlpha = 0.35f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FerretGaze|Render", meta = (ClampMin = "0.0", ClampMax = "1.0"))
	float MinConfidenceToRender = 0.10f;

	UPROPERTY(BlueprintReadOnly, Category = "FerretGaze|Render|Stats")
	int64 AppliedPacketCount = 0;

	UPROPERTY(BlueprintReadOnly, Category = "FerretGaze|Render|Stats")
	int64 StalePacketDropCount = 0;

	UPROPERTY(BlueprintReadOnly, Category = "FerretGaze|Render|Stats")
	int64 LowConfidenceDropCount = 0;

	UPROPERTY(BlueprintReadOnly, Category = "FerretGaze|Render|Stats")
	float LastAppliedPacketAgeMs = 0.0f;

	// Rolling packet age metrics for live tuning.
	UPROPERTY(BlueprintReadOnly, Category = "FerretGaze|Render|Stats")
	float RollingPacketAgeP50Ms = 0.0f;

	UPROPERTY(BlueprintReadOnly, Category = "FerretGaze|Render|Stats")
	float RollingPacketAgeP95Ms = 0.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FerretGaze|Render|Stats", meta = (ClampMin = "10", ClampMax = "2000"))
	int32 RollingWindowSize = 240;

	// Set to 0 to disable periodic logging.
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FerretGaze|Render|Stats", meta = (ClampMin = "0"))
	int32 LogEveryNAppliedPackets = 120;

private:
	void BindReceiver();
	void UnbindReceiver();
	void ApplyHeadPose(const FFerretGazePacket& Packet);
	void DrawGazeDebugRays(const FFerretGazePacket& Packet) const;
	double GetUnixTimeNanoseconds() const;
	void UpdateRollingAgeMetrics(float PacketAgeMs);
	void MaybeLogStats() const;

private:
	bool bHasSmoothedPose = false;
	FVector SmoothedPositionCm = FVector::ZeroVector;
	FQuat SmoothedOrientation = FQuat::Identity;
	TArray<float> RecentPacketAgesMs;
};
