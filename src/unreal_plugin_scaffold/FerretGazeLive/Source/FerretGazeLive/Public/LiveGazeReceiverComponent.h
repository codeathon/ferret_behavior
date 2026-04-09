#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "FerretGazePacket.h"
#include "LiveGazeReceiverComponent.generated.h"

class FFerretGazeSubscriberWorker;
class FRunnableThread;

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FFerretGazePacketEvent, const FFerretGazePacket&, Packet);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_FiveParams(
	FFerretGazeHealthEvent,
	bool, bConnected,
	int64, LastSequence,
	int64, LastCaptureUtcNs,
	int64, TransportDroppedPackets,
	int64, PolicyDroppedPackets
);

/**
 * Game-thread bridge for realtime gaze packets.
 *
 * Thread model:
 * - Worker thread receives packets and writes to queue (MPSC).
 * - Tick consumes queue and applies latest-frame-wins policy.
 */
UCLASS(ClassGroup = (FerretGaze), meta = (BlueprintSpawnableComponent))
class FERRETGAZELIVE_API ULiveGazeReceiverComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	ULiveGazeReceiverComponent();

	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

	// Allows test feeds and scripted injections without network transport.
	UFUNCTION(BlueprintCallable, Category = "FerretGaze")
	void EnqueuePacketForTesting(const FFerretGazePacket& Packet);

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FerretGaze|Transport")
	FString Endpoint = TEXT("tcp://127.0.0.1:5556");

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FerretGaze|Transport")
	FString Topic = TEXT("gaze.live");

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FerretGaze|Policy")
	int32 MaxPacketsPerTick = 128;

	UPROPERTY(BlueprintAssignable, Category = "FerretGaze")
	FFerretGazePacketEvent OnGazePacket;

	// Emitted each tick so blueprints can show stream health/quality.
	UPROPERTY(BlueprintAssignable, Category = "FerretGaze")
	FFerretGazeHealthEvent OnGazeHealth;

	UPROPERTY(BlueprintReadOnly, Category = "FerretGaze")
	FFerretGazePacket LastPacket;

	UPROPERTY(BlueprintReadOnly, Category = "FerretGaze|Health")
	bool bTransportConnected = false;

	UPROPERTY(BlueprintReadOnly, Category = "FerretGaze|Health")
	int64 LastSequence = -1;

	UPROPERTY(BlueprintReadOnly, Category = "FerretGaze|Health")
	int64 TransportDroppedPackets = 0;

	UPROPERTY(BlueprintReadOnly, Category = "FerretGaze|Health")
	int64 PolicyDroppedPackets = 0;

private:
	void StartWorker();
	void StopWorker();

private:
	TQueue<TSharedPtr<FFerretGazePacket>, EQueueMode::Mpsc> PacketQueue;
	TUniquePtr<FFerretGazeSubscriberWorker> Worker;
	FRunnableThread* WorkerThread = nullptr;
};
