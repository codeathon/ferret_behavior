#pragma once

#include "CoreMinimal.h"
#include "HAL/Runnable.h"
#include "FerretGazePacket.h"

/**
 * Worker thread that receives gaze packets from transport.
 *
 * This scaffold leaves transport calls abstract so you can plug in ZMQ
 * without changing the game-thread consumer path.
 */
class FFerretGazeSubscriberWorker : public FRunnable
{
public:
	FFerretGazeSubscriberWorker(
		const FString& InEndpoint,
		const FString& InTopic,
		TQueue<TSharedPtr<FFerretGazePacket>, EQueueMode::Mpsc>* InQueue
	);

	virtual ~FFerretGazeSubscriberWorker() override;

	virtual uint32 Run() override;
	virtual void Stop() override;

	// Thread-safe metric snapshots for game-thread monitoring.
	uint64 GetDroppedPacketCount() const;
	int64 GetLastSequence() const;
	bool IsTransportConnected() const;

private:
	bool EnsureTransportReady();
	bool ReceivePayloadBytes(TArray<uint8>& OutPayloadBytes);
	bool ParsePayloadMsgpack(const TArray<uint8>& PayloadBytes, FFerretGazePacket& OutPacket) const;
	bool ParsePayloadJson(const FString& Payload, FFerretGazePacket& OutPacket) const;
	void RecordDrop();

private:
	class FFerretGazeTransportState;
	FString Endpoint;
	FString Topic;
	TQueue<TSharedPtr<FFerretGazePacket>, EQueueMode::Mpsc>* Queue = nullptr;
	FThreadSafeBool bStopRequested = false;
	TUniquePtr<FFerretGazeTransportState> TransportState;
	TAtomic<uint64> DroppedPackets = 0;
	TAtomic<int64> LastSequence = -1;
	TAtomic<bool> bIsConnected = false;
	double NextReconnectAtSeconds = 0.0;
};
