#include "LiveGazeReceiverComponent.h"

#include "FerretGazeSubscriberWorker.h"
#include "HAL/RunnableThread.h"

ULiveGazeReceiverComponent::ULiveGazeReceiverComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
}

void ULiveGazeReceiverComponent::BeginPlay()
{
	Super::BeginPlay();
	StartWorker();
}

void ULiveGazeReceiverComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	StopWorker();
	Super::EndPlay(EndPlayReason);
}

void ULiveGazeReceiverComponent::TickComponent(
	float DeltaTime,
	ELevelTick TickType,
	FActorComponentTickFunction* ThisTickFunction
)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	// latest-frame-wins policy: drain queue, keep newest packet.
	int32 DrainedCount = 0;
	TSharedPtr<FFerretGazePacket> CurrentPacket;
	TSharedPtr<FFerretGazePacket> LatestPacket;
	while (DrainedCount < MaxPacketsPerTick && PacketQueue.Dequeue(CurrentPacket))
	{
		LatestPacket = CurrentPacket;
		++DrainedCount;
	}

	if (LatestPacket.IsValid())
	{
		LastPacket = *LatestPacket;
		OnGazePacket.Broadcast(*LatestPacket);
	}
}

void ULiveGazeReceiverComponent::EnqueuePacketForTesting(const FFerretGazePacket& Packet)
{
	PacketQueue.Enqueue(MakeShared<FFerretGazePacket>(Packet));
}

void ULiveGazeReceiverComponent::StartWorker()
{
	if (WorkerThread != nullptr)
	{
		return;
	}

	Worker = MakeUnique<FFerretGazeSubscriberWorker>(Endpoint, Topic, &PacketQueue);
	WorkerThread = FRunnableThread::Create(Worker.Get(), TEXT("FerretGazeSubscriberWorker"));
}

void ULiveGazeReceiverComponent::StopWorker()
{
	if (Worker.IsValid())
	{
		Worker->Stop();
	}

	if (WorkerThread != nullptr)
	{
		WorkerThread->WaitForCompletion();
		delete WorkerThread;
		WorkerThread = nullptr;
	}

	Worker.Reset();
}
