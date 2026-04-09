#include "GazeRenderApplierComponent.h"

#include "DrawDebugHelpers.h"
#include "GameFramework/Actor.h"
#include "LiveGazeReceiverComponent.h"

#include <chrono>

UGazeRenderApplierComponent::UGazeRenderApplierComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void UGazeRenderApplierComponent::BeginPlay()
{
	Super::BeginPlay();
	BindReceiver();
}

void UGazeRenderApplierComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	UnbindReceiver();
	Super::EndPlay(EndPlayReason);
}

void UGazeRenderApplierComponent::HandleGazePacket(const FFerretGazePacket& Packet)
{
	if (Packet.Confidence < MinConfidenceToRender)
	{
		++LowConfidenceDropCount;
		return;
	}

	const int64 PacketTimeNs = Packet.CaptureUtcNs;
	const double NowNs = GetUnixTimeNanoseconds();
	if (PacketTimeNs > 0)
	{
		LastAppliedPacketAgeMs = static_cast<float>((NowNs - static_cast<double>(PacketTimeNs)) / 1'000'000.0);
		if (LastAppliedPacketAgeMs > MaxPacketAgeMs)
		{
			++StalePacketDropCount;
			return;
		}
		UpdateRollingAgeMetrics(LastAppliedPacketAgeMs);
	}

	if (bApplyHeadPose)
	{
		ApplyHeadPose(Packet);
	}
	if (bDrawGazeDebugRays)
	{
		DrawGazeDebugRays(Packet);
	}

	++AppliedPacketCount;
	MaybeLogStats();
}

void UGazeRenderApplierComponent::RebindReceiver()
{
	UnbindReceiver();
	BindReceiver();
}

void UGazeRenderApplierComponent::BindReceiver()
{
	if (ReceiverComponent == nullptr)
	{
		ReceiverComponent = GetOwner() != nullptr ? GetOwner()->FindComponentByClass<ULiveGazeReceiverComponent>() : nullptr;
	}
	if (ReceiverComponent != nullptr)
	{
		ReceiverComponent->OnGazePacket.AddDynamic(this, &UGazeRenderApplierComponent::HandleGazePacket);
	}
}

void UGazeRenderApplierComponent::UnbindReceiver()
{
	if (ReceiverComponent != nullptr)
	{
		ReceiverComponent->OnGazePacket.RemoveDynamic(this, &UGazeRenderApplierComponent::HandleGazePacket);
	}
}

void UGazeRenderApplierComponent::ApplyHeadPose(const FFerretGazePacket& Packet)
{
	AActor* Target = TargetHeadActor != nullptr ? TargetHeadActor.Get() : GetOwner();
	if (Target == nullptr)
	{
		return;
	}

	FVector PositionCm = Packet.SkullPositionCm;
	FQuat Orientation = Packet.SkullOrientation;

	if (bEnableSmoothing)
	{
		if (!bHasSmoothedPose)
		{
			SmoothedPositionCm = PositionCm;
			SmoothedOrientation = Orientation;
			bHasSmoothedPose = true;
		}
		else
		{
			SmoothedPositionCm = FMath::Lerp(SmoothedPositionCm, PositionCm, SmoothingAlpha);
			SmoothedOrientation = FQuat::Slerp(SmoothedOrientation, Orientation, SmoothingAlpha).GetNormalized();
		}
		PositionCm = SmoothedPositionCm;
		Orientation = SmoothedOrientation;
	}

	Target->SetActorLocationAndRotation(PositionCm, Orientation, false, nullptr, ETeleportType::None);
}

void UGazeRenderApplierComponent::DrawGazeDebugRays(const FFerretGazePacket& Packet) const
{
	UWorld* World = GetWorld();
	if (World == nullptr)
	{
		return;
	}

	const FVector LeftDir = Packet.LeftGazeDirection.GetSafeNormal();
	const FVector RightDir = Packet.RightGazeDirection.GetSafeNormal();
	const FVector LeftEnd = Packet.LeftEyeOriginCm + (LeftDir * GazeRayLengthCm);
	const FVector RightEnd = Packet.RightEyeOriginCm + (RightDir * GazeRayLengthCm);

	// Very short persistence prevents long trails while still visualizing continuity.
	DrawDebugLine(World, Packet.LeftEyeOriginCm, LeftEnd, FColor::Blue, false, 0.03f, 0, 1.8f);
	DrawDebugLine(World, Packet.RightEyeOriginCm, RightEnd, FColor::Red, false, 0.03f, 0, 1.8f);
}

double UGazeRenderApplierComponent::GetUnixTimeNanoseconds() const
{
	const auto Now = std::chrono::system_clock::now();
	const auto SinceEpoch = Now.time_since_epoch();
	return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(SinceEpoch).count());
}

void UGazeRenderApplierComponent::UpdateRollingAgeMetrics(float PacketAgeMs)
{
	if (RollingWindowSize < 10)
	{
		RollingWindowSize = 10;
	}
	RecentPacketAgesMs.Add(PacketAgeMs);
	if (RecentPacketAgesMs.Num() > RollingWindowSize)
	{
		const int32 ExcessCount = RecentPacketAgesMs.Num() - RollingWindowSize;
		RecentPacketAgesMs.RemoveAt(0, ExcessCount, false);
	}
	if (RecentPacketAgesMs.Num() == 0)
	{
		return;
	}

	TArray<float> SortedAges = RecentPacketAgesMs;
	SortedAges.Sort();
	const int32 LastIndex = SortedAges.Num() - 1;
	const int32 P50Index = FMath::Clamp(FMath::FloorToInt(static_cast<float>(LastIndex) * 0.50f), 0, LastIndex);
	const int32 P95Index = FMath::Clamp(FMath::FloorToInt(static_cast<float>(LastIndex) * 0.95f), 0, LastIndex);
	RollingPacketAgeP50Ms = SortedAges[P50Index];
	RollingPacketAgeP95Ms = SortedAges[P95Index];
}

void UGazeRenderApplierComponent::MaybeLogStats() const
{
	if (LogEveryNAppliedPackets <= 0 || AppliedPacketCount <= 0)
	{
		return;
	}
	if ((AppliedPacketCount % LogEveryNAppliedPackets) != 0)
	{
		return;
	}

	UE_LOG(
		LogTemp,
		Log,
		TEXT("GazeRender stats | applied=%lld stale_drop=%lld conf_drop=%lld age_ms[p50=%.2f p95=%.2f last=%.2f]"),
		AppliedPacketCount,
		StalePacketDropCount,
		LowConfidenceDropCount,
		RollingPacketAgeP50Ms,
		RollingPacketAgeP95Ms,
		LastAppliedPacketAgeMs
	);
}
