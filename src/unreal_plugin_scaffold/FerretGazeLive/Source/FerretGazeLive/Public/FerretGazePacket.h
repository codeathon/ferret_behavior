#pragma once

#include "CoreMinimal.h"
#include "FerretGazePacket.generated.h"

/**
 * Runtime gaze payload matching Python realtime packet schema.
 */
USTRUCT(BlueprintType)
struct FFerretGazePacket
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, Category = "FerretGaze")
	int64 Sequence = 0;

	UPROPERTY(BlueprintReadOnly, Category = "FerretGaze")
	int64 CaptureUtcNs = 0;

	UPROPERTY(BlueprintReadOnly, Category = "FerretGaze")
	int64 ProcessStartNs = 0;

	UPROPERTY(BlueprintReadOnly, Category = "FerretGaze")
	int64 PublishUtcNs = 0;

	UPROPERTY(BlueprintReadOnly, Category = "FerretGaze")
	FVector SkullPositionCm = FVector::ZeroVector;

	// Unreal uses X,Y,Z,W ordering for quaternions.
	UPROPERTY(BlueprintReadOnly, Category = "FerretGaze")
	FQuat SkullOrientation = FQuat::Identity;

	UPROPERTY(BlueprintReadOnly, Category = "FerretGaze")
	FVector LeftEyeOriginCm = FVector::ZeroVector;

	UPROPERTY(BlueprintReadOnly, Category = "FerretGaze")
	FVector LeftGazeDirection = FVector::ForwardVector;

	UPROPERTY(BlueprintReadOnly, Category = "FerretGaze")
	FVector RightEyeOriginCm = FVector::ZeroVector;

	UPROPERTY(BlueprintReadOnly, Category = "FerretGaze")
	FVector RightGazeDirection = FVector::ForwardVector;

	UPROPERTY(BlueprintReadOnly, Category = "FerretGaze")
	float Confidence = 0.0f;
};
