#pragma once

#include "CoreMinimal.h"
#include "ArenaWallMaterialBinding.generated.h"

class AActor;
class UMaterialInterface;
class UPrimitiveComponent;

/**
 * Maps a timeline wall id (north/south/east/west) to a wall mesh material.
 * WallMaterial accepts parent materials or material instances from Content Browser.
 */
USTRUCT(BlueprintType)
struct FERRETARENARENDER_API FArenaWallMaterialBinding
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Arena")
	FString WallId;

	// Assign M_WallStimulus or MI_Wall_* here; runtime textures use a dynamic instance.
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Arena")
	TObjectPtr<UMaterialInterface> WallMaterial = nullptr;

	// Drag the wall actor here (easiest), or assign WallMesh directly.
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Arena")
	TObjectPtr<AActor> WallActor = nullptr;

	// Optional mesh override; defaults to the actor's Static Mesh component.
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Arena")
	TObjectPtr<UPrimitiveComponent> WallMesh = nullptr;

	// Texture parameter on WallMaterial (default matches starter material).
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Arena")
	FName TextureParameterName = TEXT("WallTexture");
};
