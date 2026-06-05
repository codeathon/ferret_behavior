#pragma once

#include "CoreMinimal.h"
#include "ArenaWallMaterialBinding.generated.h"

class UMaterialInstanceDynamic;

/**
 * Maps a timeline wall id (north/south/east/west) to a dynamic material on a mesh.
 */
USTRUCT(BlueprintType)
struct FERRETARENARENDER_API FArenaWallMaterialBinding
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Arena")
	FString WallId;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Arena")
	TObjectPtr<UMaterialInstanceDynamic> WallMaterial = nullptr;

	// Texture parameter on WallMaterial (default matches starter material).
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Arena")
	FName TextureParameterName = TEXT("WallTexture");
};
