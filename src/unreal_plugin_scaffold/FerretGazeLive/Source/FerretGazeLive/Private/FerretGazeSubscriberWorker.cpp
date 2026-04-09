#include "FerretGazeSubscriberWorker.h"

#include "Dom/JsonObject.h"
#include "HAL/PlatformTime.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

#if FERRET_GAZE_WITH_ZMQ_MSGPACK
#include <msgpack.hpp>
#include <zmq.hpp>
#endif

#if FERRET_GAZE_WITH_ZMQ_MSGPACK
class FFerretGazeSubscriberWorker::FFerretGazeTransportState
{
public:
	FFerretGazeTransportState()
		: Context(1)
		, Socket(Context, zmq::socket_type::sub)
	{
	}

	zmq::context_t Context;
	zmq::socket_t Socket;
	bool bConnected = false;
};
#else
class FFerretGazeSubscriberWorker::FFerretGazeTransportState
{
public:
	bool bWarnedUnavailable = false;
};
#endif

FFerretGazeSubscriberWorker::FFerretGazeSubscriberWorker(
	const FString& InEndpoint,
	const FString& InTopic,
	TQueue<TSharedPtr<FFerretGazePacket>, EQueueMode::Mpsc>* InQueue
)
	: Endpoint(InEndpoint)
	, Topic(InTopic)
	, Queue(InQueue)
	, TransportState(MakeUnique<FFerretGazeTransportState>())
{
}

FFerretGazeSubscriberWorker::~FFerretGazeSubscriberWorker()
{
}

uint64 FFerretGazeSubscriberWorker::GetDroppedPacketCount() const
{
	return DroppedPackets.Load();
}

int64 FFerretGazeSubscriberWorker::GetLastSequence() const
{
	return LastSequence.Load();
}

bool FFerretGazeSubscriberWorker::IsTransportConnected() const
{
	return bIsConnected.Load();
}

uint32 FFerretGazeSubscriberWorker::Run()
{
	while (!bStopRequested)
	{
		TArray<uint8> PayloadBytes;
		if (!ReceivePayloadBytes(PayloadBytes))
		{
			// No data available yet; avoid busy-spin.
			FPlatformProcess::SleepNoStats(0.001f);
			continue;
		}

		FFerretGazePacket Packet;
		if (!ParsePayloadMsgpack(PayloadBytes, Packet))
		{
			RecordDrop();
			continue;
		}

		if (Queue != nullptr)
		{
			Queue->Enqueue(MakeShared<FFerretGazePacket>(Packet));
			LastSequence.Store(Packet.Sequence);
		}
	}

	return 0;
}

void FFerretGazeSubscriberWorker::Stop()
{
	bStopRequested = true;
}

bool FFerretGazeSubscriberWorker::EnsureTransportReady()
{
	if (TransportState == nullptr)
	{
		bIsConnected.Store(false);
		return false;
	}

#if FERRET_GAZE_WITH_ZMQ_MSGPACK
	if (TransportState->bConnected)
	{
		bIsConnected.Store(true);
		return true;
	}

	const double NowSeconds = FPlatformTime::Seconds();
	if (NowSeconds < NextReconnectAtSeconds)
	{
		return false;
	}

	try
	{
		ReconnectAttempts.FetchAdd(1);
		const FTCHARToUTF8 EndpointUtf8(*Endpoint);
		const FTCHARToUTF8 TopicUtf8(*Topic);
		TransportState->Socket.set(zmq::sockopt::subscribe, TopicUtf8.Get(), static_cast<size_t>(TopicUtf8.Length()));
		TransportState->Socket.set(zmq::sockopt::linger, 0);
		TransportState->Socket.connect(std::string(EndpointUtf8.Get(), EndpointUtf8.Length()));
		TransportState->bConnected = true;
		bIsConnected.Store(true);
		LogTransportEvent(TEXT("connected"));
		return true;
	}
	catch (...)
	{
		// Backoff reduces reconnect pressure if endpoint is unavailable.
		NextReconnectAtSeconds = NowSeconds + 0.5;
		bIsConnected.Store(false);
		LogTransportEvent(TEXT("connect_failed"));
		return false;
	}
#else
	if (!TransportState->bWarnedUnavailable)
	{
		TransportState->bWarnedUnavailable = true;
	}
	bIsConnected.Store(false);
	return false;
#endif
}

bool FFerretGazeSubscriberWorker::ReceivePayloadBytes(TArray<uint8>& OutPayloadBytes)
{
	OutPayloadBytes.Reset();
	if (!EnsureTransportReady())
	{
		return false;
	}

#if FERRET_GAZE_WITH_ZMQ_MSGPACK
	try
	{
		zmq::message_t TopicFrame;
		zmq::message_t PayloadFrame;
		const auto TopicOk = TransportState->Socket.recv(TopicFrame, zmq::recv_flags::dontwait);
		if (!TopicOk.has_value())
		{
			return false;
		}
		const auto PayloadOk = TransportState->Socket.recv(PayloadFrame, zmq::recv_flags::dontwait);
		if (!PayloadOk.has_value())
		{
			return false;
		}
		OutPayloadBytes.Append(static_cast<const uint8*>(PayloadFrame.data()), static_cast<int32>(PayloadFrame.size()));
		return true;
	}
	catch (...)
	{
		TransportState->bConnected = false;
		NextReconnectAtSeconds = FPlatformTime::Seconds() + 0.5;
		bIsConnected.Store(false);
		ReceiveErrors.FetchAdd(1);
		LogTransportEvent(TEXT("receive_error"));
		return false;
	}
#else
	bIsConnected.Store(false);
	return false;
#endif
}

bool FFerretGazeSubscriberWorker::ParsePayloadMsgpack(const TArray<uint8>& PayloadBytes, FFerretGazePacket& OutPacket) const
{
#if FERRET_GAZE_WITH_ZMQ_MSGPACK
	if (PayloadBytes.Num() == 0)
	{
		return false;
	}

	try
	{
		auto TryReadVec3 = [](const msgpack::object& Value, FVector& OutVector) -> bool
		{
			if (Value.type != msgpack::type::ARRAY || Value.via.array.size != 3)
			{
				return false;
			}
			OutVector = FVector(
				static_cast<float>(Value.via.array.ptr[0].as<double>()),
				static_cast<float>(Value.via.array.ptr[1].as<double>()),
				static_cast<float>(Value.via.array.ptr[2].as<double>())
			);
			return true;
		};

		const msgpack::object_handle Unpacked = msgpack::unpack(reinterpret_cast<const char*>(PayloadBytes.GetData()), static_cast<size_t>(PayloadBytes.Num()));
		const msgpack::object Root = Unpacked.get();
		if (Root.type != msgpack::type::MAP)
		{
			return false;
		}
		for (uint32 i = 0; i < Root.via.map.size; ++i)
		{
			const msgpack::object_kv& Pair = Root.via.map.ptr[i];
			if (Pair.key.type != msgpack::type::STR)
			{
				continue;
			}
			const std::string Key(Pair.key.via.str.ptr, Pair.key.via.str.size);
			if (Key == "seq") { OutPacket.Sequence = Pair.val.as<int64>(); continue; }
			if (Key == "capture_utc_ns") { OutPacket.CaptureUtcNs = Pair.val.as<int64>(); continue; }
			if (Key == "process_start_ns") { OutPacket.ProcessStartNs = Pair.val.as<int64>(); continue; }
			if (Key == "publish_utc_ns") { OutPacket.PublishUtcNs = Pair.val.as<int64>(); continue; }
			if (Key == "confidence") { OutPacket.Confidence = Pair.val.as<float>(); continue; }
			if (Key == "skull_position_xyz") { TryReadVec3(Pair.val, OutPacket.SkullPositionCm); continue; }
			if (Key == "left_eye_origin_xyz") { TryReadVec3(Pair.val, OutPacket.LeftEyeOriginCm); continue; }
			if (Key == "left_gaze_direction_xyz") { TryReadVec3(Pair.val, OutPacket.LeftGazeDirection); continue; }
			if (Key == "right_eye_origin_xyz") { TryReadVec3(Pair.val, OutPacket.RightEyeOriginCm); continue; }
			if (Key == "right_gaze_direction_xyz") { TryReadVec3(Pair.val, OutPacket.RightGazeDirection); continue; }
			if (Key == "skull_quaternion_wxyz" && Pair.val.type == msgpack::type::ARRAY && Pair.val.via.array.size == 4)
			{
				const float W = static_cast<float>(Pair.val.via.array.ptr[0].as<double>());
				const float X = static_cast<float>(Pair.val.via.array.ptr[1].as<double>());
				const float Y = static_cast<float>(Pair.val.via.array.ptr[2].as<double>());
				const float Z = static_cast<float>(Pair.val.via.array.ptr[3].as<double>());
				OutPacket.SkullOrientation = FQuat(X, Y, Z, W);
			}
		}
		return true;
	}
	catch (...)
	{
		return false;
	}
#else
	return false;
#endif
}

void FFerretGazeSubscriberWorker::RecordDrop()
{
	const uint64 Dropped = DroppedPackets.FetchAdd(1) + 1;
	const uint64 PreviousLogged = LastLoggedDroppedPackets.Load();
	if (Dropped >= PreviousLogged + 120)
	{
		LastLoggedDroppedPackets.Store(Dropped);
		LogTransportEvent(TEXT("drop_spike"));
	}
}

void FFerretGazeSubscriberWorker::LogTransportEvent(const TCHAR* Message) const
{
	UE_LOG(
		LogTemp,
		Log,
		TEXT("GazeSubscriber %s | connected=%d reconnect_attempts=%llu receive_errors=%llu dropped=%llu last_seq=%lld"),
		Message,
		IsTransportConnected() ? 1 : 0,
		ReconnectAttempts.Load(),
		ReceiveErrors.Load(),
		GetDroppedPacketCount(),
		GetLastSequence()
	);
}

bool FFerretGazeSubscriberWorker::ParsePayloadJson(const FString& Payload, FFerretGazePacket& OutPacket) const
{
	auto TryReadVec3 = [](const TSharedPtr<FJsonObject>& Obj, const TCHAR* Field, FVector& OutVec) -> bool
	{
		const TArray<TSharedPtr<FJsonValue>>* RawArray = nullptr;
		if (!Obj->TryGetArrayField(Field, RawArray) || RawArray == nullptr || RawArray->Num() != 3)
		{
			return false;
		}
		OutVec = FVector(
			static_cast<float>((*RawArray)[0]->AsNumber()),
			static_cast<float>((*RawArray)[1]->AsNumber()),
			static_cast<float>((*RawArray)[2]->AsNumber())
		);
		return true;
	};

	TSharedPtr<FJsonObject> Root;
	const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Payload);
	if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
	{
		return false;
	}

	// Keep parser tolerant; missing fields preserve defaults.
	double RawNumber = 0.0;
	if (Root->TryGetNumberField(TEXT("seq"), RawNumber))
	{
		OutPacket.Sequence = static_cast<int64>(RawNumber);
	}
	if (Root->TryGetNumberField(TEXT("capture_utc_ns"), RawNumber))
	{
		OutPacket.CaptureUtcNs = static_cast<int64>(RawNumber);
	}
	if (Root->TryGetNumberField(TEXT("process_start_ns"), RawNumber))
	{
		OutPacket.ProcessStartNs = static_cast<int64>(RawNumber);
	}
	if (Root->TryGetNumberField(TEXT("publish_utc_ns"), RawNumber))
	{
		OutPacket.PublishUtcNs = static_cast<int64>(RawNumber);
	}
	if (Root->TryGetNumberField(TEXT("confidence"), RawNumber))
	{
		OutPacket.Confidence = static_cast<float>(RawNumber);
	}

	TryReadVec3(Root, TEXT("skull_position_xyz"), OutPacket.SkullPositionCm);
	TryReadVec3(Root, TEXT("left_eye_origin_xyz"), OutPacket.LeftEyeOriginCm);
	TryReadVec3(Root, TEXT("left_gaze_direction_xyz"), OutPacket.LeftGazeDirection);
	TryReadVec3(Root, TEXT("right_eye_origin_xyz"), OutPacket.RightEyeOriginCm);
	TryReadVec3(Root, TEXT("right_gaze_direction_xyz"), OutPacket.RightGazeDirection);

	// Python packets are [w, x, y, z]; Unreal FQuat ctor is (x, y, z, w).
	const TArray<TSharedPtr<FJsonValue>>* QuatRawArray = nullptr;
	if (Root->TryGetArrayField(TEXT("skull_quaternion_wxyz"), QuatRawArray) && QuatRawArray != nullptr && QuatRawArray->Num() == 4)
	{
		const float W = static_cast<float>((*QuatRawArray)[0]->AsNumber());
		const float X = static_cast<float>((*QuatRawArray)[1]->AsNumber());
		const float Y = static_cast<float>((*QuatRawArray)[2]->AsNumber());
		const float Z = static_cast<float>((*QuatRawArray)[3]->AsNumber());
		OutPacket.SkullOrientation = FQuat(X, Y, Z, W);
	}

	return true;
}
