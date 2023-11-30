# Communication approaches experiments


### Serialization types:
- Protobuf serialization (`TensorProto` message type in `protos/services.proto`)
- [Safetensors](https://huggingface.co/docs/safetensors/api/torch) serialization (`SafetensorDataProto` message type in `protos/services.proto`)

### Methods:
- Batched bidirectional stream-stream RPC
- Unary RPC 

### Variables:
- Serializer (protobuf / safetensors)
- Number of rows (1_000_000);
- Number of columns (10; 100);
- Number of clients (2; 3; 5; 10);
- Batch size (0.1, 0.01);
- Network (All devices in one network / Server in one network + devices in other / All in different networks);
