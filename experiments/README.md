# Communication approaches experiments

## Launch:
**gRPC server**:
```bash
python grpc_server.py -w <WORLD_SIZE>
```
Two versions of **gRPC client** implementations are now presented:
`grpc_task_client.py` and `grpc_client.py`. 
The `grpc_task_client.py` client implementation relies on a task queue and launches client with a context manager. 
Both are WIP implementations. But preferably, for experiments, you should launch client via:
```bash
python grpc_task_client.py
```
Clients will not start an exchange until `<WORLD_SIZE>` client processes will be launched.

---
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
