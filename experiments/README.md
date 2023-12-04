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


## To run experiment:
0. Set the variables in experiments/bin/expctl
```bash
num_rows=1000000
num_cols=10
batch_size=10000
```
1. Upload files to cluster:
```bash
SYNC_HOST=<hostname> HOST_NAME=<username> bash rsync-repo upload
```
2. On the main node:
e.g. id_of_start_node=3, id_of_last_node=6 (node3.bdcl - server, node4/5/6.bdcl - clients)

```bash
MAIN_NODE=<id_of_start_node> LAST_NODE=<id_of_last_node> bash experiments/bin/expctl pull-on-nodes
MAIN_NODE=<id_of_start_node> LAST_NODE=<id_of_last_node> bash experiments/bin/expctl run
```

3. Check logs
```bash
docker logs --follow vfl-experiments
```
3. Stop the server (and clients if still run)
```bash
MAIN_NODE=<id_of_start_node> LAST_NODE=<id_of_last_node> bash experiments/bin/expctl halt
```