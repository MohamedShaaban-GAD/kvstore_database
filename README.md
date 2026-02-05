# KVStore - Distributed Key-Value Store

A persistent, distributed key-value store built with Python and Flask.

## Features

- **Basic Operations**: Set, Get, Delete, BulkSet
- **Persistence**: Write-Ahead Log (WAL) for durability
- **Cluster Mode**: Primary/Secondary replication with automatic leader election
- **Masterless Mode**: Write to any node
- **Indexes**:
  - Inverted Index (Full-text search)
  - Word Embeddings (Semantic search)
- **ACID Compliance**: Atomic bulk operations with crash recovery

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Start a Single Server

```bash
python server.py 5000
```

### Use the Client

```python
from client import KVStoreClient

client = KVStoreClient(host="localhost", port=5000)

# Set a value
client.set("key1", "value1")

# Get a value
value = client.get("key1")

# Delete a key
client.delete("key1")

# Bulk set
client.bulk_set([("k1", "v1"), ("k2", "v2"), ("k3", "v3")])
```

## Cluster Mode

### Start a 3-Node Cluster

```bash
# Terminal 1
set KVSTORE_NODE_ID=node1
python cluster_server.py 5001

# Terminal 2
set KVSTORE_NODE_ID=node2
python cluster_server.py 5002

# Terminal 3
set KVSTORE_NODE_ID=node3
python cluster_server.py 5003
```

### Join Nodes to Cluster

```python
import requests

# Join node1 to cluster
requests.post("http://localhost:5001/cluster/join", json={
    "peers": [
        {"node_id": "node2", "url": "http://localhost:5002"},
        {"node_id": "node3", "url": "http://localhost:5003"}
    ]
})

# Make node1 primary
requests.post("http://localhost:5001/become_primary")
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/get/<key>` | GET | Get value by key |
| `/set` | POST | Set key-value pair |
| `/delete/<key>` | DELETE | Delete a key |
| `/bulk_set` | POST | Bulk set multiple key-value pairs |
| `/search?q=<query>` | GET | Full-text search |
| `/semantic_search?q=<query>&k=<num>` | GET | Semantic search |
| `/health` | GET | Health check |
| `/cluster/status` | GET | Cluster status |

## Running Tests

```bash
# Run all tests
python run.py --tests

# Run benchmarks
python run.py --benchmarks

# Run everything
python run.py --all
```

## Architecture

### Write-Ahead Log (WAL)
- All writes go to WAL first (synchronous)
- Data file updated after WAL
- Recovery replays WAL on startup
- Ensures 100% durability for acknowledged writes

### Leader Election
- Simplified Raft-like election
- Heartbeat-based failure detection
- Automatic failover when primary fails

### Indexes
- **Inverted Index**: Maps words to document keys for fast text search
- **Embeddings**: Simple word vector averaging for semantic similarity

## Debug Mode

For testing crash scenarios, use the `debug_fail` parameter:

```python
client.set("key", "value", debug_fail=True)  # 1% chance of simulated failure
```

## File Structure

```
kvstore/
├── server.py          # Simple single-node server
├── cluster_server.py  # Clustered server with replication
├── client.py          # Python client library
├── test_kvstore.py    # Basic tests
├── test_cluster.py    # Cluster tests
├── benchmark.py       # Performance benchmarks
├── run.py             # Test runner
├── requirements.txt   # Dependencies
└── README.md          # This file
```
