"""
Clustered Key-Value Store Server with Replication and Leader Election.
Supports primary/secondary replication with automatic failover.
"""
import os
import sys
import json
import random
import threading
import time
import requests
from flask import Flask, request, jsonify
from datetime import datetime
from collections import defaultdict

app = Flask(__name__)


class WriteAheadLog:
    """Write-Ahead Log for durability."""
    
    def __init__(self, log_path="wal.log"):
        self.log_path = log_path
        self.lock = threading.Lock()
    
    def append(self, operation, payload):
        """Append operation to WAL synchronously."""
        with self.lock:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "data": payload
            }
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
                f.flush()
                os.fsync(f.fileno())
            return entry
    
    def read_all(self):
        """Read all entries from WAL."""
        entries = []
        if os.path.exists(self.log_path):
            with open(self.log_path, "r") as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
        return entries
    
    def clear(self):
        """Clear WAL after checkpoint."""
        with self.lock:
            if os.path.exists(self.log_path):
                os.remove(self.log_path)


class SimpleEmbeddingModel:
    """Simple word embedding using deterministic random vectors."""
    
    def __init__(self, dimension=50):
        self.dimension = dimension
        self.word_vectors = {}
    
    def _word_vector(self, word):
        word = word.lower()
        if word not in self.word_vectors:
            random.seed(hash(word))
            self.word_vectors[word] = [
                random.gauss(0, 1) for _ in range(self.dimension)
            ]
        return self.word_vectors[word]
    
    def embed_text(self, text):
        words = text.lower().split()
        if not words:
            return [0.0] * self.dimension
        
        vectors = [self._word_vector(w) for w in words]
        return [
            sum(v[i] for v in vectors) / len(vectors)
            for i in range(self.dimension)
        ]
    
    def cosine_similarity(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class ClusteredKVStore:
    """Key-Value Store with clustering, replication, and indexes."""
    
    def __init__(self, data_path="kvstore.json", wal_path="wal.log", node_id=None):
        self.data_path = data_path
        self.wal = WriteAheadLog(wal_path)
        self.data = {}
        self.lock = threading.Lock()
        self.node_id = node_id or "node1"
        
        # Cluster state
        self.is_primary = False
        self.primary_node = None
        self.peers = []          # [(node_id, url)]
        self.cluster_enabled = False
        
        # Election state
        self.term = 0
        self.voted_for = None
        self.last_heartbeat = time.time()
        self.election_timeout = random.uniform(1.5, 3.0)
        self.heartbeat_interval = 0.5
        
        # Indexing
        self.inverted_index = {}
        self.embeddings = {}
        self.embedding_model = SimpleEmbeddingModel()
        
        # Masterless mode
        self.masterless = False
        self.vector_clock = defaultdict(int)
        
        self._recover()
    
    def _recover(self):
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, "r") as f:
                    self.data = json.load(f)
            except:
                self.data = {}
        
        for entry in self.wal.read_all():
            op = entry["operation"]
            payload = entry["data"]
            if op == "set":
                self.data[payload["key"]] = payload["value"]
            elif op == "delete":
                self.data.pop(payload["key"], None)
            elif op == "bulk_set":
                for key, value in payload["items"]:
                    self.data[key] = value
        
        self._rebuild_indexes()
        self._checkpoint()
    
    def _rebuild_indexes(self):
        self.inverted_index = {}
        self.embeddings = {}
        for key, value in self.data.items():
            self._index_value(key, value)
    
    def _index_value(self, key, value):
        if isinstance(value, str):
            for word in value.lower().split():
                self.inverted_index.setdefault(word, set()).add(key)
            self.embeddings[key] = self.embedding_model.embed_text(value)
    
    def _remove_from_index(self, key):
        for word in list(self.inverted_index.keys()):
            self.inverted_index[word].discard(key)
            if not self.inverted_index[word]:
                del self.inverted_index[word]
        self.embeddings.pop(key, None)
    
    def _checkpoint(self):
        with self.lock:
            with open(self.data_path, "w") as f:
                json.dump(self.data, f)
                f.flush()
                os.fsync(f.fileno())
            self.wal.clear()
    
    def _save(self, debug_fail=False):
        if debug_fail and random.random() < 0.01:
            return False
        with open(self.data_path, "w") as f:
            json.dump(self.data, f)
            f.flush()
            os.fsync(f.fileno())
        return True
    
    def _replicate(self, operation, payload):
        if not self.cluster_enabled or self.masterless:
            return
        for _, url in self.peers:
            try:
                requests.post(
                    f"{url}/replicate",
                    json={"operation": operation, "data": payload, "term": self.term},
                    timeout=1
                )
            except:
                pass
    
    def apply_replicated(self, operation, payload):
        with self.lock:
            if operation == "set":
                self.data[payload["key"]] = payload["value"]
                self._index_value(payload["key"], payload["value"])
            elif operation == "delete":
                self._remove_from_index(payload["key"])
                self.data.pop(payload["key"], None)
            elif operation == "bulk_set":
                for key, value in payload["items"]:
                    self.data[key] = value
                    self._index_value(key, value)
            self._save()
    
    # ===== Public KV API (unchanged names) =====
    
    def get(self, key):
        with self.lock:
            return self.data.get(key)
    
    def set(self, key, value, debug_fail=False):
        if self.cluster_enabled and not self.is_primary and not self.masterless:
            return False, "Not primary"
        
        with self.lock:
            self.wal.append("set", {"key": key, "value": value})
            if key in self.data:
                self._remove_from_index(key)
            self.data[key] = value
            self._index_value(key, value)
            self._save(debug_fail)
            if self.masterless:
                self.vector_clock[self.node_id] += 1
        
        self._replicate("set", {"key": key, "value": value})
        return True, "OK"
    
    def delete(self, key, debug_fail=False):
        if self.cluster_enabled and not self.is_primary and not self.masterless:
            return False, "Not primary"
        
        with self.lock:
            if key not in self.data:
                return False, "Key not found"
            self.wal.append("delete", {"key": key})
            self._remove_from_index(key)
            del self.data[key]
            self._save(debug_fail)
            if self.masterless:
                self.vector_clock[self.node_id] += 1
        
        self._replicate("delete", {"key": key})
        return True, "OK"
    
    def bulk_set(self, items, debug_fail=False):
        if self.cluster_enabled and not self.is_primary and not self.masterless:
            return False, "Not primary"
        
        with self.lock:
            self.wal.append("bulk_set", {"items": items})
            for key, value in items:
                if key in self.data:
                    self._remove_from_index(key)
                self.data[key] = value
                self._index_value(key, value)
            self._save(debug_fail)
            if self.masterless:
                self.vector_clock[self.node_id] += 1
        
        self._replicate("bulk_set", {"items": items})
        return True, "OK"
    
    def search(self, query):
        words = query.lower().split()
        if not words:
            return []
        result = None
        for word in words:
            keys = self.inverted_index.get(word, set())
            result = keys if result is None else result & keys
        return [(k, self.data[k]) for k in result] if result else []
    
    def semantic_search(self, query, top_k=5):
        query_vec = self.embedding_model.embed_text(query)
        scores = [
            (k, self.embedding_model.cosine_similarity(query_vec, v), self.data[k])
            for k, v in self.embeddings.items()
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def get_all_keys(self):
        with self.lock:
            return list(self.data.keys())
    
    # ===== Election / Cluster Control (unchanged behavior) =====
    
    def start_election(self):
        self.term += 1
        self.voted_for = self.node_id
        votes = 1
        
        for node_id, url in self.peers:
            try:
                r = requests.post(f"{url}/vote", json={
                    "term": self.term,
                    "candidate": self.node_id
                }, timeout=1)
                if r.status_code == 200 and r.json().get("granted"):
                    votes += 1
            except:
                pass
        
        if votes > (len(self.peers) + 1) // 2:
            self.is_primary = True
            return True
        return False
    
    def handle_vote_request(self, term, candidate):
        if term > self.term:
            self.term = term
            self.voted_for = None
            self.is_primary = False
        if self.voted_for in (None, candidate):
            self.voted_for = candidate
            self.last_heartbeat = time.time()
            return True
        return False
    
    def handle_heartbeat(self, term, leader):
        if term >= self.term:
            self.term = term
            self.is_primary = False
            self.primary_node = leader
            self.last_heartbeat = time.time()
            self.voted_for = None
    
    def send_heartbeats(self):
        if not self.is_primary:
            return
        for _, url in self.peers:
            try:
                requests.post(f"{url}/heartbeat", json={
                    "term": self.term,
                    "leader": self.node_id
                }, timeout=0.5)
            except:
                pass
    
    def election_timed_out(self):
        if self.is_primary:
            return False
        return time.time() - self.last_heartbeat > self.election_timeout

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    node_id = os.environ.get("KVSTORE_NODE_ID", f"node_{port}")
    
    print(f"Starting KVStore server on port {port} with node_id {node_id}...")
    
    kvstore = ClusteredKVStore(node_id=node_id)
    
    @app.route("/health")
    def health():
        return jsonify({
            "status": "ok",
            "node_id": kvstore.node_id,
            "is_primary": kvstore.is_primary,
            "term": kvstore.term
        })
    
    @app.route("/set", methods=["POST"])
    def set_key():
        data = request.json
        key = data.get("key")
        value = data.get("value")
        ok, msg = kvstore.set(key, value)
        return jsonify({"ok": ok, "msg": msg}), 200 if ok else 400

    @app.route("/get")
    def get_key():
        key = request.args.get("key")
        value = kvstore.get(key)
        return jsonify({"key": key, "value": value})

    app.run(host="0.0.0.0", port=port, threaded=True, debug=True)
