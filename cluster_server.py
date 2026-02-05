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
import numpy as np
from collections import defaultdict

app = Flask(__name__)


class WAL:
    """Write-Ahead Log for durability."""
    
    def __init__(self, wal_file="wal.log"):
        self.wal_file = wal_file
        self.lock = threading.Lock()
    
    def append(self, operation, data):
        """Append operation to WAL synchronously."""
        with self.lock:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "data": data
            }
            with open(self.wal_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
                f.flush()
                os.fsync(f.fileno())
            return entry
    
    def read_all(self):
        """Read all entries from WAL."""
        entries = []
        if os.path.exists(self.wal_file):
            with open(self.wal_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        return entries
    
    def get_last_index(self):
        """Get the last log index."""
        entries = self.read_all()
        return len(entries)
    
    def clear(self):
        """Clear WAL after checkpoint."""
        with self.lock:
            if os.path.exists(self.wal_file):
                os.remove(self.wal_file)


class SimpleEmbedding:
    """Simple word embedding using random vectors (for demo)."""
    
    def __init__(self, dim=50):
        self.dim = dim
        self.word_vectors = {}
    
    def get_word_vector(self, word):
        """Get or create a vector for a word."""
        word = word.lower()
        if word not in self.word_vectors:
            # Create a deterministic random vector based on word hash
            random.seed(hash(word))
            self.word_vectors[word] = [random.gauss(0, 1) for _ in range(self.dim)]
        return self.word_vectors[word]
    
    def get_text_embedding(self, text):
        """Get embedding for text (average of word vectors)."""
        words = text.lower().split()
        if not words:
            return [0.0] * self.dim
        
        vectors = [self.get_word_vector(w) for w in words]
        avg = [sum(v[i] for v in vectors) / len(vectors) for i in range(self.dim)]
        return avg
    
    def cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0
        return dot / (norm1 * norm2)


class ClusteredKVStore:
    """Key-Value Store with clustering, replication, and indexes."""
    
    def __init__(self, data_file="kvstore.json", wal_file="wal.log", node_id=None):
        self.data_file = data_file
        self.wal = WAL(wal_file)
        self.data = {}
        self.lock = threading.Lock()
        self.node_id = node_id or "node1"
        
        # Cluster configuration
        self.is_primary = False
        self.primary_url = None
        self.peers = []  # List of (node_id, url) tuples
        self.cluster_mode = False
        
        # Leader election
        self.term = 0
        self.voted_for = None
        self.last_heartbeat = time.time()
        self.election_timeout = random.uniform(1.5, 3.0)
        self.heartbeat_interval = 0.5
        
        # Index structures
        self.inverted_index = {}  # word -> set of keys
        self.embeddings = {}  # key -> embedding vector
        self.embedding_model = SimpleEmbedding()
        
        # Masterless mode
        self.masterless = False
        self.vector_clock = defaultdict(int)
        
        self._recover()
    
    def _recover(self):
        """Recover state from WAL and data file."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, "r") as f:
                    self.data = json.load(f)
            except:
                self.data = {}
        
        for entry in self.wal.read_all():
            op = entry["operation"]
            data = entry["data"]
            if op == "set":
                self.data[data["key"]] = data["value"]
            elif op == "delete":
                self.data.pop(data["key"], None)
            elif op == "bulk_set":
                for key, value in data["items"]:
                    self.data[key] = value
        
        self._rebuild_indexes()
        self._checkpoint()
    
    def _rebuild_indexes(self):
        """Rebuild all indexes from data."""
        self.inverted_index = {}
        self.embeddings = {}
        for key, value in self.data.items():
            self._index_value(key, value)
    
    def _index_value(self, key, value):
        """Add value to indexes."""
        if isinstance(value, str):
            # Inverted index
            words = value.lower().split()
            for word in words:
                if word not in self.inverted_index:
                    self.inverted_index[word] = set()
                self.inverted_index[word].add(key)
            
            # Embedding
            self.embeddings[key] = self.embedding_model.get_text_embedding(value)
    
    def _remove_from_index(self, key):
        """Remove key from all indexes."""
        for word in list(self.inverted_index.keys()):
            self.inverted_index[word].discard(key)
            if not self.inverted_index[word]:
                del self.inverted_index[word]
        self.embeddings.pop(key, None)
    
    def _checkpoint(self):
        """Save current state to disk and clear WAL."""
        with self.lock:
            with open(self.data_file, "w") as f:
                json.dump(self.data, f)
                f.flush()
                os.fsync(f.fileno())
            self.wal.clear()
    
    def _save(self, debug_fail=False):
        """Save data to disk."""
        if debug_fail:
            if random.random() < 0.01:
                return False
        with open(self.data_file, "w") as f:
            json.dump(self.data, f)
            f.flush()
            os.fsync(f.fileno())
        return True
    
    def _replicate_to_peers(self, operation, data):
        """Replicate operation to peer nodes."""
        if not self.cluster_mode or self.masterless:
            return
        
        for node_id, url in self.peers:
            try:
                requests.post(
                    f"{url}/replicate",
                    json={"operation": operation, "data": data, "term": self.term},
                    timeout=1
                )
            except:
                pass  # Peer might be down
    
    def apply_replicated(self, operation, data):
        """Apply a replicated operation."""
        with self.lock:
            if operation == "set":
                self.data[data["key"]] = data["value"]
                self._index_value(data["key"], data["value"])
            elif operation == "delete":
                self._remove_from_index(data["key"])
                self.data.pop(data["key"], None)
            elif operation == "bulk_set":
                for key, value in data["items"]:
                    self.data[key] = value
                    self._index_value(key, value)
            self._save()
    
    def get(self, key):
        """Get value by key."""
        with self.lock:
            return self.data.get(key)
    
    def set(self, key, value, debug_fail=False):
        """Set key-value pair."""
        if self.cluster_mode and not self.is_primary and not self.masterless:
            return False, "Not primary"
        
        with self.lock:
            self.wal.append("set", {"key": key, "value": value})
            
            old_value = self.data.get(key)
            self.data[key] = value
            
            if old_value is not None:
                self._remove_from_index(key)
            self._index_value(key, value)
            
            self._save(debug_fail)
            
            if self.masterless:
                self.vector_clock[self.node_id] += 1
            
        self._replicate_to_peers("set", {"key": key, "value": value})
        return True, "OK"
    
    def delete(self, key, debug_fail=False):
        """Delete key."""
        if self.cluster_mode and not self.is_primary and not self.masterless:
            return False, "Not primary"
        
        with self.lock:
            if key in self.data:
                self.wal.append("delete", {"key": key})
                self._remove_from_index(key)
                del self.data[key]
                self._save(debug_fail)
                
                if self.masterless:
                    self.vector_clock[self.node_id] += 1
                
                self._replicate_to_peers("delete", {"key": key})
                return True, "OK"
            return False, "Key not found"
    
    def bulk_set(self, items, debug_fail=False):
        """Bulk set multiple key-value pairs atomically."""
        if self.cluster_mode and not self.is_primary and not self.masterless:
            return False, "Not primary"
        
        with self.lock:
            self.wal.append("bulk_set", {"items": items})
            
            for key, value in items:
                old_value = self.data.get(key)
                if old_value is not None:
                    self._remove_from_index(key)
                self.data[key] = value
                self._index_value(key, value)
            
            self._save(debug_fail)
            
            if self.masterless:
                self.vector_clock[self.node_id] += 1
        
        self._replicate_to_peers("bulk_set", {"items": items})
        return True, "OK"
    
    def search(self, query):
        """Full-text search using inverted index."""
        words = query.lower().split()
        if not words:
            return []
        
        result_keys = None
        for word in words:
            keys = self.inverted_index.get(word, set())
            if result_keys is None:
                result_keys = keys.copy()
            else:
                result_keys &= keys
        
        if result_keys is None:
            return []
        
        return [(key, self.data[key]) for key in result_keys if key in self.data]
    
    def semantic_search(self, query, top_k=5):
        """Semantic search using word embeddings."""
        query_embedding = self.embedding_model.get_text_embedding(query)
        
        similarities = []
        for key, embedding in self.embeddings.items():
            sim = self.embedding_model.cosine_similarity(query_embedding, embedding)
            similarities.append((key, sim, self.data.get(key)))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_all_keys(self):
        """Get all keys."""
        with self.lock:
            return list(self.data.keys())
    
    # Leader Election Methods
    def start_election(self):
        """Start a leader election."""
        self.term += 1
        self.voted_for = self.node_id
        votes = 1  # Vote for self
        
        for node_id, url in self.peers:
            try:
                response = requests.post(
                    f"{url}/vote",
                    json={"term": self.term, "candidate": self.node_id},
                    timeout=1
                )
                if response.status_code == 200 and response.json().get("granted"):
                    votes += 1
            except:
                pass
        
        # Need majority
        total_nodes = len(self.peers) + 1
        if votes > total_nodes // 2:
            self.is_primary = True
            print(f"[{self.node_id}] Became primary with {votes}/{total_nodes} votes")
            return True
        return False
    
    def handle_vote_request(self, term, candidate):
        """Handle a vote request from another node."""
        if term > self.term:
            self.term = term
            self.voted_for = None
            self.is_primary = False
        
        if term >= self.term and (self.voted_for is None or self.voted_for == candidate):
            self.voted_for = candidate
            self.last_heartbeat = time.time()
            return True
        return False
    
    def handle_heartbeat(self, term, leader):
        """Handle heartbeat from leader."""
        if term >= self.term:
            self.term = term
            self.is_primary = False
            self.primary_url = leader
            self.last_heartbeat = time.time()
            self.voted_for = None
    
    def send_heartbeats(self):
        """Send heartbeats to followers."""
        if not self.is_primary:
            return
        
        for node_id, url in self.peers:
            try:
                requests.post(
                    f"{url}/heartbeat",
                    json={"term": self.term, "leader": self.node_id},
                    timeout=0.5
                )
            except:
                pass
    
    def check_election_timeout(self):
        """Check if election timeout has passed."""
        if self.is_primary:
            return False
        return time.time() - self.last_heartbeat > self.election_timeout


# Global store instance
store = None


def get_store():
    global store
    if store is None:
        data_dir = os.environ.get("KVSTORE_DATA_DIR", ".")
        node_id = os.environ.get("KVSTORE_NODE_ID", "node1")
        data_file = os.path.join(data_dir, f"kvstore_{node_id}.json")
        wal_file = os.path.join(data_dir, f"wal_{node_id}.log")
        store = ClusteredKVStore(data_file, wal_file, node_id)
    return store


# Background thread for leader election
def election_loop():
    """Background thread for leader election and heartbeats."""
    while True:
        try:
            s = get_store()
            if s.cluster_mode and not s.masterless:
                if s.is_primary:
                    s.send_heartbeats()
                elif s.check_election_timeout():
                    print(f"[{s.node_id}] Election timeout, starting election")
                    s.start_election()
            time.sleep(0.3)
        except:
            time.sleep(1)


# Start election thread
election_thread = threading.Thread(target=election_loop, daemon=True)


@app.route("/get/<key>", methods=["GET"])
def get_key(key):
    """Get value by key."""
    value = get_store().get(key)
    if value is None:
        return jsonify({"error": "Key not found"}), 404
    return jsonify({"key": key, "value": value})


@app.route("/set", methods=["POST"])
def set_key():
    """Set key-value pair."""
    data = request.get_json()
    key = data.get("key")
    value = data.get("value")
    debug_fail = data.get("debug_fail", False)
    
    if key is None or value is None:
        return jsonify({"error": "Missing key or value"}), 400
    
    success, msg = get_store().set(key, value, debug_fail)
    if not success:
        return jsonify({"error": msg}), 400
    return jsonify({"success": True, "key": key})


@app.route("/delete/<key>", methods=["DELETE"])
def delete_key(key):
    """Delete key."""
    debug_fail = request.args.get("debug_fail", "false").lower() == "true"
    success, msg = get_store().delete(key, debug_fail)
    if not success:
        return jsonify({"error": msg}), 404
    return jsonify({"success": True, "key": key})


@app.route("/bulk_set", methods=["POST"])
def bulk_set():
    """Bulk set multiple key-value pairs."""
    data = request.get_json()
    items = data.get("items", [])
    debug_fail = data.get("debug_fail", False)
    
    if not items:
        return jsonify({"error": "No items provided"}), 400
    
    success, msg = get_store().bulk_set(items, debug_fail)
    if not success:
        return jsonify({"error": msg}), 400
    return jsonify({"success": True, "count": len(items)})


@app.route("/search", methods=["GET"])
def search():
    """Full-text search."""
    query = request.args.get("q", "")
    results = get_store().search(query)
    return jsonify({"results": results})


@app.route("/semantic_search", methods=["GET"])
def semantic_search():
    """Semantic search using embeddings."""
    query = request.args.get("q", "")
    top_k = int(request.args.get("k", 5))
    results = get_store().semantic_search(query, top_k)
    return jsonify({"results": results})


@app.route("/keys", methods=["GET"])
def get_keys():
    """Get all keys."""
    keys = get_store().get_all_keys()
    return jsonify({"keys": keys})


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    s = get_store()
    return jsonify({
        "status": "ok",
        "node_id": s.node_id,
        "is_primary": s.is_primary,
        "term": s.term
    })


@app.route("/cluster/status", methods=["GET"])
def cluster_status():
    """Get cluster status."""
    s = get_store()
    return jsonify({
        "node_id": s.node_id,
        "is_primary": s.is_primary,
        "term": s.term,
        "cluster_mode": s.cluster_mode,
        "masterless": s.masterless,
        "peers": s.peers
    })


@app.route("/cluster/join", methods=["POST"])
def cluster_join():
    """Join a cluster."""
    data = request.get_json()
    peers = data.get("peers", [])
    masterless = data.get("masterless", False)
    
    s = get_store()
    s.peers = [(p["node_id"], p["url"]) for p in peers]
    s.cluster_mode = True
    s.masterless = masterless
    
    if not masterless and not s.is_primary:
        s.last_heartbeat = time.time()
    
    return jsonify({"success": True})


@app.route("/replicate", methods=["POST"])
def replicate():
    """Receive replicated operation from primary."""
    data = request.get_json()
    operation = data.get("operation")
    op_data = data.get("data")
    term = data.get("term", 0)
    
    s = get_store()
    if term >= s.term:
        s.apply_replicated(operation, op_data)
    
    return jsonify({"success": True})


@app.route("/vote", methods=["POST"])
def vote():
    """Handle vote request."""
    data = request.get_json()
    term = data.get("term")
    candidate = data.get("candidate")
    
    granted = get_store().handle_vote_request(term, candidate)
    return jsonify({"granted": granted})


@app.route("/heartbeat", methods=["POST"])
def heartbeat():
    """Handle heartbeat from leader."""
    data = request.get_json()
    term = data.get("term")
    leader = data.get("leader")
    
    get_store().handle_heartbeat(term, leader)
    return jsonify({"success": True})


@app.route("/become_primary", methods=["POST"])
def become_primary():
    """Force this node to become primary (for testing)."""
    s = get_store()
    s.is_primary = True
    s.term += 1
    return jsonify({"success": True, "term": s.term})


if __name__ == "__main__":
    election_thread.start()
    
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    app.run(host="0.0.0.0", port=port, threaded=True)
