"""
Key-Value Store Server with WAL for durability.
Supports Set, Get, Delete, BulkSet operations.
"""
import os
import json
import random
import threading
import time
from flask import Flask, request, jsonify
from datetime import datetime

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
    
    def clear(self):
        """Clear WAL after checkpoint."""
        with self.lock:
            if os.path.exists(self.wal_file):
                os.remove(self.wal_file)


class KVStore:
    """Key-Value Store with persistence and WAL."""
    
    def __init__(self, data_file="kvstore.json", wal_file="wal.log", node_id=None):
        self.data_file = data_file
        self.wal = WAL(wal_file)
        self.data = {}
        self.lock = threading.Lock()
        self.node_id = node_id
        
        # Cluster configuration
        self.is_primary = True
        self.primary_url = None
        self.secondaries = []
        self.cluster_mode = False
        
        # Index structures
        self.inverted_index = {}  # word -> set of keys
        self.embeddings = {}  # key -> embedding vector
        
        self._recover()
    
    def _recover(self):
        """Recover state from WAL and data file."""
        # Load base data
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, "r") as f:
                    self.data = json.load(f)
            except:
                self.data = {}
        
        # Replay WAL
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
        
        # Rebuild indexes
        self._rebuild_indexes()
        
        # Checkpoint
        self._checkpoint()
    
    def _rebuild_indexes(self):
        """Rebuild all indexes from data."""
        self.inverted_index = {}
        for key, value in self.data.items():
            self._index_value(key, value)
    
    def _index_value(self, key, value):
        """Add value to inverted index."""
        if isinstance(value, str):
            words = value.lower().split()
            for word in words:
                if word not in self.inverted_index:
                    self.inverted_index[word] = set()
                self.inverted_index[word].add(key)
    
    def _remove_from_index(self, key):
        """Remove key from all indexes."""
        for word in list(self.inverted_index.keys()):
            self.inverted_index[word].discard(key)
            if not self.inverted_index[word]:
                del self.inverted_index[word]
    
    def _checkpoint(self):
        """Save current state to disk and clear WAL."""
        with self.lock:
            with open(self.data_file, "w") as f:
                json.dump(self.data, f)
                f.flush()
                os.fsync(f.fileno())
            self.wal.clear()
    
    def _save(self, debug_fail=False):
        """Save data to disk with optional simulated failure."""
        if debug_fail:
            if random.random() < 0.01:
                return False
        with open(self.data_file, "w") as f:
            json.dump(self.data, f)
            f.flush()
            os.fsync(f.fileno())
        return True
    
    def get(self, key):
        """Get value by key."""
        with self.lock:
            return self.data.get(key)
    
    def set(self, key, value, debug_fail=False):
        """Set key-value pair."""
        with self.lock:
            # WAL first (always synchronous)
            self.wal.append("set", {"key": key, "value": value})
            
            # Update in-memory
            old_value = self.data.get(key)
            self.data[key] = value
            
            # Update indexes
            if old_value is not None:
                self._remove_from_index(key)
            self._index_value(key, value)
            
            # Save to disk
            self._save(debug_fail)
            
            return True
    
    def delete(self, key, debug_fail=False):
        """Delete key."""
        with self.lock:
            if key in self.data:
                # WAL first
                self.wal.append("delete", {"key": key})
                
                # Update indexes
                self._remove_from_index(key)
                
                # Update in-memory
                del self.data[key]
                
                # Save to disk
                self._save(debug_fail)
                
                return True
            return False
    
    def bulk_set(self, items, debug_fail=False):
        """Bulk set multiple key-value pairs atomically."""
        with self.lock:
            # WAL first (atomic operation)
            self.wal.append("bulk_set", {"items": items})
            
            # Update in-memory
            for key, value in items:
                old_value = self.data.get(key)
                if old_value is not None:
                    self._remove_from_index(key)
                self.data[key] = value
                self._index_value(key, value)
            
            # Save to disk
            self._save(debug_fail)
            
            return True
    
    def search(self, query):
        """Full-text search using inverted index."""
        words = query.lower().split()
        if not words:
            return []
        
        # Find keys containing all words (AND search)
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
    
    def get_all_keys(self):
        """Get all keys."""
        with self.lock:
            return list(self.data.keys())


# Global store instance
store = None


def get_store():
    global store
    if store is None:
        data_dir = os.environ.get("KVSTORE_DATA_DIR", ".")
        node_id = os.environ.get("KVSTORE_NODE_ID", "node1")
        data_file = os.path.join(data_dir, f"kvstore_{node_id}.json")
        wal_file = os.path.join(data_dir, f"wal_{node_id}.log")
        store = KVStore(data_file, wal_file, node_id)
    return store


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
    
    get_store().set(key, value, debug_fail)
    return jsonify({"success": True, "key": key})


@app.route("/delete/<key>", methods=["DELETE"])
def delete_key(key):
    """Delete key."""
    debug_fail = request.args.get("debug_fail", "false").lower() == "true"
    success = get_store().delete(key, debug_fail)
    if not success:
        return jsonify({"error": "Key not found"}), 404
    return jsonify({"success": True, "key": key})


@app.route("/bulk_set", methods=["POST"])
def bulk_set():
    """Bulk set multiple key-value pairs."""
    data = request.get_json()
    items = data.get("items", [])
    debug_fail = data.get("debug_fail", False)
    
    if not items:
        return jsonify({"error": "No items provided"}), 400
    
    get_store().bulk_set(items, debug_fail)
    return jsonify({"success": True, "count": len(items)})


@app.route("/search", methods=["GET"])
def search():
    """Full-text search."""
    query = request.args.get("q", "")
    results = get_store().search(query)
    return jsonify({"results": results})


@app.route("/keys", methods=["GET"])
def get_keys():
    """Get all keys."""
    keys = get_store().get_all_keys()
    return jsonify({"keys": keys})


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "node_id": get_store().node_id})


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    app.run(host="0.0.0.0", port=port, threaded=True)
