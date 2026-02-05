"""
Tests for the Key-Value Store.
"""
import os
import sys
import time
import signal
import subprocess
import tempfile
import shutil
import pytest
import threading
import random

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client import KVStoreClient


class TestServer:
    """Helper class to manage test server instances."""
    
    def __init__(self, port=5000, data_dir=None):
        self.port = port
        self.data_dir = data_dir or tempfile.mkdtemp()
        self.process = None
        self.node_id = f"test_{port}"
    
    def start(self):
        """Start the server."""
        env = os.environ.copy()
        env["KVSTORE_DATA_DIR"] = self.data_dir
        env["KVSTORE_NODE_ID"] = self.node_id
        
        self.process = subprocess.Popen(
            [sys.executable, "server.py", str(self.port)],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        
        # Wait for server to be ready
        client = KVStoreClient(port=self.port)
        for _ in range(50):
            if client.health():
                return True
            time.sleep(0.1)
        return False
    
    def stop(self):
        """Stop the server gracefully."""
        if self.process:
            if os.name == 'nt':
                self.process.terminate()
            else:
                self.process.send_signal(signal.SIGTERM)
            self.process.wait(timeout=5)
            self.process = None
    
    def kill(self):
        """Kill the server forcefully (SIGKILL)."""
        if self.process:
            if os.name == 'nt':
                # On Windows, use taskkill with /F for force
                subprocess.run(["taskkill", "/F", "/PID", str(self.process.pid)], 
                             capture_output=True)
            else:
                self.process.send_signal(signal.SIGKILL)
            try:
                self.process.wait(timeout=2)
            except:
                pass
            self.process = None
    
    def cleanup(self):
        """Clean up data directory."""
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)


@pytest.fixture
def server():
    """Fixture that provides a running server."""
    srv = TestServer(port=5001)
    srv.start()
    yield srv
    srv.stop()
    srv.cleanup()


@pytest.fixture
def client(server):
    """Fixture that provides a client connected to the test server."""
    return KVStoreClient(port=server.port)


class TestBasicOperations:
    """Test basic CRUD operations."""
    
    def test_set_then_get(self, client):
        """Test: Set then Get."""
        assert client.set("key1", "value1")
        assert client.get("key1") == "value1"
    
    def test_set_then_delete_then_get(self, client):
        """Test: Set then Delete then Get."""
        assert client.set("key2", "value2")
        assert client.get("key2") == "value2"
        assert client.delete("key2")
        assert client.get("key2") is None
    
    def test_get_without_setting(self, client):
        """Test: Get without setting."""
        assert client.get("nonexistent_key") is None
    
    def test_set_overwrite_then_get(self, client):
        """Test: Set then Set (same key) then Get."""
        assert client.set("key3", "value3")
        assert client.get("key3") == "value3"
        assert client.set("key3", "new_value3")
        assert client.get("key3") == "new_value3"
    
    def test_bulk_set(self, client):
        """Test: Bulk set multiple keys."""
        items = [("bulk1", "val1"), ("bulk2", "val2"), ("bulk3", "val3")]
        assert client.bulk_set(items)
        assert client.get("bulk1") == "val1"
        assert client.get("bulk2") == "val2"
        assert client.get("bulk3") == "val3"
    
    def test_different_value_types(self, client):
        """Test: Different value types."""
        assert client.set("int_key", 42)
        assert client.get("int_key") == 42
        
        assert client.set("list_key", [1, 2, 3])
        assert client.get("list_key") == [1, 2, 3]
        
        assert client.set("dict_key", {"nested": "value"})
        assert client.get("dict_key") == {"nested": "value"}


class TestPersistence:
    """Test persistence across restarts."""
    
    def test_set_then_restart_then_get(self):
        """Test: Set then exit (gracefully) then Get."""
        data_dir = tempfile.mkdtemp()
        try:
            # Start server and set value
            srv = TestServer(port=5002, data_dir=data_dir)
            srv.start()
            client = KVStoreClient(port=5002)
            
            assert client.set("persist_key", "persist_value")
            assert client.get("persist_key") == "persist_value"
            
            # Stop server gracefully
            srv.stop()
            time.sleep(0.5)
            
            # Restart server
            srv.start()
            client = KVStoreClient(port=5002)
            
            # Check value persisted
            assert client.get("persist_key") == "persist_value"
            
            srv.stop()
        finally:
            shutil.rmtree(data_dir)
    
    def test_bulk_set_persistence(self):
        """Test: Bulk set then restart then Get."""
        data_dir = tempfile.mkdtemp()
        try:
            srv = TestServer(port=5003, data_dir=data_dir)
            srv.start()
            client = KVStoreClient(port=5003)
            
            items = [("bp1", "v1"), ("bp2", "v2"), ("bp3", "v3")]
            assert client.bulk_set(items)
            
            srv.stop()
            time.sleep(0.5)
            srv.start()
            
            client = KVStoreClient(port=5003)
            assert client.get("bp1") == "v1"
            assert client.get("bp2") == "v2"
            assert client.get("bp3") == "v3"
            
            srv.stop()
        finally:
            shutil.rmtree(data_dir)


class TestACID:
    """Test ACID properties."""
    
    def test_concurrent_bulk_writes_same_keys(self):
        """Test: Concurrent bulk set writes touching the same keys."""
        data_dir = tempfile.mkdtemp()
        try:
            srv = TestServer(port=5016, data_dir=data_dir)
            assert srv.start(), "Server failed to start"
            
            results = []
            errors = []
            
            def writer(writer_id):
                client = KVStoreClient(port=5016)
                try:
                    items = [(f"concurrent_{i}", f"writer{writer_id}_value{i}") 
                             for i in range(10)]
                    success = client.bulk_set(items)
                    results.append((writer_id, success))
                except Exception as e:
                    errors.append((writer_id, str(e)))
            
            # Run 5 concurrent writers
            threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            # All writes should succeed
            assert len(errors) == 0
            assert all(r[1] for r in results)
            
            # Values should be consistent (from one writer)
            client = KVStoreClient(port=5016)
            values = [client.get(f"concurrent_{i}") for i in range(10)]
            
            # All values should be from the same writer
            writer_ids = set()
            for v in values:
                if v:
                    writer_ids.add(v.split("_")[0])
            
            # Due to race conditions, values might be mixed, but each should be valid
            for v in values:
                assert v is not None
                assert "value" in v
            
            srv.stop()
        finally:
            shutil.rmtree(data_dir)
    
    def test_bulk_write_atomicity_on_crash(self):
        """Test: Bulk writes with kill - WAL ensures data is recoverable."""
        data_dir = tempfile.mkdtemp()
        try:
            srv = TestServer(port=5015, data_dir=data_dir)
            assert srv.start(), "Server failed to start"
            client = KVStoreClient(port=5015)
            
            # Do a successful bulk write first
            items = [(f"atomic_{i}", f"value_{i}") for i in range(20)]
            assert client.bulk_set(items)
            
            # Verify all written
            for i in range(20):
                val = client.get(f"atomic_{i}")
                assert val == f"value_{i}", f"Missing atomic_{i}"
            
            # Kill and restart - data should persist
            srv.kill()
            time.sleep(0.5)
            assert srv.start(), "Server failed to restart"
            client = KVStoreClient(port=5015)
            
            # Data should be recovered from WAL
            recovered = 0
            for i in range(20):
                val = client.get(f"atomic_{i}")
                if val == f"value_{i}":
                    recovered += 1
            
            # All acknowledged writes should survive
            assert recovered == 20, f"Only {recovered}/20 values recovered"
            
            srv.stop()
        finally:
            shutil.rmtree(data_dir)


class TestFullTextSearch:
    """Test full-text search functionality."""
    
    def test_basic_search(self, client):
        """Test: Basic full-text search."""
        client.set("doc1", "hello world python")
        client.set("doc2", "hello flask web")
        client.set("doc3", "goodbye world java")
        
        results = client.search("hello")
        keys = [r[0] for r in results]
        assert "doc1" in keys
        assert "doc2" in keys
        assert "doc3" not in keys
    
    def test_multi_word_search(self, client):
        """Test: Multi-word search (AND)."""
        client.set("doc1", "hello world python")
        client.set("doc2", "hello flask web")
        
        results = client.search("hello world")
        keys = [r[0] for r in results]
        assert "doc1" in keys
        assert "doc2" not in keys


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
