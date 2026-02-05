"""
Benchmarks for the Key-Value Store.
"""
import os
import sys
import time
import signal
import subprocess
import tempfile
import shutil
import threading
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import KVStoreClient


class BenchmarkServer:
    """Helper class to manage benchmark server instances."""
    
    def __init__(self, port=5000, data_dir=None):
        self.port = port
        self.data_dir = data_dir or tempfile.mkdtemp()
        self.process = None
        self.node_id = f"bench_{port}"
    
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
            try:
                self.process.wait(timeout=5)
            except:
                pass
            self.process = None
    
    def kill(self):
        """Kill the server forcefully."""
        if self.process:
            if os.name == 'nt':
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
            shutil.rmtree(self.data_dir, ignore_errors=True)


def benchmark_write_throughput():
    """Benchmark write throughput with different data sizes."""
    print("\n" + "="*60)
    print("BENCHMARK: Write Throughput")
    print("="*60)
    
    data_dir = tempfile.mkdtemp()
    results = []
    
    try:
        for pre_populated_size in [0, 100, 500, 1000]:
            srv = BenchmarkServer(port=5010, data_dir=data_dir)
            srv.start()
            client = KVStoreClient(port=5010)
            
            # Pre-populate data
            if pre_populated_size > 0:
                items = [(f"pre_{i}", f"value_{i}" * 10) for i in range(pre_populated_size)]
                client.bulk_set(items)
            
            # Benchmark writes
            num_writes = 100
            start_time = time.time()
            
            for i in range(num_writes):
                client.set(f"bench_key_{i}", f"bench_value_{i}" * 10)
            
            elapsed = time.time() - start_time
            writes_per_second = num_writes / elapsed
            
            results.append({
                "pre_populated": pre_populated_size,
                "writes": num_writes,
                "time": elapsed,
                "writes_per_second": writes_per_second
            })
            
            print(f"Pre-populated: {pre_populated_size:5d} | "
                  f"Writes: {num_writes} | "
                  f"Time: {elapsed:.3f}s | "
                  f"Throughput: {writes_per_second:.1f} writes/sec")
            
            srv.stop()
            
            # Clear data for next run
            for f in os.listdir(data_dir):
                os.remove(os.path.join(data_dir, f))
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)
    
    return results


def benchmark_bulk_write_throughput():
    """Benchmark bulk write throughput."""
    print("\n" + "="*60)
    print("BENCHMARK: Bulk Write Throughput")
    print("="*60)
    
    data_dir = tempfile.mkdtemp()
    results = []
    
    try:
        for batch_size in [10, 50, 100]:
            srv = BenchmarkServer(port=5011, data_dir=data_dir)
            srv.start()
            client = KVStoreClient(port=5011)
            
            num_batches = 20
            total_writes = num_batches * batch_size
            
            start_time = time.time()
            
            for batch in range(num_batches):
                items = [(f"bulk_{batch}_{i}", f"value_{i}" * 10) 
                         for i in range(batch_size)]
                client.bulk_set(items)
            
            elapsed = time.time() - start_time
            writes_per_second = total_writes / elapsed
            
            results.append({
                "batch_size": batch_size,
                "total_writes": total_writes,
                "time": elapsed,
                "writes_per_second": writes_per_second
            })
            
            print(f"Batch Size: {batch_size:3d} | "
                  f"Total: {total_writes:5d} | "
                  f"Time: {elapsed:.3f}s | "
                  f"Throughput: {writes_per_second:.1f} writes/sec")
            
            srv.stop()
            
            for f in os.listdir(data_dir):
                os.remove(os.path.join(data_dir, f))
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)
    
    return results


def benchmark_read_throughput():
    """Benchmark read throughput."""
    print("\n" + "="*60)
    print("BENCHMARK: Read Throughput")
    print("="*60)
    
    data_dir = tempfile.mkdtemp()
    
    try:
        srv = BenchmarkServer(port=5012, data_dir=data_dir)
        srv.start()
        client = KVStoreClient(port=5012)
        
        # Pre-populate
        items = [(f"read_{i}", f"value_{i}" * 10) for i in range(500)]
        client.bulk_set(items)
        
        # Benchmark reads
        num_reads = 500
        start_time = time.time()
        
        for i in range(num_reads):
            client.get(f"read_{i % 500}")
        
        elapsed = time.time() - start_time
        reads_per_second = num_reads / elapsed
        
        print(f"Reads: {num_reads} | "
              f"Time: {elapsed:.3f}s | "
              f"Throughput: {reads_per_second:.1f} reads/sec")
        
        srv.stop()
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def benchmark_durability():
    """Benchmark durability under random crashes."""
    print("\n" + "="*60)
    print("BENCHMARK: Durability (Crash Recovery)")
    print("="*60)
    
    data_dir = tempfile.mkdtemp()
    
    acknowledged_keys = set()
    lock = threading.Lock()
    stop_flag = threading.Event()
    
    try:
        srv = BenchmarkServer(port=5013, data_dir=data_dir)
        srv.start()
        
        def writer_thread():
            """Thread that writes data and tracks acknowledged keys."""
            client = KVStoreClient(port=5013, timeout=2)
            key_counter = 0
            
            while not stop_flag.is_set():
                key = f"durability_{key_counter}"
                value = f"value_{key_counter}"
                
                try:
                    if client.set(key, value):
                        with lock:
                            acknowledged_keys.add(key)
                        key_counter += 1
                except:
                    # Server might be down, retry
                    time.sleep(0.1)
        
        def killer_thread():
            """Thread that randomly kills the server."""
            nonlocal srv
            
            for _ in range(3):  # Kill 3 times
                time.sleep(random.uniform(0.3, 0.5))
                
                if stop_flag.is_set():
                    break
                
                print("  [Killer] Killing server...")
                srv.kill()
                time.sleep(0.2)
                
                print("  [Killer] Restarting server...")
                srv = BenchmarkServer(port=5013, data_dir=data_dir)
                srv.start()
        
        # Start threads
        writer = threading.Thread(target=writer_thread)
        killer = threading.Thread(target=killer_thread)
        
        writer.start()
        killer.start()
        
        # Let it run for a short time
        killer.join(timeout=5)
        stop_flag.set()
        writer.join(timeout=2)
        
        # Wait for server to be stable
        time.sleep(0.5)
        if srv.process is None or srv.process.poll() is not None:
            srv = BenchmarkServer(port=5013, data_dir=data_dir)
            srv.start()
        
        # Check which acknowledged keys survived
        client = KVStoreClient(port=5013, timeout=5)
        
        survived_keys = set()
        lost_keys = set()
        
        with lock:
            ack_keys = acknowledged_keys.copy()
        
        for key in ack_keys:
            value = client.get(key)
            if value is not None:
                survived_keys.add(key)
            else:
                lost_keys.add(key)
        
        total_ack = len(ack_keys)
        total_survived = len(survived_keys)
        total_lost = len(lost_keys)
        
        durability_pct = (total_survived / total_ack * 100) if total_ack > 0 else 100
        
        print(f"\nDurability Results:")
        print(f"  Acknowledged writes: {total_ack}")
        print(f"  Survived after crashes: {total_survived}")
        print(f"  Lost keys: {total_lost}")
        print(f"  Durability: {durability_pct:.2f}%")
        
        srv.stop()
        
        return {
            "acknowledged": total_ack,
            "survived": total_survived,
            "lost": total_lost,
            "durability_pct": durability_pct
        }
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def benchmark_concurrent_writes():
    """Benchmark concurrent write performance."""
    print("\n" + "="*60)
    print("BENCHMARK: Concurrent Writes")
    print("="*60)
    
    data_dir = tempfile.mkdtemp()
    
    try:
        srv = BenchmarkServer(port=5014, data_dir=data_dir)
        srv.start()
        
        for num_threads in [1, 2, 4]:
            writes_per_thread = 50
            total_writes = num_threads * writes_per_thread
            
            def writer(thread_id):
                client = KVStoreClient(port=5014)
                for i in range(writes_per_thread):
                    client.set(f"concurrent_{thread_id}_{i}", f"value_{i}")
            
            start_time = time.time()
            
            threads = [threading.Thread(target=writer, args=(i,)) 
                      for i in range(num_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            elapsed = time.time() - start_time
            writes_per_second = total_writes / elapsed
            
            print(f"Threads: {num_threads} | "
                  f"Total Writes: {total_writes} | "
                  f"Time: {elapsed:.3f}s | "
                  f"Throughput: {writes_per_second:.1f} writes/sec")
        
        srv.stop()
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def run_all_benchmarks():
    """Run all benchmarks."""
    print("\n" + "#"*60)
    print("# KVStore Benchmarks")
    print("#"*60)
    
    benchmark_write_throughput()
    benchmark_bulk_write_throughput()
    benchmark_read_throughput()
    benchmark_concurrent_writes()
    benchmark_durability()
    
    print("\n" + "#"*60)
    print("# Benchmarks Complete")
    print("#"*60)


if __name__ == "__main__":
    run_all_benchmarks()
