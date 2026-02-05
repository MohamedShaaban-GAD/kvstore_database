"""
KVStore Client - Python client for the Key-Value Store.
"""
import requests
from typing import List, Tuple, Optional, Any


class KVStoreClient:
    """Client for the Key-Value Store."""
    
    def __init__(self, host: str = "localhost", port: int = 5000, timeout: int = 10):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
    
    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        try:
            response = requests.get(
                f"{self.base_url}/get/{key}",
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json().get("value")
            return None
        except requests.RequestException:
            return None
    
    def set(self, key: str, value: Any, debug_fail: bool = False) -> bool:
        """Set key-value pair."""
        try:
            response = requests.post(
                f"{self.base_url}/set",
                json={"key": key, "value": value, "debug_fail": debug_fail},
                timeout=self.timeout
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def delete(self, key: str, debug_fail: bool = False) -> bool:
        """Delete key."""
        try:
            response = requests.delete(
                f"{self.base_url}/delete/{key}",
                params={"debug_fail": str(debug_fail).lower()},
                timeout=self.timeout
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def bulk_set(self, items: List[Tuple[str, Any]], debug_fail: bool = False) -> bool:
        """Bulk set multiple key-value pairs."""
        try:
            response = requests.post(
                f"{self.base_url}/bulk_set",
                json={"items": items, "debug_fail": debug_fail},
                timeout=self.timeout
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def search(self, query: str) -> List[Tuple[str, Any]]:
        """Full-text search."""
        try:
            response = requests.get(
                f"{self.base_url}/search",
                params={"q": query},
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json().get("results", [])
            return []
        except requests.RequestException:
            return []
    
    def get_all_keys(self) -> List[str]:
        """Get all keys."""
        try:
            response = requests.get(
                f"{self.base_url}/keys",
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json().get("keys", [])
            return []
        except requests.RequestException:
            return []
    
    def health(self) -> bool:
        """Check if server is healthy."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )
            return response.status_code == 200
        except requests.RequestException:
            return False


class ClusterClient:
    """Client for clustered KVStore with automatic failover."""
    
    def __init__(self, nodes: List[Tuple[str, int]], timeout: int = 5):
        self.nodes = nodes
        self.timeout = timeout
        self.clients = [KVStoreClient(host, port, timeout) for host, port in nodes]
        self.primary_idx = 0
    
    def _find_primary(self) -> Optional[KVStoreClient]:
        """Find the current primary node."""
        for i, client in enumerate(self.clients):
            try:
                response = requests.get(
                    f"{client.base_url}/cluster/status",
                    timeout=self.timeout
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("is_primary"):
                        self.primary_idx = i
                        return client
            except:
                continue
        return self.clients[self.primary_idx] if self.clients else None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value (from primary)."""
        client = self._find_primary()
        return client.get(key) if client else None
    
    def set(self, key: str, value: Any, debug_fail: bool = False) -> bool:
        """Set value (on primary)."""
        client = self._find_primary()
        return client.set(key, value, debug_fail) if client else False
    
    def delete(self, key: str, debug_fail: bool = False) -> bool:
        """Delete key (on primary)."""
        client = self._find_primary()
        return client.delete(key, debug_fail) if client else False
    
    def bulk_set(self, items: List[Tuple[str, Any]], debug_fail: bool = False) -> bool:
        """Bulk set (on primary)."""
        client = self._find_primary()
        return client.bulk_set(items, debug_fail) if client else False
