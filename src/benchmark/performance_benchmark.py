"""
Suite de benchmarking de performance pour Neural Chat Engine.
"""
import time
import requests
import tracemalloc
import threading
import psutil
from typing import List, Dict, Any
import logging

class PerformanceBenchmark:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.results: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

    def benchmark_api_endpoints(self, concurrent_users: List[int] = [10, 50, 100]):
        endpoint = f"{self.api_url}/chat"
        for users in concurrent_users:
            times = []
            def worker():
                start = time.time()
                try:
                    requests.post(endpoint, json={"message": "test"})
                except Exception:
                    pass
                times.append(time.time() - start)
            threads = [threading.Thread(target=worker) for _ in range(users)]
            for t in threads: t.start()
            for t in threads: t.join()
            avg_time = sum(times) / len(times) if times else 0
            self.results[f"api_{users}_users"] = avg_time
            self.logger.info(f"API {users} users: {avg_time:.3f}s")

    def benchmark_model_inference(self, model, batch_sizes: List[int] = [1, 8, 16, 32]):
        for batch in batch_sizes:
            inputs = ["test"] * batch
            start = time.time()
            try:
                _ = model(inputs)
            except Exception:
                pass
            duration = time.time() - start
            self.results[f"model_batch_{batch}"] = duration
            self.logger.info(f"Model batch {batch}: {duration:.3f}s")

    def benchmark_database_queries(self, db_session, query: str = "SELECT 1"):
        start = time.time()
        try:
            db_session.execute(query)
        except Exception:
            pass
        duration = time.time() - start
        self.results["db_query_time"] = duration
        self.logger.info(f"DB query: {duration:.3f}s")

    def benchmark_cache_performance(self, cache, key: str = "test", value: str = "val"):
        start = time.time()
        cache.set(key, value)
        set_time = time.time() - start
        start = time.time()
        _ = cache.get(key)
        get_time = time.time() - start
        self.results["cache_set_time"] = set_time
        self.results["cache_get_time"] = get_time
        self.logger.info(f"Cache set: {set_time:.3f}s, get: {get_time:.3f}s")

    def generate_performance_profile(self) -> Dict[str, Any]:
        # Profiling mémoire et CPU
        tracemalloc.start()
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().used / (1024 * 1024)
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')[:5]
        self.results["cpu_usage"] = cpu
        self.results["memory_usage_mb"] = mem
        self.results["top_memory_stats"] = [(str(stat), stat.size / 1024) for stat in top_stats]
        tracemalloc.stop()
        self.logger.info(f"CPU: {cpu}%, MEM: {mem}MB")
        return self.results

    def identify_bottlenecks(self) -> List[str]:
        bottlenecks = []
        # Simple heuristique : temps > seuil
        for k, v in self.results.items():
            if isinstance(v, (int, float)) and v > 1.0:
                bottlenecks.append(k)
        self.logger.info(f"Bottlenecks: {bottlenecks}")
        return bottlenecks

if __name__ == "__main__":
    bench = PerformanceBenchmark()
    bench.benchmark_api_endpoints()
    # bench.benchmark_model_inference(model)  # À adapter selon votre modèle
    # bench.benchmark_database_queries(db_session)
    # bench.benchmark_cache_performance(cache)
    profile = bench.generate_performance_profile()
    bottlenecks = bench.identify_bottlenecks()
    print("Performance profile:", profile)
    print("Bottlenecks:", bottlenecks)
