from typing import Dict, Optional
from app.ai.pipeline.processor import StreamWorker

class StreamWorkerPool:
    """Manages concurrent CPU stream processing workers for multiple authorized cameras."""

    def __init__(self):
        self.workers: Dict[str, StreamWorker] = {}

    def get_or_create_worker(self, camera_id: str, source: str, location: str = "Authorized CCTV") -> StreamWorker:
        if camera_id in self.workers:
            existing = self.workers[camera_id]
            if source and existing.source != source:
                self.stop_worker(camera_id)
                worker = StreamWorker(source=source, camera_id=camera_id, camera_location=location)
                self.workers[camera_id] = worker
                return worker
            return existing

        worker = StreamWorker(source=source, camera_id=camera_id, camera_location=location)
        self.workers[camera_id] = worker
        return worker

    def stop_worker(self, camera_id: str):
        if camera_id in self.workers:
            self.workers[camera_id].stop()
            del self.workers[camera_id]

    def stop_all(self):
        for worker in self.workers.values():
            worker.stop()
        self.workers.clear()

worker_pool = StreamWorkerPool()

__all__ = ["StreamWorker", "StreamWorkerPool", "worker_pool"]
