import os
import platform
import psutil
import time
import torch

try:
    import ultralytics
    ULTRALYTICS_VERSION = ultralytics.__version__
except ImportError:
    ULTRALYTICS_VERSION = "N/A"

class HardwareProfile:
    """Provides detailed inspection and real-time monitoring of CPU resources."""

    @staticmethod
    def get_cpu_info():
        # Get processor name safely across Windows/Linux
        cpu_name = platform.processor() or platform.machine()
        if platform.system() == "Windows":
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                cpu_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                winreg.CloseKey(key)
            except Exception:
                pass

        physical_cores = psutil.cpu_count(logical=False) or 1
        logical_cores = psutil.cpu_count(logical=True) or 1
        ram = psutil.virtual_memory()

        return {
            "cpu_model": cpu_name.strip(),
            "physical_cores": physical_cores,
            "logical_cores": logical_cores,
            "ram_total_gb": round(ram.total / (1024 ** 3), 2),
            "ram_available_gb": round(ram.available / (1024 ** 3), 2),
            "ram_percent": ram.percent,
            "os": f"{platform.system()} {platform.release()} ({platform.architecture()[0]})",
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "ultralytics_version": ULTRALYTICS_VERSION,
            "cuda_available": False,  # Strict CPU Only
            "device": "cpu"
        }

    @staticmethod
    def print_cpu_training_banner(script_name: str = "Training"):
        info = HardwareProfile.get_cpu_info()
        print("=" * 65)
        print(f"  {script_name.upper()} - CPU-ONLY COMPUTATIONAL PROFILE")
        print("=" * 65)
        print(f"  CPU Model:           {info['cpu_model']}")
        print(f"  CPU Cores:           {info['physical_cores']} Physical / {info['logical_cores']} Logical")
        print(f"  Total RAM:           {info['ram_total_gb']} GB (Available: {info['ram_available_gb']} GB)")
        print(f"  Operating System:    {info['os']}")
        print(f"  Python Version:      {info['python_version']}")
        print(f"  PyTorch Version:     {info['torch_version']}")
        print(f"  Ultralytics Version: {info['ultralytics_version']}")
        print("-" * 65)
        print("  Training device:     CPU (Strict enforcement)")
        print("=" * 65)

    @staticmethod
    def get_runtime_metrics():
        """Returns instantaneous CPU and memory metrics for live dashboard telemetry."""
        process = psutil.Process(os.getpid())
        ram = psutil.virtual_memory()
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "process_cpu_percent": process.cpu_percent(interval=None),
            "ram_used_gb": round((ram.total - ram.available) / (1024 ** 3), 2),
            "ram_total_gb": round(ram.total / (1024 ** 3), 2),
            "ram_percent": ram.percent,
            "process_memory_mb": round(process.memory_info().rss / (1024 ** 2), 2),
            "device": "cpu"
        }


class RealTimeProfiler:
    """Tracks latency, FPS, and frame throughput."""
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = []
        self.total_frames = 0
        self.start_time = time.time()

    def tick(self, duration_sec: float):
        self.frame_times.append(duration_sec)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
        self.total_frames += 1

    @property
    def current_fps(self) -> float:
        if not self.frame_times:
            return 0.0
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return round(1.0 / avg_time, 1) if avg_time > 0 else 0.0

    @property
    def current_latency_ms(self) -> float:
        if not self.frame_times:
            return 0.0
        return round((sum(self.frame_times) / len(self.frame_times)) * 1000.0, 1)

    def get_summary(self):
        metrics = HardwareProfile.get_runtime_metrics()
        metrics.update({
            "fps": self.current_fps,
            "latency_ms": self.current_latency_ms,
            "total_frames_processed": self.total_frames,
            "elapsed_seconds": round(time.time() - self.start_time, 1)
        })
        return metrics
