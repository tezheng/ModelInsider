"""
Base Monitor Class for System Monitoring

Provides abstract base class and common functionality for all system monitors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import asyncio
import threading
import time
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetricSample:
    """Single metric sample with timestamp"""
    timestamp: float  # Unix timestamp with microsecond precision
    metric_name: str
    value: Any
    unit: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "timestamp": self.timestamp,
            "metric_name": self.metric_name,
            "value": self.value,
            "unit": self.unit,
            "metadata": self.metadata
        }


@dataclass
class MonitorConfig:
    """Configuration for system monitors"""
    sampling_rate_hz: float = 10.0  # Sampling frequency in Hz
    buffer_size: int = 10000  # Number of samples to buffer
    enable_logging: bool = True
    enable_async: bool = False  # Use async collection
    warmup_samples: int = 10  # Samples to discard during warmup
    
    @property
    def sampling_interval(self) -> float:
        """Get sampling interval in seconds"""
        return 1.0 / self.sampling_rate_hz if self.sampling_rate_hz > 0 else 1.0


class BaseMonitor(ABC):
    """Abstract base class for all system monitors"""
    
    def __init__(self, config: Optional[MonitorConfig] = None):
        """
        Initialize base monitor
        
        Args:
            config: Monitor configuration
        """
        self.config = config or MonitorConfig()
        self.is_monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._async_task: Optional[asyncio.Task] = None
        
        # Metric storage
        self.samples = deque(maxlen=self.config.buffer_size)
        self.callbacks: List[Callable[[MetricSample], None]] = []
        
        # Performance tracking
        self._collection_times = deque(maxlen=100)
        self._last_collection_time = 0.0
        
        # Initialize monitor-specific resources
        self._initialize()
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize monitor-specific resources"""
        pass
    
    @abstractmethod
    def collect_metrics(self) -> List[MetricSample]:
        """
        Collect current metrics
        
        Returns:
            List of metric samples
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup monitor resources"""
        pass
    
    def start_monitoring(self) -> None:
        """Start continuous monitoring"""
        if self.is_monitoring:
            logger.warning("Monitor already running")
            return
        
        self.is_monitoring = True
        
        if self.config.enable_async:
            # Async monitoring (for integration with async frameworks)
            loop = asyncio.get_event_loop()
            self._async_task = loop.create_task(self._async_monitor_loop())
        else:
            # Thread-based monitoring (default)
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name=f"{self.__class__.__name__}-Thread"
            )
            self._monitor_thread.start()
        
        logger.info(f"{self.__class__.__name__} started (Rate: {self.config.sampling_rate_hz}Hz)")
    
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
        
        if self._async_task:
            self._async_task.cancel()
            self._async_task = None
        
        logger.info(f"{self.__class__.__name__} stopped")
    
    def _monitor_loop(self) -> None:
        """Synchronous monitoring loop"""
        warmup_counter = 0
        
        while self.is_monitoring:
            start_time = time.perf_counter()
            
            try:
                # Collect metrics
                samples = self.collect_metrics()
                
                # Skip warmup samples
                if warmup_counter < self.config.warmup_samples:
                    warmup_counter += 1
                else:
                    # Store samples
                    for sample in samples:
                        self.samples.append(sample)
                        
                        # Notify callbacks
                        for callback in self.callbacks:
                            try:
                                callback(sample)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")
                
                # Track collection time
                collection_time = time.perf_counter() - start_time
                self._collection_times.append(collection_time)
                
                # Sleep for remainder of interval
                sleep_time = self.config.sampling_interval - collection_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif collection_time > self.config.sampling_interval * 1.5:
                    logger.warning(
                        f"Collection time ({collection_time:.3f}s) exceeds interval "
                        f"({self.config.sampling_interval:.3f}s)"
                    )
            
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(self.config.sampling_interval)
    
    async def _async_monitor_loop(self) -> None:
        """Asynchronous monitoring loop"""
        warmup_counter = 0
        
        while self.is_monitoring:
            start_time = time.perf_counter()
            
            try:
                # Collect metrics
                samples = self.collect_metrics()
                
                # Skip warmup samples
                if warmup_counter < self.config.warmup_samples:
                    warmup_counter += 1
                else:
                    # Store samples
                    for sample in samples:
                        self.samples.append(sample)
                        
                        # Notify callbacks
                        for callback in self.callbacks:
                            try:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(sample)
                                else:
                                    callback(sample)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")
                
                # Track collection time
                collection_time = time.perf_counter() - start_time
                self._collection_times.append(collection_time)
                
                # Sleep for remainder of interval
                sleep_time = self.config.sampling_interval - collection_time
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            except Exception as e:
                logger.error(f"Error in async monitor loop: {e}")
                await asyncio.sleep(self.config.sampling_interval)
    
    def register_callback(self, callback: Callable[[MetricSample], None]) -> None:
        """Register a callback for metric samples"""
        self.callbacks.append(callback)
    
    def get_recent_samples(self, count: int = 100) -> List[MetricSample]:
        """Get recent metric samples"""
        return list(self.samples)[-count:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        if not self._collection_times:
            return {}
        
        collection_times = list(self._collection_times)
        return {
            "samples_collected": len(self.samples),
            "avg_collection_time_ms": sum(collection_times) / len(collection_times) * 1000,
            "max_collection_time_ms": max(collection_times) * 1000,
            "min_collection_time_ms": min(collection_times) * 1000,
            "overhead_percentage": (sum(collection_times) / len(collection_times)) / self.config.sampling_interval * 100
        }
    
    def __enter__(self):
        """Context manager entry"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_monitoring()
        self.cleanup()