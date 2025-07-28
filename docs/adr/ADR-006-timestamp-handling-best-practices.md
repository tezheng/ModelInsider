# ADR-006: Timestamp Handling Best Practices

## Decision Status: 🔄 SUPERSEDED - NEEDS REVISION

**Final Decision**: Store timestamps as UTC Unix epoch (float) internally, serialize only at system boundaries.

### Core Decision Points

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Storage Format** | UTC Unix epoch (float) | Minimal memory (8 bytes vs 48 bytes), fast arithmetic |
| **Timezone Handling** | Always UTC | No timezone confusion, eliminates DST bugs |
| **Serialization Strategy** | Only at boundaries | No early formatting, each writer chooses format |
| **Time Calculations** | Float subtraction | Simple arithmetic: `end_time - start_time` |
| **Data Locality** | Embedded in step data | Better cohesion, self-contained structures |

### Key Benefits

• **Always UTC** ✅ - No timezone confusion
• **Unix Epoch (float)** ✅ - Minimal memory (8 bytes vs 48 bytes), fast arithmetic  
• **No early formatting** ✅ - Serialize only at boundaries
• **Time elapsed calculations** ✅ - Simple float subtraction: `end_time - start_time`
• **Performance optimized** ✅ - Critical for export workflows
• **Data locality** ✅ - Timestamps embedded where they're used

**Implementation Status**: Current datetime implementation needs revision for performance optimization.

| Status | Date | Decision Maker(s) | Consulted | Informed |
|--------|------|-------------------|-----------|----------|
| Superseded | 2025-07-24 | Architecture Team | Dev Team | All Teams |

## Context and Problem Statement

Modern Python applications, especially distributed systems and data processing pipelines, require consistent and reliable timestamp handling for:

- Event tracking and audit trails
- Performance monitoring and analytics
- Data synchronization across systems
- Debugging and troubleshooting
- Compliance and regulatory requirements

Without standardized timestamp practices, systems suffer from timezone bugs, serialization inconsistencies, and difficulties in debugging time-related issues.

## Decision Drivers

- **Correctness**: Avoid timezone-related bugs and DST issues
- **Consistency**: Uniform timestamp handling across all components
- **Performance**: Efficient storage and processing
- **Testability**: Deterministic testing of time-dependent code
- **Interoperability**: Compatible with external systems and standards
- **Maintainability**: Clear patterns that developers can follow

## Considered Options

1. **Store timestamps as Unix epoch (float/int)**
2. **Store timestamps as ISO 8601 strings throughout**
3. **Store as timezone-aware datetime objects, serialize at boundaries**
4. **Use naive datetime objects with implicit timezone**
5. **Mixed approach based on use case**

## Decision Outcome

**Chosen option**: Option 1 - Store timestamps as UTC Unix epoch (float) internally, with serialization only at system boundaries.

### Rationale

- **Performance**: Minimal memory usage and fast arithmetic operations
- **Simplicity**: Direct time elapsed calculations with float subtraction
- **UTC Consistency**: Always UTC, no timezone confusion
- **Serialization Flexibility**: Each boundary chooses appropriate format
- **Export Optimization**: Critical for performance-sensitive export workflows

### Consequences

**Positive:**

- Minimal performance overhead and memory usage
- Simple time elapsed calculations (t2 - t1)
- Always UTC, eliminates timezone bugs
- Fast serialization to any format needed

**Negative:**

- Less human-readable in raw form (requires conversion to view)
- No built-in timezone information (must assume UTC)
- Potential precision loss for sub-second timing

**Neutral:**

- May require migration from existing datetime-based code
- Developers need to remember UTC assumption

## Implementation Notes

### 1. Core Timestamp Creation

```python
import time

# Always capture as UTC epoch float
def now_utc_timestamp() -> float:
    """Get current UTC timestamp as float."""
    return time.time()

# For step data classes
@dataclass
class ModelPrepData:
    model_class: str
    total_modules: int
    total_parameters: int
    timestamp: float = field(default_factory=time.time)  # UTC epoch

### 2. Time Elapsed Calculations
```python
# Simple float arithmetic for time differences
def calculate_duration(start_time: float, end_time: float) -> float:
    """Calculate duration in seconds."""
    return end_time - start_time

# Usage example
start = time.time()
# ... do work ...
end = time.time()
duration = calculate_duration(start, end)  # Simple subtraction!
```

### 3. Data Structure Pattern

```python
from dataclasses import dataclass, field
import time
import uuid

@dataclass
class EventData:
    """Example of embedding UTC epoch timestamps in domain objects."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    payload: dict = field(default_factory=dict)
    
    # Always UTC epoch floats
    created_at: float = field(default_factory=time.time)
    updated_at: float | None = None
    
    def update(self, **kwargs):
        """Update event and timestamp."""
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.updated_at = time.time()  # Simple UTC epoch update!
```

### 4. Serialization at Boundaries

```python
import time
from datetime import datetime, timezone
from typing import Protocol

class TimestampSerializer(Protocol):
    """Protocol for timestamp serialization from Unix epoch."""
    def serialize(self, epoch_time: float) -> str | int | float:
        ...

class ISOSerializer:
    """ISO 8601 format for APIs and external systems."""
    def serialize(self, epoch_time: float) -> str:
        # Convert epoch to ISO with milliseconds: 2024-01-24T15:30:45.123Z
        dt = datetime.fromtimestamp(epoch_time, tz=timezone.utc)
        return dt.isoformat(timespec='milliseconds').replace('+00:00', 'Z')

class UnixMillisSerializer:
    """Unix timestamp in milliseconds for JavaScript compatibility."""
    def serialize(self, epoch_time: float) -> int:
        # Milliseconds since epoch
        return int(epoch_time * 1000)

class HumanSerializer:
    """Human-readable format for logs and displays."""
    def serialize(self, epoch_time: float) -> str:
        # Convert to local time for display
        dt = datetime.fromtimestamp(epoch_time, tz=timezone.utc)
        local_dt = dt.astimezone()
        return local_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
```

### 4. Testing Time-Dependent Code

```python
from datetime import datetime, timezone
from freezegun import freeze_time
import pytest

class TimeProvider:
    """Injectable time source for testing."""
    def now(self) -> datetime:
        return datetime.now(timezone.utc)

# Production code
def process_event(data: dict, time_provider: TimeProvider = None):
    if time_provider is None:
        time_provider = TimeProvider()
    
    timestamp = time_provider.now()
    return {"data": data, "processed_at": timestamp}

# Test code
@freeze_time("2024-01-01 12:00:00+00:00")
def test_process_event():
    result = process_event({"value": 42})
    expected = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert result["processed_at"] == expected

# Mock time provider
class MockTimeProvider(TimeProvider):
    def __init__(self, fixed_time: datetime):
        self.fixed_time = fixed_time
    
    def now(self) -> datetime:
        return self.fixed_time
```

### 5. Performance Optimization

```python
from datetime import timezone
from zoneinfo import ZoneInfo

# Cache frequently used timezone objects
UTC = timezone.utc
EASTERN = ZoneInfo('America/New_York')
PACIFIC = ZoneInfo('America/Los_Angeles')

class TimestampCache:
    """Cache for performance-critical timestamp operations."""
    def __init__(self):
        self._tz_cache = {}
    
    def get_timezone(self, tz_name: str) -> ZoneInfo:
        """Get cached timezone object."""
        if tz_name not in self._tz_cache:
            self._tz_cache[tz_name] = ZoneInfo(tz_name)
        return self._tz_cache[tz_name]
```

## Implementation Evidence

**Current Status**: 🔄 **NEEDS REVISION** - Decision updated to use Unix epoch (float) for better performance.

### 🔄 **Previous Implementation (datetime objects)**

- **ExportData**: Added UUID session tracking and embedded timestamp access methods
- **Step Data Classes**: All step data classes include automatic datetime timestamp capture  
- **Writers**: Serialize timestamps at boundaries per format requirements
- **ExportMonitor**: Updated to use embedded timestamps, no separate tracking

### 🔄 **Required Changes for Unix Epoch Implementation**

- **Step Data Classes**: Change `timestamp: datetime` to `timestamp: float` with `time.time()` default
- **ExportData Methods**: Update timestamp arithmetic to use float operations
- **Writers**: Update serialization from `float` → target format instead of `datetime` → target format
- **Tests**: Update freezegun tests to work with `time.time()` instead of `datetime.now()`

### ✅ **Architectural Principles Established**

- **Data locality** with embedded timestamps in step data
- **Serialization at boundaries** design pattern
- **Single source of truth** for timestamp capture

## Validation/Confirmation

- ✅ **Unit Tests**: Use `freezegun` to verify timestamp capture
- ✅ **Integration Tests**: Verify chronological order of step timestamps  
- ✅ **Performance Tests**: Ensure minimal overhead from datetime objects
- ✅ **Output Validation**: Check each writer produces expected format

## Detailed Analysis of Options

### 🟢 Option 1: Unix Epoch (float) - **CHOSEN**

- **Description**: Store as UTC seconds since epoch (float)
- **Pros**:
  - **Minimal memory usage** (8 bytes per timestamp)
  - **Simple time elapsed calculations** (just subtraction)
  - **Fast serialization** to any format needed
  - **Always UTC** (eliminates timezone bugs)
  - **High precision** (sub-second timing supported)
- **Cons**:
  - Less human-readable in raw form
  - Must assume UTC convention
- **Technical Impact**: Optimal for performance-critical export workflows

### 🔴 Option 2: ISO Strings Throughout - **NOT RECOMMENDED**

- **Description**: Store timestamps as ISO 8601 strings internally
- **Pros**:
  - Already serialized
  - Human-readable
- **Cons**:
  - **String parsing required** for time calculations
  - **No type safety** for arithmetic operations
  - **Larger memory footprint** (~25 bytes vs 8 bytes)
  - **Poor performance** for duration calculations
- **Technical Impact**: Severely impacts performance, type-unsafe

### 🟡 Option 3: datetime Objects - **GOOD BUT NOT OPTIMAL**

- **Description**: Use Python's datetime with UTC timezone
- **Pros**:
  - Type-safe with IDE support
  - Rich datetime API
  - Timezone-aware
- **Cons**:
  - **Higher memory usage** (~48 bytes per timestamp)
  - **Slower arithmetic** compared to float operations
  - Requires serialization step
- **Technical Impact**: Good for general use, but performance overhead for exports

### 🔴 Option 4: Separate Timestamp Tracking - **NOT RECOMMENDED**

- **Description**: Track timestamps in separate dictionary/manager
- **Pros**:
  - Centralized timestamp management
  - Easy to add/remove steps
- **Cons**:
  - **Poor data locality** (timestamps separated from data)
  - **Synchronization complexity** between data and timestamps
  - **Harder to maintain** consistency
- **Technical Impact**: Violates cohesion principles, increases complexity

### 🟢 Option 5: Embedded Timestamps - **ARCHITECTURAL PRINCIPLE**

- **Description**: Include timestamp in each step's data class
- **Pros**:
  - **Better data cohesion** (timestamp with related data)
  - **Self-contained data structures**
  - **Easier testing and debugging**
  - **Natural data locality**
- **Cons**:
  - Requires updating all step data classes
  - Slightly more fields per class
- **Technical Impact**: Cleaner architecture, better maintainability

## Related Decisions

- ADR-001: Record Architecture Decisions
- ADR-002: Auxiliary Operations Tagging (uses timestamps)
- Export Monitor Architecture design document

## More Information

- [Python datetime Best Practices](https://docs.python.org/3/library/datetime.html#aware-and-naive-objects)
- [ISO 8601 Standard](https://www.iso.org/iso-8601-date-and-time-format.html)
- [PEP 615 - IANA Time Zone Database](https://peps.python.org/pep-0615/)

---
*Last updated: 2025-07-24*
*Next review: 2025-10-24*
