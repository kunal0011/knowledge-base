---
date: "2026-04-06"
type: lld-question
difficulty: easy
status: active
tags: [lld, interview-prep, traffic-light, state-pattern]
---

# Design a Traffic Light System

## 1. Problem Statement
Design a traffic light control system for an intersection with timed transitions, pedestrian crossings, and emergency override.

## 2. Key Implementation (Python)

```python
from enum import Enum
from abc import ABC, abstractmethod
import time, threading

class LightColor(Enum):
    RED = ("RED", 30)
    YELLOW = ("YELLOW", 5)
    GREEN = ("GREEN", 25)

class TrafficLightState(ABC):
    @abstractmethod
    def next(self, light: 'TrafficLight'): pass
    @abstractmethod
    def display(self) -> str: pass

class RedState(TrafficLightState):
    def next(self, light):
        print(f"  {light.direction}: 🔴 → 🟢")
        light.set_state(GreenState())
    def display(self): return "🔴 RED"

class GreenState(TrafficLightState):
    def next(self, light):
        print(f"  {light.direction}: 🟢 → 🟡")
        light.set_state(YellowState())
    def display(self): return "🟢 GREEN"

class YellowState(TrafficLightState):
    def next(self, light):
        print(f"  {light.direction}: 🟡 → 🔴")
        light.set_state(RedState())
    def display(self): return "🟡 YELLOW"

class TrafficLight:
    def __init__(self, direction: str, initial_state: TrafficLightState):
        self.direction = direction
        self._state = initial_state

    def set_state(self, state: TrafficLightState):
        self._state = state

    def transition(self):
        self._state.next(self)

    def display(self) -> str:
        return f"{self.direction}: {self._state.display()}"

class Intersection:
    def __init__(self):
        self.ns_light = TrafficLight("North-South", GreenState())
        self.ew_light = TrafficLight("East-West", RedState())
        self.emergency_mode = False

    def cycle(self):
        """One full cycle: NS green → yellow → red, then EW green → yellow → red"""
        if self.emergency_mode:
            print("🚨 Emergency: All RED")
            return

        print(f"\nCycle: {self.ns_light.display()} | {self.ew_light.display()}")
        
        # NS: Green → Yellow → Red
        self.ns_light.transition()  # Green → Yellow
        self.ns_light.transition()  # Yellow → Red

        # EW: Red → Green
        self.ew_light.transition()  # Red → Green

        print(f"After: {self.ns_light.display()} | {self.ew_light.display()}")

    def emergency_override(self):
        """All lights turn red for emergency vehicles"""
        self.emergency_mode = True
        self.ns_light.set_state(RedState())
        self.ew_light.set_state(RedState())
        print("🚨 EMERGENCY: All lights RED")

    def resume_normal(self):
        self.emergency_mode = False
        self.ns_light.set_state(GreenState())
        self.ew_light.set_state(RedState())
        print("✅ Normal operation resumed")
```

## 3. Patterns: **State** (light color transitions) | **Observer** (pedestrian → light coordination) | **Command** (emergency override)

## 4. Follow-ups
- **Pedestrian crossing?** Button triggers pedestrian phase insertion.
- **Adaptive timing?** Strategy — adjust durations based on traffic sensor data.
- **Multi-intersection coordination?** Mediator pattern for green wave optimization.

---

**Related:** [[13 - State Pattern]] | [[02 - Observer Pattern]]
