---
date: "2026-04-06"
type: lld-question
difficulty: medium
status: active
tags:
  - lld
  - interview-prep
  - elevator-system
  - state-pattern
---

# Design an Elevator System

## 1. Problem Statement

Design an elevator system for a building with multiple elevators and multiple floors. The system should efficiently handle pickup requests and move elevators optimally.

---

## 2. Requirements Clarification

### Functional Requirements

| # | Requirement |
|---|-------------|
| FR1 | Handle up/down requests from any floor |
| FR2 | Handle destination requests from inside elevator |
| FR3 | Support multiple elevators |
| FR4 | Open/close doors at destination |
| FR5 | Display current floor and direction |

### Non-Functional Requirements

| # | Requirement |
|---|-------------|
| NFR1 | Minimize wait time |
| NFR2 | Efficient dispatching (don't send all elevators to one floor) |
| NFR3 | Thread-safe |

---

## 3. Class Design

```mermaid
classDiagram
    class ElevatorSystem {
        -List~Elevator~ elevators
        -ElevatorScheduler scheduler
        +requestElevator(floor, direction)
        +status() List~ElevatorStatus~
    }

    class Elevator {
        -int id
        -int currentFloor
        -Direction direction
        -State state
        -Set~int~ destinationFloors
        +addDestination(floor)
        +move()
        +openDoors()
        +closeDoors()
    }

    class ElevatorScheduler {
        <<interface>>
        +selectElevator(elevators, floor, direction)* Elevator
    }

    class FCFSScheduler {
        +selectElevator() Elevator
    }

    class SCANScheduler {
        +selectElevator() Elevator
    }

    class Direction {
        <<enumeration>>
        UP
        DOWN
        IDLE
    }

    class ElevatorState {
        <<enumeration>>
        MOVING
        STOPPED
        DOOR_OPEN
        MAINTENANCE
    }

    ElevatorSystem *-- Elevator
    ElevatorSystem --> ElevatorScheduler
    ElevatorScheduler <|.. FCFSScheduler
    ElevatorScheduler <|.. SCANScheduler
    Elevator --> Direction
    Elevator --> ElevatorState
```

### Sequence Flow

```mermaid
sequenceDiagram
    autonumber
    actor Passenger
    participant Button as FloorButton
    participant System as ElevatorSystem
    participant Scheduler as ElevatorScheduler
    participant Elevator as ElevatorCar
    participant Door as ElevatorDoor

    Passenger->>Button: Press Floor Button (floor=5, direction=UP)
    Button->>System: requestElevator(floor=5, UP)
    System->>Scheduler: selectElevator(elevators, 5, UP)
    Scheduler-->>System: Selected Elevator 1
    System->>Elevator: addDestination(5)
    loop Elevator Step Cycle
        Elevator->>Elevator: moveTowardsDestination()
    end
    Elevator->>Door: openDoor()
    Door-->>Passenger: Doors Open (Ding!)
    Passenger->>Elevator: Step Inside & Select Floor 9
    Elevator->>Elevator: addDestination(9)
    Elevator->>Door: closeDoor()
```


---

## 4. Design Patterns Used

| Pattern | Where | Why |
|---------|-------|-----|
| **State** | `ElevatorState` (MOVING, STOPPED, DOOR_OPEN) | Behavior changes per state |
| **Strategy** | `ElevatorScheduler` | Swap scheduling algorithms |
| **Observer** | Floor displays | Notify displays of elevator position |

→ See: [[13 - State Pattern]] | [[01 - Strategy Pattern]] | [[02 - Observer Pattern]]

---

## 5. Key Implementation

### Python

```python
from enum import Enum
from typing import List, Set, Optional
import threading
import heapq


class Direction(Enum):
    UP = 1
    DOWN = -1
    IDLE = 0


class ElevatorState(Enum):
    MOVING = "MOVING"
    STOPPED = "STOPPED"
    DOOR_OPEN = "DOOR_OPEN"
    MAINTENANCE = "MAINTENANCE"


class Elevator:
    def __init__(self, elevator_id: int, min_floor: int = 0, max_floor: int = 10):
        self.id = elevator_id
        self.current_floor = 0
        self.direction = Direction.IDLE
        self.state = ElevatorState.STOPPED
        self.min_floor = min_floor
        self.max_floor = max_floor
        self._up_stops: Set[int] = set()
        self._down_stops: Set[int] = set()
        self._lock = threading.Lock()

    def add_destination(self, floor: int):
        with self._lock:
            if floor > self.current_floor:
                self._up_stops.add(floor)
            elif floor < self.current_floor:
                self._down_stops.add(floor)
            else:
                self._open_doors()

    def move(self):
        """Process one step of movement"""
        with self._lock:
            if self.direction == Direction.UP:
                if self._up_stops:
                    self.current_floor += 1
                    if self.current_floor in self._up_stops:
                        self._up_stops.remove(self.current_floor)
                        self._open_doors()
                    if not self._up_stops:
                        self.direction = Direction.DOWN if self._down_stops else Direction.IDLE
                else:
                    self.direction = Direction.DOWN if self._down_stops else Direction.IDLE

            elif self.direction == Direction.DOWN:
                if self._down_stops:
                    self.current_floor -= 1
                    if self.current_floor in self._down_stops:
                        self._down_stops.remove(self.current_floor)
                        self._open_doors()
                    if not self._down_stops:
                        self.direction = Direction.UP if self._up_stops else Direction.IDLE
                else:
                    self.direction = Direction.UP if self._up_stops else Direction.IDLE

            elif self.direction == Direction.IDLE:
                if self._up_stops:
                    self.direction = Direction.UP
                elif self._down_stops:
                    self.direction = Direction.DOWN

    def _open_doors(self):
        self.state = ElevatorState.DOOR_OPEN
        print(f"  Elevator {self.id}: Doors open at floor {self.current_floor}")
        self.state = ElevatorState.STOPPED

    def is_idle(self) -> bool:
        return self.direction == Direction.IDLE

    def distance_to(self, floor: int) -> int:
        return abs(self.current_floor - floor)

    def __repr__(self):
        return (f"Elevator(id={self.id}, floor={self.current_floor}, "
                f"dir={self.direction.name}, up={self._up_stops}, "
                f"down={self._down_stops})")


# ──── Strategy: Scheduling ────

class ElevatorScheduler:
    """Strategy interface for elevator dispatch"""
    def select_elevator(self, elevators: List[Elevator],
                        floor: int, direction: Direction) -> Optional[Elevator]:
        raise NotImplementedError


class NearestElevatorScheduler(ElevatorScheduler):
    """Select the nearest idle or same-direction elevator"""
    def select_elevator(self, elevators: List[Elevator],
                        floor: int, direction: Direction) -> Optional[Elevator]:
        best = None
        best_distance = float('inf')

        for elevator in elevators:
            if elevator.state == ElevatorState.MAINTENANCE:
                continue

            distance = elevator.distance_to(floor)

            # Prefer idle elevators
            if elevator.is_idle() and distance < best_distance:
                best = elevator
                best_distance = distance
            # Or elevators heading toward requested floor
            elif (elevator.direction == direction and
                  ((direction == Direction.UP and elevator.current_floor <= floor) or
                   (direction == Direction.DOWN and elevator.current_floor >= floor))):
                if distance < best_distance:
                    best = elevator
                    best_distance = distance

        # Fallback: any elevator with smallest distance
        if best is None:
            available = [e for e in elevators
                        if e.state != ElevatorState.MAINTENANCE]
            if available:
                best = min(available, key=lambda e: e.distance_to(floor))

        return best


# ──── Controller ────

class ElevatorSystem:
    def __init__(self, num_elevators: int, num_floors: int,
                 scheduler: ElevatorScheduler = None):
        self.elevators = [Elevator(i, 0, num_floors) for i in range(num_elevators)]
        self.scheduler = scheduler or NearestElevatorScheduler()
        self.num_floors = num_floors

    def request_elevator(self, floor: int, direction: Direction):
        elevator = self.scheduler.select_elevator(
            self.elevators, floor, direction
        )
        if elevator:
            elevator.add_destination(floor)
            print(f"📞 Floor {floor} {direction.name} → Elevator {elevator.id} dispatched")
        else:
            print(f"❌ No elevator available for floor {floor}")

    def select_floor(self, elevator_id: int, floor: int):
        """Passenger inside elevator selects a floor"""
        self.elevators[elevator_id].add_destination(floor)
        print(f"🔘 Elevator {elevator_id}: Floor {floor} selected")

    def step(self):
        """Advance all elevators by one step"""
        for elevator in self.elevators:
            elevator.move()

    def status(self):
        for e in self.elevators:
            print(f"  {e}")


# ──── Usage ────

if __name__ == "__main__":
    system = ElevatorSystem(num_elevators=3, num_floors=10)

    system.request_elevator(5, Direction.UP)
    system.request_elevator(3, Direction.DOWN)
    system.request_elevator(7, Direction.DOWN)

    for _ in range(8):
        system.step()

    print("\nStatus:")
    system.status()
```

### Java

```java
package com.lld.elevator;

import java.util.*;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.concurrent.locks.ReentrantLock;

enum Direction {
    UP, DOWN, IDLE
}

enum ElevatorState {
    MOVING, STOPPED, DOOR_OPEN, MAINTENANCE
}

class Elevator {
    private final int id;
    private int currentFloor;
    private Direction direction;
    private ElevatorState state;
    private final int minFloor;
    private final int maxFloor;
    private final ConcurrentSkipListSet<Integer> upStops = new ConcurrentSkipListSet<>();
    private final ConcurrentSkipListSet<Integer> downStops = new ConcurrentSkipListSet<>(Collections.reverseOrder());
    private final ReentrantLock lock = new ReentrantLock();

    public Elevator(int id, int minFloor, int maxFloor) {
        this.id = id;
        this.minFloor = minFloor;
        this.maxFloor = maxFloor;
        this.currentFloor = minFloor;
        this.direction = Direction.IDLE;
        this.state = ElevatorState.STOPPED;
    }

    public void addDestination(int floor) {
        lock.lock();
        try {
            if (floor > currentFloor) {
                upStops.add(floor);
                if (direction == Direction.IDLE) direction = Direction.UP;
            } else if (floor < currentFloor) {
                downStops.add(floor);
                if (direction == Direction.IDLE) direction = Direction.DOWN;
            } else {
                openDoors();
            }
        } finally {
            lock.unlock();
        }
    }

    public void move() {
        lock.lock();
        try {
            if (direction == Direction.UP) {
                if (!upStops.isEmpty()) {
                    currentFloor++;
                    if (upStops.contains(currentFloor)) {
                        upStops.remove(currentFloor);
                        openDoors();
                    }
                } else if (!downStops.isEmpty()) {
                    direction = Direction.DOWN;
                } else {
                    direction = Direction.IDLE;
                    state = ElevatorState.STOPPED;
                }
            } else if (direction == Direction.DOWN) {
                if (!downStops.isEmpty()) {
                    currentFloor--;
                    if (downStops.contains(currentFloor)) {
                        downStops.remove(currentFloor);
                        openDoors();
                    }
                } else if (!upStops.isEmpty()) {
                    direction = Direction.UP;
                } else {
                    direction = Direction.IDLE;
                    state = ElevatorState.STOPPED;
                }
            }
        } finally {
            lock.unlock();
        }
    }

    private void openDoors() {
        state = ElevatorState.DOOR_OPEN;
        System.out.printf("Elevator %d stopped at floor %d. Doors opening.%n", id, currentFloor);
        state = ElevatorState.STOPPED;
    }

    public int getId() { return id; }
    public int getCurrentFloor() { return currentFloor; }
    public Direction getDirection() { return direction; }
    public ElevatorState getState() { return state; }
}

interface ElevatorScheduler {
    Elevator selectElevator(List<Elevator> elevators, int floor, Direction direction);
}

class NearestElevatorScheduler implements ElevatorScheduler {
    @Override
    public Elevator selectElevator(List<Elevator> elevators, int floor, Direction direction) {
        Elevator best = null;
        int minDistance = Integer.MAX_VALUE;

        for (Elevator e : elevators) {
            int dist = Math.abs(e.getCurrentFloor() - floor);
            if (e.getDirection() == Direction.IDLE) {
                dist += 0;
            } else if (e.getDirection() == direction &&
                    ((direction == Direction.UP && e.getCurrentFloor() <= floor) ||
                     (direction == Direction.DOWN && e.getCurrentFloor() >= floor))) {
                dist += 1;
            } else {
                dist += 10;
            }
            if (dist < minDistance) {
                minDistance = dist;
                best = e;
            }
        }
        return best;
    }
}

public class ElevatorSystem {
    private final List<Elevator> elevators;
    private final ElevatorScheduler scheduler;
    private final int numFloors;

    public ElevatorSystem(int numElevators, int numFloors, ElevatorScheduler scheduler) {
        this.numFloors = numFloors;
        this.scheduler = scheduler != null ? scheduler : new NearestElevatorScheduler();
        this.elevators = new ArrayList<>();
        for (int i = 0; i < numElevators; i++) {
            this.elevators.add(new Elevator(i, 0, numFloors));
        }
    }

    public void requestElevator(int floor, Direction direction) {
        Elevator chosen = scheduler.selectElevator(elevators, floor, direction);
        if (chosen != null) {
            chosen.addDestination(floor);
            System.out.printf("Floor %d %s dispatched to Elevator %d%n", floor, direction, chosen.getId());
        }
    }

    public void selectFloor(int elevatorId, int floor) {
        elevators.get(elevatorId).addDestination(floor);
    }

    public void step() {
        for (Elevator elevator : elevators) {
            elevator.move();
        }
    }
}
```


---

## 6. Scheduling Algorithms

| Algorithm | How It Works | Pros | Cons |
|-----------|-------------|------|------|
| **FCFS** | First come, first served | Simple | Inefficient |
| **Nearest** | Dispatch closest elevator | Low wait time | May starve far floors |
| **SCAN (Elevator)** | Move in one direction, reverse at end | Fair, like disk scheduling | Less optimal for sparse requests |
| **LOOK** | Like SCAN but reverses when no more requests in current direction | Better than SCAN | More complex |

---


---

## Thread Safety Considerations

| Concern | Solution |
|---|---|
| Concurrent destination requests | `ReentrantLock` guards `addDestination()` and `move()` |
| Destination ordering | `ConcurrentSkipListSet` maintains sorted unique stop requests |
| Floor button spamming | Idempotent set additions prevent duplicate stop scheduling |
| Multi-car starvation | Strategy scheduler isolates elevator state evaluation |

## Extensibility & SOLID Principles

| Principle | Architectural Implementation |
|---|---|
| **S** — Single Responsibility | `Elevator` manages physical kinematics; `ElevatorScheduler` governs dispatching heuristics |
| **O** — Open/Closed | Pluggable dispatchers (`FCFS`, `LOOK`, `SCAN`, `Zoned`) via `ElevatorScheduler` interface |
| **L** — Liskov Substitution | Any scheduler implementation seamlessly substitutes into `ElevatorSystem` |
| **I** — Interface Segregation | `ElevatorScheduler` interface exposes purely `selectElevator()` |
| **D** — Dependency Inversion | `ElevatorSystem` depends on `ElevatorScheduler` abstraction, not concrete implementations |

---

## 7. Follow-up Questions

**Q: How to handle peak hours (morning rush)?**
A: Pre-position elevators at lobby. Use zone-based allocation (elevators serve specific floor ranges).

**Q: How to handle emergency mode?**
A: All elevators go to ground floor, doors open. `EmergencyState` overrides all behavior.

**Q: How to handle VIP/express elevators?**
A: Separate elevator pool with restricted floor access. Priority queue for VIP requests.

---

**Related:** [[13 - State Pattern]] | [[01 - Strategy Pattern]] | [[02 - Observer Pattern]]
