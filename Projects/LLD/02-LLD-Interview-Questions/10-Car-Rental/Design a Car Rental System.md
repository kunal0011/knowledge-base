---
date: "2026-04-06"
type: lld-question
difficulty: medium
status: active
tags: [lld, interview-prep, car-rental, strategy-pattern]
---

# Design a Car Rental System

## 1. Problem Statement
Design a car rental system supporting vehicle browsing, reservation, pickup, return, and billing with different vehicle types and pricing strategies.

## 2. Class Design

```mermaid
classDiagram
    class RentalSystem {
        -List~Store~ stores
        +searchVehicles(location, type) List~Vehicle~
        +reserve(customerId, vin, dates) Reservation
    }
    class Vehicle {
        <<abstract>>
        -String vin
        -String make
        -String model
        -VehicleType type
        -VehicleStatus status
        -double dailyRate
    }
    class Car
    class SUV
    class Truck
    class Reservation {
        -String id
        -Customer customer
        -Vehicle vehicle
        -DateRange dates
        -ReservationStatus status
        +pickup()
        +returnVehicle() double
    }
    class BillingStrategy {
        <<interface>>
        +calculate(reservation)* double
    }
    class DailyBilling
    class WeeklyBilling

    Vehicle <|-- Car
    Vehicle <|-- SUV
    Vehicle <|-- Truck
    Reservation --> Vehicle
    RentalSystem --> BillingStrategy
    BillingStrategy <|.. DailyBilling
    BillingStrategy <|.. WeeklyBilling
```

### Sequence Flow

```mermaid
sequenceDiagram
    autonumber
    actor Customer
    participant Rental as RentalService
    participant Fleet as VehicleFleet
    participant Decorator as AddonDecorator
    participant Payment as PaymentProcessor

    Customer->>Rental: searchVehicles(type=SUV, dates)
    Rental->>Fleet: findAvailable(SUV, dates)
    Fleet-->>Rental: List of matching SUVs
    Rental-->>Customer: Show Vehicle Options
    Customer->>Rental: selectVehicle(carId="C-101", addons=[GPS, ChildSeat])
    Rental->>Decorator: calculateTotal(baseRate=$80, addons)
    Decorator-->>Rental: Total Daily Rate: $105
    Rental->>Payment: authorizeDeposit(deposit=$200)
    Payment-->>Rental: Deposit Authorized
    Rental->>Fleet: markReserved("C-101", dates)
    Rental-->>Customer: Reservation Confirmed (Rental Agreement)
```


## 3. Key Implementation (Python)

```python
from enum import Enum
from datetime import date
from typing import List, Optional
import uuid

class VehicleType(Enum):
    CAR = "CAR"
    SUV = "SUV"
    TRUCK = "TRUCK"

class VehicleStatus(Enum):
    AVAILABLE = "AVAILABLE"
    RESERVED = "RESERVED"
    RENTED = "RENTED"
    MAINTENANCE = "MAINTENANCE"

class Vehicle:
    def __init__(self, vin: str, make: str, model: str,
                 v_type: VehicleType, daily_rate: float):
        self.vin = vin
        self.make = make
        self.model = model
        self.vehicle_type = v_type
        self.daily_rate = daily_rate
        self.status = VehicleStatus.AVAILABLE

class Reservation:
    def __init__(self, customer_id: str, vehicle: Vehicle,
                 pickup: date, return_date: date):
        self.id = str(uuid.uuid4())[:8]
        self.customer_id = customer_id
        self.vehicle = vehicle
        self.pickup_date = pickup
        self.return_date = return_date
        self.vehicle.status = VehicleStatus.RESERVED

    def pickup(self):
        self.vehicle.status = VehicleStatus.RENTED
        print(f"✅ Picked up {self.vehicle.make} {self.vehicle.model}")

    def return_vehicle(self) -> float:
        self.vehicle.status = VehicleStatus.AVAILABLE
        days = max(1, (self.return_date - self.pickup_date).days)
        total = days * self.vehicle.daily_rate
        print(f"🚗 Returned. Bill: ${total:.2f}")
        return total

class RentalStore:
    def __init__(self, location: str):
        self.location = location
        self.vehicles: List[Vehicle] = []

    def search(self, v_type: Optional[VehicleType] = None) -> List[Vehicle]:
        return [v for v in self.vehicles
                if v.status == VehicleStatus.AVAILABLE and
                (v_type is None or v.vehicle_type == v_type)]

    def reserve(self, customer_id: str, vin: str,
                pickup: date, return_date: date) -> Optional[Reservation]:
        vehicle = next((v for v in self.vehicles
                       if v.vin == vin and v.status == VehicleStatus.AVAILABLE), None)
        if not vehicle:
            return None
        return Reservation(customer_id, vehicle, pickup, return_date)
```

### Java

```java
package com.lld.carrental;

import java.time.LocalDate;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantLock;

enum VehicleType { SEDAN, SUV, TRUCK, VAN }
enum VehicleStatus { AVAILABLE, RESERVED, RENTED, MAINTENANCE }

class Vehicle {
    private final String vehicleId;
    private final String licensePlate;
    private final VehicleType type;
    private final double baseDailyRate;
    private VehicleStatus status = VehicleStatus.AVAILABLE;

    public Vehicle(String vehicleId, String licensePlate, VehicleType type, double baseDailyRate) {
        this.vehicleId = vehicleId;
        this.licensePlate = licensePlate;
        this.type = type;
        this.baseDailyRate = baseDailyRate;
    }

    public synchronized boolean isAvailable() { return status == VehicleStatus.AVAILABLE; }
    public synchronized void setStatus(VehicleStatus status) { this.status = status; }
    public String getVehicleId() { return vehicleId; }
    public double getBaseDailyRate() { return baseDailyRate; }
    public VehicleType getType() { return type; }
}

public class CarRentalSystem {
    private final Map<String, Vehicle> fleet = new ConcurrentHashMap<>();
    private final ReentrantLock lock = new ReentrantLock();

    public void addVehicle(Vehicle v) { fleet.put(v.getVehicleId(), v); }

    public synchronized Vehicle reserveVehicle(String vehicleId, String customerId, LocalDate start, LocalDate end) {
        Vehicle v = fleet.get(vehicleId);
        if (v == null || !v.isAvailable()) {
            throw new IllegalStateException("Vehicle not available for reservation");
        }
        v.setStatus(VehicleStatus.RESERVED);
        return v;
    }
}
```


## 4. Design Patterns
| Pattern | Usage |
|---------|-------|
| **Strategy** | `BillingStrategy` — daily, weekly, membership pricing |
| **Factory** | Vehicle creation based on type |
| **Decorator** | Add-ons: insurance, GPS, child seat |


---

## Thread Safety Considerations

| Concern | Solution |
|---|---|
| Concurrent fleet reservation | `reserveVehicle()` synchronized to prevent multiple customers claiming identical car |
| State transitions | Status changes (`AVAILABLE` $	o$ `RESERVED` $	o$ `RENTED`) guarded atomically |

## Extensibility & SOLID Principles

| Principle | Architectural Implementation |
|---|---|
| **S** — Single Responsibility | `Vehicle` manages mechanical specs; `RentalService` handles reservation schedules; `AddonDecorator` handles billable add-ons |
| **O** — Open/Closed | Insurance, GPS, Roadside assistance added dynamically via Decorator pattern without altering `Vehicle` |
| **D** — Dependency Inversion | Pricing and payment engines depend on high-level strategy interfaces |

---

## 5. Follow-up Questions
- **Insurance add-ons?** Decorator pattern — wrap rental with insurance layer.
- **Late return fees?** Extra days × penalty rate in billing strategy.
- **Multi-location drop-off?** Track locations, add surcharge for different return location.
- **Damage assessment?** Add `VehicleInspection` class with pre/post rental comparison.

---

**Related:** [[01 - Strategy Pattern]] | [[03 - Decorator Pattern]]
