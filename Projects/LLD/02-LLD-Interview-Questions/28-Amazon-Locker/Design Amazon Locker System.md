---
date: "2026-04-06"
type: lld-question
difficulty: medium
status: active
tags: [lld, interview-prep, amazon-locker, state-pattern]
---

# Design Amazon Locker System

## 1. Problem Statement
Design a package locker system with locker assignment, OTP-based pickup, size-based allocation, and expiration.

## 2. Key Implementation (Python)

```python
from enum import Enum
from typing import Dict, Optional, List
import uuid, random, string
from datetime import datetime, timedelta

class LockerSize(Enum):
    SMALL = 1
    MEDIUM = 2
    LARGE = 3

class LockerStatus(Enum):
    AVAILABLE = "AVAILABLE"
    OCCUPIED = "OCCUPIED"
    EXPIRED = "EXPIRED"

class PackageSize(Enum):
    SMALL = 1
    MEDIUM = 2
    LARGE = 3

SIZE_COMPATIBILITY = {
    PackageSize.SMALL: [LockerSize.SMALL, LockerSize.MEDIUM, LockerSize.LARGE],
    PackageSize.MEDIUM: [LockerSize.MEDIUM, LockerSize.LARGE],
    PackageSize.LARGE: [LockerSize.LARGE],
}

class Locker:
    def __init__(self, locker_id: str, size: LockerSize):
        self.locker_id = locker_id
        self.size = size
        self.status = LockerStatus.AVAILABLE
        self.package_id: Optional[str] = None
        self.otp: Optional[str] = None
        self.expiry: Optional[datetime] = None

class Package:
    def __init__(self, package_id: str, size: PackageSize, recipient_id: str):
        self.package_id = package_id
        self.size = size
        self.recipient_id = recipient_id

class LockerSystem:
    EXPIRY_HOURS = 72

    def __init__(self):
        self.lockers: Dict[str, Locker] = {}
        self.packages: Dict[str, str] = {}  # package_id → locker_id

    def add_locker(self, locker_id: str, size: LockerSize):
        self.lockers[locker_id] = Locker(locker_id, size)

    def assign_locker(self, package: Package) -> Optional[str]:
        compatible = SIZE_COMPATIBILITY[package.size]
        for locker in self.lockers.values():
            if locker.status == LockerStatus.AVAILABLE and locker.size in compatible:
                locker.status = LockerStatus.OCCUPIED
                locker.package_id = package.package_id
                locker.otp = ''.join(random.choices(string.digits, k=6))
                locker.expiry = datetime.now() + timedelta(hours=self.EXPIRY_HOURS)
                self.packages[package.package_id] = locker.locker_id
                print(f"📦 Package {package.package_id} → Locker {locker.locker_id} | OTP: {locker.otp}")
                return locker.otp
        print("❌ No available locker for this package size")
        return None

    def pickup(self, locker_id: str, otp: str) -> bool:
        locker = self.lockers.get(locker_id)
        if not locker or locker.status != LockerStatus.OCCUPIED:
            return False
        if locker.otp != otp:
            print("❌ Invalid OTP")
            return False
        pkg_id = locker.package_id
        locker.status = LockerStatus.AVAILABLE
        locker.package_id = None
        locker.otp = None
        locker.expiry = None
        self.packages.pop(pkg_id, None)
        print(f"✅ Package {pkg_id} picked up from locker {locker_id}")
        return True

    def check_expired(self):
        now = datetime.now()
        for locker in self.lockers.values():
            if locker.status == LockerStatus.OCCUPIED and locker.expiry and now > locker.expiry:
                locker.status = LockerStatus.EXPIRED
                print(f"⏰ Locker {locker.locker_id} expired. Return package to warehouse.")
```

## 3. Patterns: **State** (locker lifecycle) | **Strategy** (locker allocation) | **Observer** (notify customer)

## 4. Follow-ups
- **Returns?** Reverse flow — customer scans return label, locker assigned for drop-off.
- **Notifications?** Observer pattern — send OTP via email/SMS.
- **Multiple locations?** `LockerLocation` with address and available lockers.

---

**Related:** [[13 - State Pattern]] | [[01 - Strategy Pattern]]
