---
date: "2026-04-06"
type: lld-question
difficulty: medium
status: active
tags: [lld, interview-prep, inventory-management, observer-pattern]
---

# Design an Inventory Management System

## 1. Problem Statement
Design an inventory system for a warehouse/e-commerce platform tracking products, stock levels, orders, and restocking alerts.

## 2. Key Implementation (Python)

```python
from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod

class StockAlert(ABC):
    @abstractmethod
    def on_low_stock(self, product_id: str, current: int, threshold: int): pass

class EmailAlert(StockAlert):
    def on_low_stock(self, product_id, current, threshold):
        print(f"📧 LOW STOCK ALERT: {product_id} has {current} units (threshold: {threshold})")

class Product:
    def __init__(self, pid: str, name: str, price: float, reorder_threshold: int = 10):
        self.pid = pid
        self.name = name
        self.price = price
        self.reorder_threshold = reorder_threshold

class InventoryItem:
    def __init__(self, product: Product, quantity: int = 0):
        self.product = product
        self.quantity = quantity
        self.last_restocked = datetime.now()

class InventoryService:
    def __init__(self):
        self._inventory: Dict[str, InventoryItem] = {}
        self._alerts: List[StockAlert] = []

    def add_alert(self, alert: StockAlert):
        self._alerts.append(alert)

    def add_product(self, product: Product, initial_qty: int = 0):
        self._inventory[product.pid] = InventoryItem(product, initial_qty)

    def restock(self, product_id: str, quantity: int):
        item = self._inventory.get(product_id)
        if item:
            item.quantity += quantity
            item.last_restocked = datetime.now()
            print(f"📦 Restocked {item.product.name}: +{quantity} → {item.quantity}")

    def sell(self, product_id: str, quantity: int) -> bool:
        item = self._inventory.get(product_id)
        if not item or item.quantity < quantity:
            print(f"❌ Insufficient stock for {product_id}")
            return False
        item.quantity -= quantity
        print(f"🛒 Sold {quantity}x {item.product.name} → {item.quantity} remaining")

        if item.quantity <= item.product.reorder_threshold:
            for alert in self._alerts:
                alert.on_low_stock(product_id, item.quantity, item.product.reorder_threshold)
        return True

    def get_stock(self, product_id: str) -> int:
        item = self._inventory.get(product_id)
        return item.quantity if item else 0

    def get_low_stock_report(self) -> List[dict]:
        return [{"product": item.product.name, "qty": item.quantity,
                 "threshold": item.product.reorder_threshold}
                for item in self._inventory.values()
                if item.quantity <= item.product.reorder_threshold]
```

## 3. Patterns: **Observer** (low stock alerts) | **Strategy** (reorder policies) | **Singleton** (inventory service)

## 4. Follow-ups
- **Multi-warehouse?** Each warehouse has its own inventory, transfer between warehouses.
- **Batch tracking?** Track batches with expiry dates (FIFO for perishables).
- **Reservation?** Temporary stock hold for cart items with TTL.

---

**Related:** [[02 - Observer Pattern]] | [[01 - Strategy Pattern]]
