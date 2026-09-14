---
date: "2026-04-06"
type: lld-question
difficulty: medium
tags: [lld, interview-prep, amazon-order, state-pattern]
---

# Design an Order Management System

## 1. Problem Statement
Design an OMS for an e-commerce platform tracking orders through their full lifecycle with state transitions.

## 2. Key Implementation (Python)

```python
from enum import Enum
from datetime import datetime
from typing import List, Dict, Optional
import uuid

class OrderStatus(Enum):
    CREATED = "CREATED"
    PAYMENT_PENDING = "PAYMENT_PENDING"
    CONFIRMED = "CONFIRMED"
    PROCESSING = "PROCESSING"
    SHIPPED = "SHIPPED"
    DELIVERED = "DELIVERED"
    CANCELLED = "CANCELLED"
    RETURNED = "RETURNED"

# Valid transitions
TRANSITIONS = {
    OrderStatus.CREATED: [OrderStatus.PAYMENT_PENDING, OrderStatus.CANCELLED],
    OrderStatus.PAYMENT_PENDING: [OrderStatus.CONFIRMED, OrderStatus.CANCELLED],
    OrderStatus.CONFIRMED: [OrderStatus.PROCESSING, OrderStatus.CANCELLED],
    OrderStatus.PROCESSING: [OrderStatus.SHIPPED, OrderStatus.CANCELLED],
    OrderStatus.SHIPPED: [OrderStatus.DELIVERED],
    OrderStatus.DELIVERED: [OrderStatus.RETURNED],
    OrderStatus.CANCELLED: [],
    OrderStatus.RETURNED: [],
}

class OrderItem:
    def __init__(self, product_id: str, name: str, price: float, qty: int):
        self.product_id = product_id; self.name = name
        self.price = price; self.quantity = qty

class Order:
    def __init__(self, customer_id: str, items: List[OrderItem]):
        self.order_id = str(uuid.uuid4())[:8]
        self.customer_id = customer_id
        self.items = items
        self.total = sum(i.price * i.quantity for i in items)
        self.status = OrderStatus.CREATED
        self.status_history: List[tuple] = [(OrderStatus.CREATED, datetime.now())]
        self.tracking_id: Optional[str] = None

    def transition(self, new_status: OrderStatus) -> bool:
        if new_status in TRANSITIONS.get(self.status, []):
            self.status = new_status
            self.status_history.append((new_status, datetime.now()))
            print(f"📦 Order {self.order_id}: {self.status.value}")
            return True
        print(f"❌ Cannot transition from {self.status.value} to {new_status.value}")
        return False

class OrderService:
    def __init__(self):
        self.orders: Dict[str, Order] = {}

    def create_order(self, customer_id: str, items: List[OrderItem]) -> Order:
        order = Order(customer_id, items)
        self.orders[order.order_id] = order
        print(f"✅ Order {order.order_id} created: ${order.total:.2f}")
        return order

    def process_payment(self, order_id: str) -> bool:
        order = self.orders[order_id]
        order.transition(OrderStatus.PAYMENT_PENDING)
        # Simulate payment success
        return order.transition(OrderStatus.CONFIRMED)

    def ship_order(self, order_id: str, tracking_id: str):
        order = self.orders[order_id]
        order.transition(OrderStatus.PROCESSING)
        order.tracking_id = tracking_id
        order.transition(OrderStatus.SHIPPED)

    def get_order_history(self, order_id: str) -> List[tuple]:
        return self.orders[order_id].status_history
```

## 3. Patterns: **State** (order lifecycle) | **Observer** (status change notifications) | **Command** (reversible actions like returns)
## 4. Follow-ups: **Partial refunds?** Track per-item status | **Split shipments?** Order → multiple Shipments | **Event sourcing?** Store all state changes as events.

---
**Related:** [[13 - State Pattern]] | [[07 - Command Pattern]] | [[02 - Observer Pattern]]
