---
date: "2026-04-06"
type: lld-question
difficulty: easy
status: active
tags:
  - lld
  - interview-prep
  - vending-machine
  - state-pattern
---

# Design a Vending Machine

## 1. Problem Statement

Design a vending machine that accepts coins, allows product selection, dispenses products, and returns change. The machine has different states (idle, has money, dispensing, out of stock).

---

## 2. Requirements

| # | Requirement |
|---|-------------|
| FR1 | Accept coins (1¢, 5¢, 10¢, 25¢) and bills |
| FR2 | Display products with prices |
| FR3 | Select a product |
| FR4 | Dispense product if sufficient payment |
| FR5 | Return change |
| FR6 | Handle out-of-stock and insufficient funds |

---

## 3. Class Design

```mermaid
classDiagram
    class VendingMachine {
        -State currentState
        -Inventory inventory
        -double currentBalance
        +insertMoney(amount)
        +selectProduct(code)
        +dispense()
        +returnChange()
    }

    class State {
        <<interface>>
        +insertMoney(machine, amount)*
        +selectProduct(machine, code)*
        +dispense(machine)*
        +returnChange(machine)*
    }

    class IdleState
    class HasMoneyState
    class DispensingState

    class Product {
        -String code
        -String name
        -double price
        -int quantity
    }

    State <|.. IdleState
    State <|.. HasMoneyState
    State <|.. DispensingState
    VendingMachine --> State
    VendingMachine --> Product
```

### Sequence Flow

```mermaid
sequenceDiagram
    autonumber
    actor Customer
    participant VM as VendingMachine
    participant State as MachineState
    participant Inv as Inventory
    participant Coin as CoinDispenser

    Customer->>VM: insertMoney(amount=$2.00)
    VM->>State: insertMoney(amount)
    State->>VM: balance += 2.00, setState(HasMoneyState)
    VM-->>Customer: Balance: $2.00
    Customer->>VM: selectProduct("A1")
    VM->>State: selectProduct("A1")
    State->>Inv: checkProduct("A1")
    Inv-->>State: Product(price=$1.50, qty=5)
    State->>VM: setState(DispensingState)
    VM->>State: dispense()
    State->>Inv: deductQuantity("A1")
    State->>VM: balance -= 1.50
    State->>Coin: returnChange(0.50)
    Coin-->>Customer: Dispense Change ($0.50)
    State-->>Customer: Dispense Product (A1)
    State->>VM: setState(IdleState)
```


---

## 4. Key Implementation (Python)

```python
from enum import Enum
from abc import ABC, abstractmethod
from typing import Dict, Optional


class Product:
    def __init__(self, code: str, name: str, price: float, quantity: int):
        self.code = code
        self.name = name
        self.price = price
        self.quantity = quantity


class State(ABC):
    @abstractmethod
    def insert_money(self, machine: 'VendingMachine', amount: float): pass
    @abstractmethod
    def select_product(self, machine: 'VendingMachine', code: str): pass
    @abstractmethod
    def dispense(self, machine: 'VendingMachine'): pass
    @abstractmethod
    def return_change(self, machine: 'VendingMachine'): pass


class IdleState(State):
    def insert_money(self, machine, amount):
        machine.balance += amount
        print(f"💰 Inserted ${amount:.2f}. Balance: ${machine.balance:.2f}")
        machine.set_state(machine.has_money_state)

    def select_product(self, machine, code):
        print("⚠️ Please insert money first")

    def dispense(self, machine):
        print("⚠️ Please insert money and select a product")

    def return_change(self, machine):
        print("No money to return")


class HasMoneyState(State):
    def insert_money(self, machine, amount):
        machine.balance += amount
        print(f"💰 Added ${amount:.2f}. Balance: ${machine.balance:.2f}")

    def select_product(self, machine, code):
        product = machine.inventory.get(code)
        if not product:
            print(f"❌ Product {code} not found")
            return
        if product.quantity == 0:
            print(f"❌ {product.name} is out of stock")
            return
        if machine.balance < product.price:
            print(f"⚠️ Insufficient funds. Need ${product.price:.2f}, "
                  f"have ${machine.balance:.2f}")
            return

        machine.selected_product = product
        machine.set_state(machine.dispensing_state)
        machine.dispense()

    def dispense(self, machine):
        print("⚠️ Please select a product first")

    def return_change(self, machine):
        change = machine.balance
        machine.balance = 0
        print(f"💵 Returning change: ${change:.2f}")
        machine.set_state(machine.idle_state)


class DispensingState(State):
    def insert_money(self, machine, amount):
        print("⚠️ Please wait, dispensing product")

    def select_product(self, machine, code):
        print("⚠️ Please wait, dispensing product")

    def dispense(self, machine):
        product = machine.selected_product
        product.quantity -= 1
        machine.balance -= product.price
        print(f"🎉 Dispensed: {product.name}")

        if machine.balance > 0:
            print(f"💵 Change: ${machine.balance:.2f}")
            machine.balance = 0

        machine.selected_product = None
        machine.set_state(machine.idle_state)

    def return_change(self, machine):
        print("⚠️ Already dispensing")


class VendingMachine:
    def __init__(self):
        self.idle_state = IdleState()
        self.has_money_state = HasMoneyState()
        self.dispensing_state = DispensingState()
        self._state: State = self.idle_state
        self.balance: float = 0.0
        self.selected_product: Optional[Product] = None
        self.inventory: Dict[str, Product] = {}

    def set_state(self, state: State):
        self._state = state

    def add_product(self, product: Product):
        self.inventory[product.code] = product

    def insert_money(self, amount: float):
        self._state.insert_money(self, amount)

    def select_product(self, code: str):
        self._state.select_product(self, code)

    def dispense(self):
        self._state.dispense(self)

    def return_change(self):
        self._state.return_change(self)


if __name__ == "__main__":
    vm = VendingMachine()
    vm.add_product(Product("A1", "Cola", 1.50, 5))
    vm.add_product(Product("A2", "Chips", 1.00, 3))
    vm.add_product(Product("B1", "Water", 0.75, 10))

    # Normal flow
    vm.insert_money(2.00)
    vm.select_product("A1")  # Dispenses Cola, returns $0.50 change

    # Insufficient funds
    vm.insert_money(0.50)
    vm.select_product("A1")  # Not enough money
    vm.insert_money(1.00)
    vm.select_product("A1")  # Now it works
```

### Java

```java
package com.lld.vendingmachine;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantLock;

class Product {
    private final String code;
    private final String name;
    private final double price;
    private int quantity;

    public Product(String code, String name, double price, int quantity) {
        this.code = code;
        this.name = name;
        this.price = price;
        this.quantity = quantity;
    }

    public synchronized boolean isAvailable() { return quantity > 0; }
    public synchronized void decrement() { if (quantity > 0) quantity--; }
    public String getCode() { return code; }
    public String getName() { return name; }
    public double getPrice() { return price; }
    public synchronized int getQuantity() { return quantity; }
}

interface State {
    void insertMoney(VendingMachine machine, double amount);
    void selectProduct(VendingMachine machine, String code);
    void dispense(VendingMachine machine);
    void returnChange(VendingMachine machine);
}

class IdleState implements State {
    @Override
    public void insertMoney(VendingMachine machine, double amount) {
        machine.addBalance(amount);
        System.out.printf("Inserted $%.2f. Current balance: $%.2f%n", amount, machine.getBalance());
        machine.setState(machine.getHasMoneyState());
    }

    @Override
    public void selectProduct(VendingMachine machine, String code) {
        System.out.println("Please insert money first.");
    }

    @Override
    public void dispense(VendingMachine machine) {
        System.out.println("No product selected.");
    }

    @Override
    public void returnChange(VendingMachine machine) {
        System.out.println("No money to return.");
    }
}

class HasMoneyState implements State {
    @Override
    public void insertMoney(VendingMachine machine, double amount) {
        machine.addBalance(amount);
        System.out.printf("Added $%.2f. Current balance: $%.2f%n", amount, machine.getBalance());
    }

    @Override
    public void selectProduct(VendingMachine machine, String code) {
        Product p = machine.getProduct(code);
        if (p == null) {
            System.out.println("Invalid product code.");
            return;
        }
        if (!p.isAvailable()) {
            System.out.println("Product out of stock.");
            return;
        }
        if (machine.getBalance() < p.getPrice()) {
            System.out.printf("Insufficient balance. Need $%.2f more.%n", p.getPrice() - machine.getBalance());
            return;
        }
        machine.setSelectedProduct(p);
        machine.setState(machine.getDispensingState());
        machine.dispense();
    }

    @Override
    public void dispense(VendingMachine machine) {
        System.out.println("Select a product first.");
    }

    @Override
    public void returnChange(VendingMachine machine) {
        double change = machine.resetBalance();
        System.out.printf("Refunded $%.2f.%n", change);
        machine.setState(machine.getIdleState());
    }
}

class DispensingState implements State {
    @Override
    public void insertMoney(VendingMachine machine, double amount) {
        System.out.println("Please wait, dispensing in progress.");
    }

    @Override
    public void selectProduct(VendingMachine machine, String code) {
        System.out.println("Already dispensing.");
    }

    @Override
    public void dispense(VendingMachine machine) {
        Product p = machine.getSelectedProduct();
        p.decrement();
        machine.deductBalance(p.getPrice());
        System.out.printf("Dispensing %s!%n", p.getName());

        if (machine.getBalance() > 0) {
            double change = machine.resetBalance();
            System.out.printf("Returning change: $%.2f%n", change);
        }
        machine.setSelectedProduct(null);
        machine.setState(machine.getIdleState());
    }

    @Override
    public void returnChange(VendingMachine machine) {
        System.out.println("Cannot return change during dispense.");
    }
}

public class VendingMachine {
    private final State idleState = new IdleState();
    private final State hasMoneyState = new HasMoneyState();
    private final State dispensingState = new DispensingState();
    private State currentState = idleState;

    private final Map<String, Product> inventory = new ConcurrentHashMap<>();
    private double balance = 0.0;
    private Product selectedProduct;
    private final ReentrantLock lock = new ReentrantLock();

    public void addProduct(Product p) {
        inventory.put(p.getCode(), p);
    }

    public void insertMoney(double amount) {
        lock.lock();
        try { currentState.insertMoney(this, amount); }
        finally { lock.unlock(); }
    }

    public void selectProduct(String code) {
        lock.lock();
        try { currentState.selectProduct(this, code); }
        finally { lock.unlock(); }
    }

    public void dispense() {
        lock.lock();
        try { currentState.dispense(this); }
        finally { lock.unlock(); }
    }

    public void returnChange() {
        lock.lock();
        try { currentState.returnChange(this); }
        finally { lock.unlock(); }
    }

    // State & Balance Helpers
    public void setState(State state) { this.currentState = state; }
    public State getIdleState() { return idleState; }
    public State getHasMoneyState() { return hasMoneyState; }
    public State getDispensingState() { return dispensingState; }
    public Product getProduct(String code) { return inventory.get(code); }
    public Product getSelectedProduct() { return selectedProduct; }
    public void setSelectedProduct(Product p) { this.selectedProduct = p; }
    public double getBalance() { return balance; }
    public void addBalance(double amt) { this.balance += amt; }
    public void deductBalance(double amt) { this.balance -= amt; }
    public double resetBalance() { double prev = balance; balance = 0.0; return prev; }
}
```


---

## 5. Design Patterns

| Pattern | Usage |
|---------|-------|
| **State** | Machine behavior changes per state (Idle → HasMoney → Dispensing) |
| **Strategy** | Could be used for different payment methods |

→ See: [[13 - State Pattern]]

---


---

## Thread Safety Considerations

| Concern | Solution |
|---|---|
| Concurrent money insertion & selection | `ReentrantLock` synchronizes high-level state transitions across threads |
| Race on item inventory | `Product.decrement()` is synchronized to prevent double-dispensing out-of-stock items |
| Balance corruption | Dedicated mutation helpers (`addBalance`, `resetBalance`) under lock guard |

## Extensibility & SOLID Principles

| Principle | Architectural Implementation |
|---|---|
| **S** — Single Responsibility | Each `State` handles exact behavior for its lifecycle phase; `VendingMachine` acts as context |
| **O** — Open/Closed | New states (`OutOfOrderState`, `MaintenanceState`) added without modifying existing state classes |
| **L** — Liskov Substitution | All state instances adhere strictly to the `State` contract |
| **I** — Interface Segregation | `State` interface defines only the 4 canonical user interactions |
| **D** — Dependency Inversion | Context depends on the `State` abstraction, not concrete states |

---

## 6. Follow-up Questions

**Q: How to handle card payments?** Add `PaymentStrategy` interface with `CashPayment` and `CardPayment` implementations.

**Q: How to handle refill/restocking?** Add `AdminState` accessible with a key. Admin can add products and withdraw cash.

**Q: How to handle multiple items per transaction?** Maintain a cart. Dispense all items after final confirmation.

---

**Related:** [[13 - State Pattern]] | [[01 - Strategy Pattern]]
