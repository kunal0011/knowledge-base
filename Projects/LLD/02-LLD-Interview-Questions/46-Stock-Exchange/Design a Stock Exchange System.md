---
date: "2026-04-06"
type: lld-question
difficulty: hard
tags: [lld, interview-prep, stock-exchange, observer-pattern, strategy-pattern]
---

# Design a Stock Exchange System

## 1. Problem Statement
Design a stock trading system with order placement (market/limit), order matching engine, and real-time price updates.

## 2. Key Implementation (Python)

```python
from enum import Enum
from typing import Dict, List, Optional
from collections import defaultdict
import heapq, uuid
from datetime import datetime

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    OPEN = "OPEN"; PARTIAL = "PARTIAL"; FILLED = "FILLED"; CANCELLED = "CANCELLED"

class Order:
    def __init__(self, trader_id: str, symbol: str, side: OrderSide,
                 order_type: OrderType, quantity: int, price: float = 0):
        self.order_id = str(uuid.uuid4())[:8]
        self.trader_id = trader_id; self.symbol = symbol
        self.side = side; self.order_type = order_type
        self.quantity = quantity; self.remaining = quantity
        self.price = price; self.status = OrderStatus.OPEN
        self.timestamp = datetime.now()

class Trade:
    def __init__(self, buy_order: Order, sell_order: Order, qty: int, price: float):
        self.trade_id = str(uuid.uuid4())[:8]
        self.buy_order_id = buy_order.order_id
        self.sell_order_id = sell_order.order_id
        self.quantity = qty; self.price = price
        self.timestamp = datetime.now()

class OrderBook:
    """Per-symbol order book with price-time priority matching"""
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.buy_orders = []   # max-heap (negate price)
        self.sell_orders = []  # min-heap
        self.trades: List[Trade] = []

    def add_order(self, order: Order):
        if order.side == OrderSide.BUY:
            heapq.heappush(self.buy_orders, (-order.price, order.timestamp, order))
        else:
            heapq.heappush(self.sell_orders, (order.price, order.timestamp, order))
        self._match()

    def _match(self):
        while self.buy_orders and self.sell_orders:
            best_buy = self.buy_orders[0][2]
            best_sell = self.sell_orders[0][2]

            if -self.buy_orders[0][0] < self.sell_orders[0][0]:
                break  # No match possible

            trade_qty = min(best_buy.remaining, best_sell.remaining)
            trade_price = best_sell.price  # Price-time priority

            trade = Trade(best_buy, best_sell, trade_qty, trade_price)
            self.trades.append(trade)
            print(f"⚡ Trade: {trade_qty} {self.symbol} @ ${trade_price:.2f}")

            best_buy.remaining -= trade_qty
            best_sell.remaining -= trade_qty

            if best_buy.remaining == 0:
                best_buy.status = OrderStatus.FILLED
                heapq.heappop(self.buy_orders)
            else:
                best_buy.status = OrderStatus.PARTIAL

            if best_sell.remaining == 0:
                best_sell.status = OrderStatus.FILLED
                heapq.heappop(self.sell_orders)
            else:
                best_sell.status = OrderStatus.PARTIAL

class StockExchange:
    def __init__(self):
        self.order_books: Dict[str, OrderBook] = {}

    def place_order(self, order: Order):
        if order.symbol not in self.order_books:
            self.order_books[order.symbol] = OrderBook(order.symbol)
        self.order_books[order.symbol].add_order(order)

    def get_order_book(self, symbol: str) -> Optional[OrderBook]:
        return self.order_books.get(symbol)

if __name__ == "__main__":
    exchange = StockExchange()
    exchange.place_order(Order("trader1", "AAPL", OrderSide.BUY, OrderType.LIMIT, 100, 150.00))
    exchange.place_order(Order("trader2", "AAPL", OrderSide.SELL, OrderType.LIMIT, 50, 149.50))
    exchange.place_order(Order("trader3", "AAPL", OrderSide.SELL, OrderType.LIMIT, 50, 150.00))
```

## 3. Key Concept: **Order Matching Engine** uses price-time priority with two heaps (buy max-heap, sell min-heap). Match when best buy ≥ best sell.

## 4. Patterns: **Observer** (price updates, trade notifications) | **Strategy** (matching algorithms) | **Command** (cancel/modify orders)
## 5. Follow-ups: **Stop-loss?** Triggered orders that become market orders | **Dark pools?** Hidden order matching | **Market data feed?** Pub-sub for real-time quotes.

---
**Related:** [[02 - Observer Pattern]] | [[01 - Strategy Pattern]] | [[07 - Command Pattern]]
