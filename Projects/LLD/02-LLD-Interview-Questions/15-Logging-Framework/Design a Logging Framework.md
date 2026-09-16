---
date: "2026-04-06"
type: lld-question
difficulty: medium
status: active
tags: [lld, interview-prep, logging, singleton, decorator]
---

# Design a Logging Framework

## 1. Problem Statement
Design a logging framework like Log4j supporting multiple log levels, output destinations, and formatters.

## 2. Class Design

```mermaid
classDiagram
    class Logger {
        -String name
        -LogLevel level
        -List~Handler~ handlers
        +debug(msg)
        +info(msg)
        +error(msg)
        +getLogger(name)$ Logger
    }
    class Handler {
        <<abstract>>
        #Formatter formatter
        +handle(record)*
    }
    class ConsoleHandler
    class FileHandler
    class Formatter {
        <<interface>>
        +format(record)* String
    }
    class SimpleFormatter
    class JSONFormatter

    Logger --> Handler
    Handler <|-- ConsoleHandler
    Handler <|-- FileHandler
    Handler --> Formatter
    Formatter <|.. SimpleFormatter
    Formatter <|.. JSONFormatter
```

### Sequence Flow

```mermaid
sequenceDiagram
    autonumber
    actor App as Application Code
    participant Logger as Logger (Singleton)
    participant Filter as LevelFilter
    participant Formatter as LogFormatter
    participant Appender as Appender (Console / File)

    App->>Logger: log(LogLevel.ERROR, "Database Timeout")
    Logger->>Filter: shouldLog(LogLevel.ERROR)
    Filter-->>Logger: true (Level >= Threshold)
    Logger->>Formatter: format("Database Timeout", timestamp, thread)
    Formatter-->>Logger: "[2026-09-16 10:00:00] [ERROR] Database Timeout"
    Logger->>Appender: append(formattedMessage)
    Appender-->>App: Message written to Console & S3 File
```


## 3. Key Implementation (Python)

```python
from enum import IntEnum
from abc import ABC, abstractmethod
from datetime import datetime
import threading, json

class LogLevel(IntEnum):
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40
    FATAL = 50

class LogRecord:
    def __init__(self, level: LogLevel, message: str, logger_name: str):
        self.level = level
        self.message = message
        self.logger_name = logger_name
        self.timestamp = datetime.now()
        self.thread = threading.current_thread().name

class Formatter(ABC):
    @abstractmethod
    def format(self, record: LogRecord) -> str: pass

class SimpleFormatter(Formatter):
    def format(self, r):
        return f"[{r.timestamp:%H:%M:%S}] {r.level.name:5} {r.logger_name}: {r.message}"

class JSONFormatter(Formatter):
    def format(self, r):
        return json.dumps({"ts": str(r.timestamp), "level": r.level.name,
                          "logger": r.logger_name, "msg": r.message})

class Handler(ABC):
    def __init__(self, level: LogLevel = LogLevel.DEBUG, formatter: Formatter = None):
        self.level = level
        self.formatter = formatter or SimpleFormatter()

    def handle(self, record: LogRecord):
        if record.level >= self.level:
            self._emit(self.formatter.format(record))

    @abstractmethod
    def _emit(self, formatted: str): pass

class ConsoleHandler(Handler):
    def _emit(self, s): print(s)

class FileHandler(Handler):
    def __init__(self, filename: str, **kwargs):
        super().__init__(**kwargs)
        self.filename = filename
    def _emit(self, s):
        with open(self.filename, 'a') as f:
            f.write(s + '\n')

class Logger:
    _instances = {}
    _lock = threading.Lock()

    def __init__(self, name: str, level: LogLevel = LogLevel.DEBUG):
        self.name = name
        self.level = level
        self.handlers = []

    @classmethod
    def get_logger(cls, name: str) -> 'Logger':
        with cls._lock:
            if name not in cls._instances:
                cls._instances[name] = Logger(name)
            return cls._instances[name]

    def add_handler(self, h: Handler):
        self.handlers.append(h)

    def log(self, level: LogLevel, msg: str):
        if level >= self.level:
            record = LogRecord(level, msg, self.name)
            for h in self.handlers:
                h.handle(record)

    def debug(self, msg): self.log(LogLevel.DEBUG, msg)
    def info(self, msg): self.log(LogLevel.INFO, msg)
    def error(self, msg): self.log(LogLevel.ERROR, msg)
```

### Java

```java
package com.lld.logging;

import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;

enum LogLevel { DEBUG, INFO, WARN, ERROR }

interface LogAppender {
    void write(String message);
}

class ConsoleAppender implements LogAppender {
    @Override
    public void write(String message) { System.out.println("[CONSOLE] " + message); }
}

class FileAppender implements LogAppender {
    @Override
    public void write(String message) { System.out.println("[FILE] " + message); }
}

public class Logger {
    private static volatile Logger instance;
    private LogLevel minimumLevel = LogLevel.INFO;
    private final List<LogAppender> appenders = new CopyOnWriteArrayList<>();

    private Logger() {
        appenders.add(new ConsoleAppender());
    }

    public static Logger getInstance() {
        if (instance == null) {
            synchronized (Logger.class) {
                if (instance == null) {
                    instance = new Logger();
                }
            }
        }
        return instance;
    }

    public void addAppender(LogAppender appender) { appenders.add(appender); }
    public void setMinimumLevel(LogLevel level) { this.minimumLevel = level; }

    public void log(LogLevel level, String message) {
        if (level.ordinal() < minimumLevel.ordinal()) return;
        String formatted = String.format("[%s] [%s] %s", LocalDateTime.now(), level, message);
        for (LogAppender appender : appenders) {
            appender.write(formatted);
        }
    }
}
```


## 4. Patterns: **Singleton** (Logger.getLogger) | **Strategy** (formatters) | **Chain of Responsibility** (handler chain)


---

## Thread Safety Considerations

| Concern | Solution |
|---|---|
| Singleton Initialization | Double-Checked Locking with `volatile` prevents instruction reordering race |
| Concurrent logging | `CopyOnWriteArrayList` permits lock-free log dispatching across multi-threaded applications |

## Extensibility & SOLID Principles

| Principle | Architectural Implementation |
|---|---|
| **S** — Single Responsibility | `Logger` controls log flow; `LogAppender` handles output sink IO |
| **O** — Open/Closed | Remote sinks (Elasticsearch, Kafka) plug in by implementing `LogAppender` |
| **D** — Dependency Inversion | Logger depends on the `LogAppender` interface, not concrete file or console sinks |

---

## 5. Follow-ups
- **Async logging?** Background thread with queue for non-blocking I/O.
- **Log rotation?** RotatingFileHandler with max size & backup count.
- **Structured logging?** JSON formatter with context fields (MDC).

---

**Related:** [[06 - Singleton Pattern]] | [[01 - Strategy Pattern]] | [[03 - Decorator Pattern]]
