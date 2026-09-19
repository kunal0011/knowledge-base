# Production Android & Flutter Development Master Curriculum

> "Modern mobile engineering requires understanding the complete stack: from the language type systems and compiler pipelines (Kotlin K2, Dart AOT/JIT), down through runtime memory architectures (ART Generational Concurrent Copying GC vs Dart VM Scavenger/Old Generation), rendering engines (Jetpack Compose Slot Tables vs Flutter Impeller Metal/Vulkan pipelines), to OS kernels (Linux Binder, Zygote, StrongBox TEEs)."

---

## 🏛️ Two Dedicated Autonomous Tracks

To ensure deep, publication-grade coverage without cross-polluting language paradigms or mixing idioms, this curriculum is partitioned into two comprehensive, independent tracks:

```
Projects/Production-Android-and-Flutter-Development/
├── README.md                                                 # Master Curriculum Portal
│
├── Part 1 - Deep Kotlin & Android Internals/                 # Complete Native Android Track
│   ├── README.md                                             # Part 1 Overview & Syllabus
│   ├── 01. Kotlin Language In-Depth & Type System Internals.md
│   ├── 02. Kotlin Coroutines & Structured Concurrency Under the Hood.md
│   ├── 03. Android Runtime (ART), Memory Management & Process Lifecycle.md
│   ├── 04. Jetpack Compose Compiler, Runtime & Layout Architecture.md
│   ├── 05. Modern Android Application Architecture (MVI, UDF & Clean Architecture).md
│   ├── 06. Compile-Time Dependency Injection with Dagger & Hilt.md
│   ├── 07. Enterprise Data Layer, Room SQLite Engine & Offline-First Sync.md
│   ├── 08. Resilient Networking, Protocol Buffers & OkHttp Engine.md
│   ├── 09. WorkManager, IPC (Binder/AIDL) & Background Services.md
│   ├── 10. Native NDK, JNI Interop & C++ Integration.md
│   ├── 11. Security Hardening, Hardware Keystore, StrongBox & Cryptography.md
│   └── 12. Profiling, Memory Leaks, Baseline Profiles & R8 Optimization.md
│
└── Part 2 - Deep Dart & Flutter Internals/                   # Complete Cross-Platform Flutter Track
    ├── README.md                                             # Part 2 Overview & Syllabus
    ├── 01. Dart 3+ Language In-Depth, Type System & Memory Model.md
    ├── 02. Dart Concurrency, Event Loops, Zones & Isolates Architecture.md
    ├── 03. Flutter Engine Architecture, C++ Runner & Impeller Pipeline.md
    ├── 04. The Four Trees of Flutter, Element Lifecycle & Layout Protocol.md
    ├── 05. Enterprise State Management (BLoC 8+, Riverpod 2.0 & Signals).md
    ├── 06. Reactive SQLite (Drift), Embedded NoSQL (Isar) & Offline Sync.md
    ├── 07. Production Networking, Dio Engine, Codec Serialization & SSL Pinning.md
    ├── 08. Platform Channels Architecture, BinaryMessenger & Codecs.md
    ├── 09. Dart FFI (Foreign Function Interface) & High-Performance C-Interop.md
    ├── 10. Background Execution, Workmanager & Native OS Services.md
    ├── 11. Security Hardening, Obfuscation & Binary Reverse Engineering Defense.md
    └── 12. Profiling, Frame Budgets, Jank Elimination & CI-CD Pipelines.md
```

---

## 🧭 Navigating the Curriculum

### [Track 1: Deep Kotlin & Android Internals](Part%201%20-%20Deep%20Kotlin%20&%20Android%20Internals/README.md)
Anchored in the official Kotlin 2.0 language specification, *Kotlin in Action (2nd Ed)*, *Effective Kotlin*, *Kotlin Coroutines: Deep Dive*, *Android Internals (Jonathan Levin)*, and modern Android 15 AOSP architecture. Covers everything from K2 compiler IR transformations to the Linux `/dev/binder` driver and StrongBox Keymaster.

### [Track 2: Deep Dart & Flutter Internals](Part%202%20-%20Deep%20Dart%20&%20Flutter%20Internals/README.md)
Anchored in the *Ecma-408 Dart Language Specification*, *Flutter Engineering Architecture & Engine Design Docs*, *Impeller Rendering Architecture Design Specs*, and *Effective Dart*. Covers everything from Dart 3 class modifiers (`sealed`, `base`, `mixin class`) and Scavenger GC to the C++ engine threading model, custom RenderObjects, and zero-copy Dart FFI.
