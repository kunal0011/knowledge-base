# Master Reference Literature & Authoritative Bibliography (`referencebook.md`)

> "If I have seen further, it is by standing on the shoulders of giants."  
> — *Sir Isaac Newton (1675)*

This document is the definitive master bibliography and canonical reading syllabus for every engineering discipline, platform, distributed system, and framework documented within this knowledge base. Every module is anchored in peer-reviewed computer science literature, industry-defining textbooks, foundational RFCs, and core architecture specifications.

---

## 📑 Table of Contents
1. [System Design & Large-Scale Distributed Architecture](#1-system-design--large-scale-distributed-architecture)
2. [Data-Intensive Applications & Database Storage Engines](#2-data-intensive-applications--database-storage-engines)
3. [Microservices Architecture & Distributed Patterns](#3-microservices-architecture--distributed-patterns)
4. [Enterprise Microservice Frameworks (Spring, FastAPI, Go)](#4-enterprise-microservice-frameworks-spring-fastapi-go)
5. [Distributed Data, Streaming Engines & Lakehouse Architecture](#5-distributed-data-streaming-engines--lakehouse-architecture)
6. [Kubernetes & Cloud-Native Infrastructure](#6-kubernetes--cloud-native-infrastructure)
7. [Operating Systems, Linux Kernel & Systems Programming](#7-operating-systems-linux-kernel--systems-programming)
8. [Computer Networking & Distributed Protocols](#8-computer-networking--distributed-protocols)
9. [Cryptography, Security Engineering & Identity](#9-cryptography-security-engineering--identity)
10. [Hardware, Embedded Systems & Microcontrollers](#10-hardware-embedded-systems--microcontrollers)
11. [GPU/TPU Architecture & Distributed LLM Training](#11-gputpu-architecture--distributed-llm-training)
12. [Artificial Intelligence, Deep Learning & Mathematical Foundations](#12-artificial-intelligence-deep-learning--mathematical-foundations)
13. [Native Android Engineering (Kotlin & Jetpack Compose)](#13-native-android-engineering-kotlin--jetpack-compose)
14. [Cross-Platform Mobile Engineering (Flutter & Dart)](#14-cross-platform-mobile-engineering-flutter--dart)
15. [Native iOS Engineering (Swift & SwiftUI)](#15-native-ios-engineering-swift--swiftui)
16. [Cross-Platform Mobile Engineering (React Native & Expo)](#16-cross-platform-mobile-engineering-react-native--expo)
17. [Modern Web Architecture (React 19 & Next.js 15)](#17-modern-web-architecture-react-19--nextjs-15)
18. [Low-Level Design (LLD), Object-Oriented Design & Clean Code](#18-low-level-design-lld-object-oriented-design--clean-code)
19. [Multimodal Large Language Models (MLLMs), Video Deep Learning & Real-Time Live Systems](#19-multimodal-large-language-models-mllms-video-deep-learning--real-time-live-systems)

---

## 1. System Design & Large-Scale Distributed Architecture
*Anchors: `Projects/System-Design-Alex-Xu/`*

1. **System Design Interview – An Insider's Guide (Volume 1)** — Alex Xu
2. **System Design Interview – An Insider's Guide (Volume 2)** — Alex Xu & Sahn Lam
3. **Designing Data-Intensive Applications** — Martin Kleppmann (O'Reilly)
4. **Site Reliability Engineering: How Google Runs Production Systems** — Betsy Beyer, Chris Jones, Jennifer Petoff, Niall Richard Murphy (O'Reilly)
5. **The Site Reliability Workbook** — Betsy Beyer, Niall Richard Murphy, David K. Rensin, Kent Kawahara, Stephen Thorne (O'Reilly)
6. **Building Evolutionary Architectures** — Neal Ford, Rebecca Parsons, Patrick Kua, Pramod Sadalage (O'Reilly)
7. **Release It!: Design and Deploy Production-Ready Software (2nd Edition)** — Michael T. Nygard (Pragmatic Bookshelf)
8. **Web Scalability for Startup Engineers** — Artur Ejsmont (McGraw-Hill)
9. **Software Architecture: The Hard Parts** — Neal Ford, Mark Richards, Pramod Sadalage, Zhamak Dehghani (O'Reilly)

---

## 2. Data-Intensive Applications & Database Storage Engines
*Anchors: `Projects/Designing-Data-Intensive-Applications/`*

1. **Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems** — Martin Kleppmann (O'Reilly)
2. **Database Internals: A Deep Dive into How Distributed Data Systems Work** — Alex Petrov (O'Reilly)
3. **Readings in Database Systems (The Red Book, 5th Edition)** — Peter Bailis, Joseph M. Hellerstein, Michael Stonebraker
4. **Transaction Processing: Concepts and Techniques** — Jim Gray & Andreas Reuter (Morgan Kaufmann)
5. **Principles of Distributed Database Systems (4th Edition)** — M. Tamer Özsu & Patrick Valduriez (Springer)
6. **Architecture of a Database System** — Joseph M. Hellerstein, Michael Stonebraker, James Hamilton (Foundations and Trends in Databases)
7. **The Google File System (GFS)** — Sanjay Ghemawat, Howard Gobioff, Shun-Tak Leung (ACM SOSP)
8. **Bigtable: A Distributed Storage System for Structured Data** — Fay Chang, Jeffrey Dean, Sanjay Ghemawat et al. (USENIX OSDI)
9. **Spanner: Google's Globally-Distributed Database** — James C. Corbett, Jeffrey Dean, Michael Epstein et al. (USENIX OSDI)
10. **In Search of an Understandable Consensus Algorithm (Raft)** — Diego Ongaro & John Ousterhout (USENIX ATC)

---

## 3. Microservices Architecture & Distributed Patterns
*Anchors: `Projects/Microservices-Architecture-and-Design/`*

1. **Building Microservices: Designing Fine-Grained Systems (2nd Edition)** — Sam Newman (O'Reilly)
2. **Microservices Patterns: With examples in Java** — Chris Richardson (Manning)
3. **Monolith to Microservices: Evolutionary Patterns to Transform Your Monolith** — Sam Newman (O'Reilly)
4. **Domain-Driven Design: Tackling Complexity in the Heart of Software** — Eric Evans (Addison-Wesley)
5. **Implementing Domain-Driven Design** — Vaughn Vernon (Addison-Wesley)
6. **Microservice Architecture: Aligning Principles, Practices, and Culture** — Irakli Nadareishvili, Ronnie Mitra, Matt McLarty, Mike Amundsen (O'Reilly)
7. **Enterprise Integration Patterns: Designing, Building, and Deploying Messaging Solutions** — Gregor Hohpe & Bobby Woolf (Addison-Wesley)
8. **Patterns of Enterprise Application Architecture (PoEAA)** — Martin Fowler (Addison-Wesley)

---

## 4. Enterprise Microservice Frameworks (Spring, FastAPI, Go)
*Anchors: `Projects/Microservices-Architecture-and-Design/Spring-and-Spring-Cloud/`, `FastAPI-Microservices/`, `Golang-Microservices/`*

### Spring Boot 3 & Spring Cloud
1. **Spring Microservices in Action (2nd Edition)** — John Carnell & Illary Huaylupo Sánchez (Manning)
2. **Cloud Native Spring in Action: With Spring Boot and Kubernetes** — Thomas Vitale (Manning)
3. **Spring Boot: Up and Running** — Mark Heckler (O'Reilly)
4. **Pro Spring 6: An In-Depth Guide to the Spring Framework** — Iuliana Cosmina, Rob Harrop, Chris Schaefer, Clarence Ho (Apress)
5. **Reactive Systems in Java: Resilient, Event-Driven Architecture with Spring and Project Reactor** — Clement Escoffier & Ken Finnigan (O'Reilly)

### FastAPI & Python Asynchronous Architecture
1. **Building Data Science Applications with FastAPI** — François Voron (Packt)
2. **Microservice APIs: Using Python, Flask, FastAPI, OpenAPI and more** — José Haro Peralta (Manning)
3. **Architecture Patterns with Python: Enabling Test-Driven Development, Domain-Driven Design, and Event-Driven Microservices** — Harry Percival & Bob Gregory (O'Reilly)
4. **Fluent Python: Clear, Concise, and Effective Programming (2nd Edition)** — Luciano Ramalho (O'Reilly)
5. **High Performance Python: Practical Performant Programming for Humans (2nd Edition)** — Micha Gorelick & Ian Ozsvald (O'Reilly)

### Golang Cloud-Native Architecture
1. **Cloud Native Go: Building Reliable Services in Unreliable Environments** — Matthew Titmus (O'Reilly)
2. **The Go Programming Language** — Alan A. A. Donovan & Brian W. Kernighan (Addison-Wesley)
3. **100 Go Mistakes and How to Avoid Them** — Teiva Harsanyi (Manning)
4. **Go Systems Programming: Master Linux and Unix system level programming with Go** — Mihalis Tsoukalos (Packt)
5. **gRPC: Up and Running: Building Extensible Production Services with Protocol Buffers and gRPC** — Kasun Indrasiri & Danesh Kuruppu (O'Reilly)

---

## 5. Distributed Data, Streaming Engines & Lakehouse Architecture
*Anchors: `Projects/Distributed-Data-and-Streaming-Engines/`, `Designing-Cloud-Data-Lake-and-Lakehouse/`, `big-data/`*

1. **Kafka: The Definitive Guide: Real-Time Data and Stream Processing at Scale (2nd Edition)** — Gwen Shapira, Todd Palino, Rajini Sivaram, Krit Petty (O'Reilly)
2. **Stream Processing with Apache Flink: Fundamentals, Implementation, and Operation of Streaming Applications** — Fabian Hüske & Vasiliki Kalavri (O'Reilly)
3. **Data Pipelines with Apache Airflow** — Bas P. Harenslak & Julian Rutger de Ruiter (Manning)
4. **Learning Spark: Lightning-Fast Data Analytics (2nd Edition)** — Jules S. Damji, Brooke Wenig, Tathagata Das, Denny Lee (O'Reilly)
5. **Spark: The Definitive Guide: Big Data Processing Made Simple** — Bill Chambers & Matei Zaharia (O'Reilly)
6. **The Enterprise Big Data Lake: Delivering the Promise of Big Data and Data Science** — Alex Gorelik (O'Reilly)
7. **Apache Iceberg: The Definitive Guide** — Tomer Shiran, Dain Sundstrom, Ryan Blue (O'Reilly)
8. **Delta Lake: The Definitive Guide** — Denny Lee, TD Das, Vartika Singh (O'Reilly)
9. **Streaming Systems: The What, Where, When, and How of Large-Scale Data Processing** — Tyler Akidau, Slava Chernyak, Reuven Lax (O'Reilly)
10. **Data Management at Scale: Best Practices for Enterprise Architecture (2nd Edition)** — Piethein Strengholt (O'Reilly)

---

## 6. Kubernetes & Cloud-Native Infrastructure
*Anchors: `Projects/k8s/`, `cloud/`*

1. **Kubernetes: Up and Running: Dive into the Future of Infrastructure (3rd Edition)** — Brendan Burns, Joe Beda, Kelsey Hightower, Lachlan Evenson (O'Reilly)
2. **Kubernetes in Action (2nd Edition)** — Marko Lukša (Manning)
3. **Cloud Native Patterns: Designing change-tolerant software** — Cornelia Davis (Manning)
4. **Core Kubernetes: Hands-on Service Inner Workings** — Chris Negus (Manning)
5. **Production Kubernetes: Building Successful Application Platforms** — Josh Rosso, Rich Lander, Alexander Brand, John Harris (O'Reilly)
6. **Managing Kubernetes Resources: Mastering the Kubernetes Control Plane** — Joseph Heck (O'Reilly)
7. **Certified Kubernetes Administrator (CKA) Study Guide** — Benjamin Muschko (O'Reilly)
8. **Kubernetes Security and Observability** — Brendan Creane & Amit Gupta (O'Reilly)
9. **Terraform: Up & Running: Writing Infrastructure as Code (3rd Edition)** — Yevgeniy Brikman (O'Reilly)

---

## 7. Operating Systems, Linux Kernel & Systems Programming
*Anchors: `Projects/operating-system/`*

1. **Operating Systems: Three Easy Pieces (OSTEP)** — Remzi H. Arpaci-Dusseau and Andrea C. Arpaci-Dusseau (Arpaci-Dusseau Books)
2. **Modern Operating Systems (4th/5th Edition)** — Andrew S. Tanenbaum & Herbert Bos (Pearson)
3. **Operating System Concepts (10th Edition, "The Dinosaur Book")** — Abraham Silberschatz, Peter B. Galvin, Greg Gagne (Wiley)
4. **The Linux Programming Interface: A Linux and UNIX System Programming Handbook** — Michael Kerrisk (No Starch Press)
5. **Understanding the Linux Kernel (3rd Edition)** — Daniel P. Bovet & Marco Cesati (O'Reilly)
6. **Linux Kernel Development (3rd Edition)** — Robert Love (Addison-Wesley)
7. **BPF Performance Tools: Deep Analysis and What to Do About It** — Brendan Gregg (Addison-Wesley)
8. **Systems Performance: Enterprise and the Cloud (2nd Edition)** — Brendan Gregg (Addison-Wesley)
9. **Advanced Programming in the UNIX Environment (3rd Edition)** — W. Richard Stevens & Stephen A. Rago (Addison-Wesley)

---

## 8. Computer Networking & Distributed Protocols
*Anchors: `Projects/networking/`*

1. **Computer Networking: A Top-Down Approach (8th Edition)** — James F. Kurose & Keith W. Ross (Pearson)
2. **Computer Networks (5th/6th Edition)** — Andrew S. Tanenbaum & David J. Wetherall (Pearson)
3. **TCP/IP Illustrated, Volume 1: The Protocols (2nd Edition)** — W. Richard Stevens & Kevin R. Fall (Addison-Wesley)
4. **High Performance Browser Networking: What every web developer should know about networking and web performance** — Ilya Grigorik (O'Reilly)
5. **HTTP: The Definitive Guide** — David Gourley, Brian Totty, Marjorie Sayer, Sailu Reddy, Anshu Aggarwal (O'Reilly)
6. **Network Warrior (2nd Edition)** — Gary A. Donahue (O'Reilly)
7. **BGP Design and Implementation** — Randy Zhang & Micah Bartell (Cisco Press)

---

## 9. Cryptography, Security Engineering & Identity
*Anchors: `Projects/cryptography/`*

1. **Applied Cryptography: Protocols, Algorithms, and Source Code in C (20th Anniversary Edition)** — Bruce Schneier (Wiley)
2. **Cryptography Engineering: Design Principles and Practical Applications** — Niels Ferguson, Bruce Schneier, Tadayoshi Kohno (Wiley)
3. **Serious Cryptography: A Practical Introduction to Modern Encryption** — Jean-Philippe Aumasson (No Starch Press)
4. **Understanding Cryptography: A Textbook for Students and Practitioners** — Christof Paar & Jan Pelzl (Springer)
5. **Security Engineering: A Guide to Building Dependable Distributed Systems (3rd Edition)** — Ross Anderson (Wiley)
6. **Bulletproof TLS and PKI (2nd Edition)** — Ivan Ristić (Feisty Duck)
7. **OAuth 2.0 in Action** — Justin Richer & Antonio Sanso (Manning)
8. **Zero Trust Networks: Building Secure Systems in Untrusted Networks** — Evan Gilman & Doug Barth (O'Reilly)

---

## 10. Hardware, Embedded Systems & Microcontrollers
*Anchors: `Projects/embedded-systems/`*

1. **Making Embedded Systems: Design Patterns for Great Software** — Elecia White (O'Reilly)
2. **The Art of Designing Embedded Systems (2nd Edition)** — Jack Ganssle (Newnes)
3. **Embedded Systems Architecture: Explore architectural concepts, pragmatic design patterns, and best practices (2nd Edition)** — Daniele Lacamera (Packt)
4. **Programming Embedded Systems: With C and GNU Development Tools (2nd Edition)** — Michael Barr & Anthony Massa (O'Reilly)
5. **Designing Embedded Hardware (2nd Edition)** — John Catsoulis (O'Reilly)
6. **Real-Time Systems: Design Principles for Distributed Embedded Applications** — Hermann Kopetz (Springer)
7. **The Linux Kernel Module Programming Guide** — Peter Jay Salzman, Michael Burian, Ori Pomerantz
8. **Raspberry Pi Cookbook: Software and Hardware Problems and Solutions (4th Edition)** — Simon Monk (O'Reilly)
9. **Exploring BeagleBone / Exploring Raspberry Pi** — Derek Molloy (Wiley)

---

## 11. GPU/TPU Architecture & Distributed LLM Training
*Anchors: `Projects/gpu-tpu-architecture-and-distributed-llm-training/`*

1. **Programming Massively Parallel Processors: A Hands-on Approach (4th Edition)** — David B. Kirk & Wen-mei W. Hwu (Morgan Kaufmann)
2. **Computer Architecture: A Quantitative Approach (6th Edition)** — John L. Hennessy & David A. Patterson (Morgan Kaufmann)
3. **General-Purpose Graphics Processor Architectures** — Tor M. Aamodt, Wilson Wai Lun Fung, Timothy G. Rogers (Synthesis Lectures on Computer Architecture)
4. **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism** — Mohammad Shoeybi et al. (NVIDIA Research)
5. **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** — Samyam Rajbhandari et al. (Microsoft DeepSpeed / SC20)
6. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** — Tri Dao et al. (NeurIPS 2022)
7. **A Domain-Specific Architecture for Deep Neural Networks (Google TPU v1)** — Norman P. Jouppi et al. (ISCA 2017)
8. **Scale-Out Supercomputing with Cloud TPU v4** — Norman P. Jouppi et al. (ISCA 2023)
9. **Efficient Large-Scale Language Model Training on GPU Clusters: The Co-design of DeepSpeed and Megatron-LM** — DeepSpeed Team (NVIDIA/Microsoft)

---

## 12. Artificial Intelligence, Deep Learning & Mathematical Foundations
*Anchors: `Projects/ai_math/`, `Data-and-AI-Frameworks/`*

### Core Deep Learning & Mathematical Foundations
1. **Deep Learning** — Ian Goodfellow, Yoshua Bengio, Aaron Courville (MIT Press)
2. **Deep Learning: Foundations and Concepts** — Christopher M. Bishop & Hugh Bishop (Springer, 2024)
3. **Pattern Recognition and Machine Learning** — Christopher M. Bishop (Springer)
4. **Mathematics for Machine Learning** — Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong (Cambridge University Press)
5. **Linear Algebra and Its Applications (5th Edition)** — David C. Lay, Steven R. Lay, Judi J. McDonald (Pearson)
6. **Introduction to Applied Linear Algebra: Vectors, Matrices, and Least Squares** — Stephen Boyd & Lieven Vandenberghe (Cambridge University Press)
7. **Convex Optimization** — Stephen Boyd & Lieven Vandenberghe (Cambridge University Press)
8. **Reinforcement Learning: An Introduction (2nd Edition)** — Richard S. Sutton & Andrew G. Barto (MIT Press)
9. **Speech and Language Processing (3rd Edition Draft)** — Dan Jurafsky & James H. Martin (Stanford University)

### Deep Generative Models, Diffusion Architectures & Flow Matching
1. **Deep Generative Models: CS236 Monograph & Course Notes** — Stefano Ermon & Aditya Grover (Stanford University)
2. **Deep Unsupervised Learning using Nonequilibrium Thermodynamics** — Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, Surya Ganguli (*International Conference on Machine Learning - ICML 2015*)
3. **Denoising Diffusion Probabilistic Models (DDPM)** — Jonathan Ho, Ajay Jain, Pieter Abbeel (*Conference on Neural Information Processing Systems - NeurIPS 2020*)
4. **Denoising Diffusion Implicit Models (DDIM)** — Jiaming Song, Chenlin Meng, Stefano Ermon (*International Conference on Learning Representations - ICLR 2021*)
5. **Score-Based Generative Modeling through Stochastic Differential Equations** — Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole (*International Conference on Learning Representations - ICLR 2021*)
6. **Classifier-Free Diffusion Guidance** — Jonathan Ho & Tim Salimans (*NeurIPS Workshop on NeurIPS NeurIPS 2021*)
7. **Elucidating the Design Space of Diffusion-Based Generative Models (EDM)** — Tero Karras, Miika Aittala, Timo Aila, Samuli Laine (*Conference on Neural Information Processing Systems - NeurIPS 2022*)
8. **Neural Ordinary Differential Equations** — Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud (*Conference on Neural Information Processing Systems - NeurIPS 2018 Best Paper Award*)
9. **Flow Matching for Generative Modeling** — Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le (*International Conference on Learning Representations - ICLR 2023*)
10. **Building Normalizing Flows with Stochastic Interpolants** — Michael S. Albergo & Eric Vanden-Eijnden (*International Conference on Learning Representations - ICLR 2023*)
11. **Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow** — Xingchao Liu, Chengyue Gong, Qiang Liu (*International Conference on Learning Representations - ICLR 2023*)
12. **Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (Stable Diffusion 3)** — Patrick Esser, Sumith Kulal, Andreas Blattmann et al. (*Conference on Computer Vision and Pattern Recognition - CVPR 2024*)
13. **High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)** — Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer (*IEEE/CVF Conference on Computer Vision and Pattern Recognition - CVPR 2022*)
14. **Scalable Diffusion Models with Transformers (DiT)** — William Peebles & Saining Xie (*IEEE/CVF International Conference on Computer Vision - ICCV 2023*)

---

## 13. Native Android Engineering (Kotlin & Jetpack Compose)
*Anchors: `Projects/Production-Android-and-Flutter-Development/Part 1 - Deep Kotlin & Android Internals/`*

1. **Kotlin in Action (2nd Edition)** — Dmitry Jemerov, Svetlana Isakova, Sebastian Aigner, Roman Elizarov (Manning)
2. **Effective Kotlin: Best Practices** — Marcin Moskala (Kt. Academy)
3. **Kotlin Coroutines: Deep Dive** — Marcin Moskala (Kt. Academy)
4. **Android Internals: A Confectioner's Cookbook (Volume 1 & 2)** — Jonathan Levin
5. **Under the Hood of Jetpack Compose** — Jorge Castillo
6. **Android Security Internals: An In-Depth Guide to Android's Security Architecture** — Nikolay Elenkov (No Starch Press)
7. **High Performance Android Apps** — Doug Sillars (O'Reilly)
8. **Android System Architecture (AOSP Core Documentation)** — Google Open Source (source.android.com)

---

## 14. Cross-Platform Mobile Engineering (Flutter & Dart)
*Anchors: `Projects/Production-Android-and-Flutter-Development/Part 2 - Deep Dart & Flutter Internals/`*

1. **The Dart Programming Language Specification (Ecma-408)** — Ecma International
2. **Flutter Complete Reference** — Alberto Miola
3. **Flutter Engineering Architecture & Engine Design Documentation** — Google Flutter Team (github.com/flutter/engine)
4. **Impeller Rendering Architecture Design Specifications** — Google Flutter Team (flutter.dev/to/impeller)
5. **Effective Dart: Style, Documentation, Usage & Design** — Dart Core Team (dart.dev)
6. **Concurrent Programming in Dart & Isolates Guide** — Dart Language Team

---

## 15. Native iOS Engineering (Swift & SwiftUI)
*Anchors: `Projects/Production-iOS-Development-Swift-and-SwiftUI/`*

1. **The Swift Programming Language (Swift 6.0+)** — Apple Inc. (swift.org)
2. **Advanced Swift** — Chris Eidhof, Ole Begemann, Airspeed Velocity (objc.io)
3. **Thinking in SwiftUI** — Florian Kugler & Chris Eidhof (objc.io)
4. **Swift Concurrency by Example** — Paul Hudson (Hacking with Swift)
5. **Pro Swift** — Paul Hudson (Hacking with Swift)
6. **Apple Platform Security: Hardware, software, and services security documentation** — Apple Security Team
7. **Point-Free Architecture & The Composable Architecture (TCA)** — Brandon Williams & Stephen Celis (pointfree.co)
8. **High Performance iOS Apps: Optimize Your Code for Speed and Power Efficiency** — Gaurav Vaish (O'Reilly)

---

## 16. Cross-Platform Mobile Engineering (React Native & Expo)
*Anchors: `Projects/Production-React-Native-and-Expo-Development/`*

1. **React Native New Architecture Specification (Fabric & TurboModules)** — Meta Open Source (reactnative.dev)
2. **The Ultimate Guide to React Native Optimization (2024 Edition)** — Callstack Engineering Team
3. **React Native in Action** — Nader Dabit (Manning)
4. **Fullstack React Native: The Complete Guide to React Native** — Devin Abbott, Houssein Djirdeh, Sophia Shoemaker (Fullstack.io)
5. **Continuous Native Generation (CNG) & Expo Architecture Documentation** — Expo Team (docs.expo.dev)
6. **Yoga Layout Engine Architecture Documentation** — Meta Open Source (yogacss.com)
7. **Mobile Application Security Verification Standard (MASVS)** — OWASP Foundation

---

## 17. Modern Web Architecture (React 19 & Next.js 15)
*Anchors: `Projects/Production-React-and-Nextjs-Web-Development/`*

1. **React 19 Official Documentation, RFCs & Core Source Architecture** — Meta React Team (react.dev)
2. **Next.js 15 App Router Architecture & Caching Specification** — Vercel Engineering (nextjs.org/docs)
3. **Inside Fiber: In-depth overview of the reconciliation algorithm in React** — Max Koretskyi (ag-grid / Netbasal)
4. **Learning React: Modern Patterns for Developing React Apps (2nd Edition)** — Alex Banks & Eve Porcello (O'Reilly)
5. **Effective TypeScript: 62 Specific Ways to Improve Your TypeScript** — Dan Vanderkam (O'Reilly)
6. **Web Vitals & Performance Engineering** — Google Chrome Web Standards Team (web.dev)
7. **Production Web Security: OWASP Top 10 Web Application Security Risks** — OWASP Foundation

---

## 18. Low-Level Design (LLD), Object-Oriented Design & Clean Code
*Anchors: `Projects/LLD/`, `Projects/Coding/`*

1. **Design Patterns: Elements of Reusable Object-Oriented Software (Gang of Four)** — Erich Gamma, Richard Helm, Ralph Johnson, John Vlissides (Addison-Wesley)
2. **Clean Code: A Handbook of Agile Software Craftsmanship** — Robert C. Martin (Prentice Hall)
3. **Clean Architecture: A Craftsman's Guide to Software Structure and Design** — Robert C. Martin (Prentice Hall)
4. **Refactoring: Improving the Design of Existing Code (2nd Edition)** — Martin Fowler (Addison-Wesley)
5. **Head First Design Patterns (2nd Edition)** — Eric Freeman & Elisabeth Robson (O'Reilly)
6. **Working Effectively with Legacy Code** — Michael C. Feathers (Prentice Hall)
7. **Cracking the Coding Interview (6th Edition)** — Gayle Laakmann McDowell (CareerCup)
8. **Introduction to Algorithms (CLRS, 4th Edition)** — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein (MIT Press)

---

## 19. Multimodal Large Language Models (MLLMs), Video Deep Learning & Real-Time Live Systems
*Anchors: `Projects/Multimodal-Deep-Learning-and-Production-MLLM-Systems/`*

### Foundational Textbooks & Monographs
1. **Deep Learning: Foundations and Concepts** — Christopher M. Bishop & Hugh Bishop (Springer, 2024, Ch. 19 & 20)
2. **Speech and Language Processing (3rd Edition Draft)** — Dan Jurafsky & James H. Martin (Stanford University, 2024, Ch. 26 "Dialogue Systems and Chatbots")
3. **Computer Vision: Algorithms and Applications (2nd Edition)** — Richard Szeliski (Springer)

### Vision-Language Alignment & Dynamic Resolution
4. **Visual Instruction Tuning (LLaVA)** — Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee (*NeurIPS 2023*)
5. **Improved Baselines with Visual Instruction Tuning (LLaVA-NeXT)** — Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yong Jae Lee (CVPR 2024)
6. **Flamingo: a Visual Language Model for Few-Shot Learning** — Jean-Baptiste Alayrac et al. (*Google DeepMind, NeurIPS 2022*)
7. **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models** — Junnan Li, Dongxu Li, Silvio Savarese, Steven C.H. Hoi (*Salesforce Research, ICML 2023*)
8. **SigLIP: Sigmoid Loss for Language Image Pre-Training** — Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer (*Google DeepMind, ICCV 2023*)
9. **NaViT: Native Resolution ViT** — Mostafa Dehghani, Basil Mustafa et al. (*Google DeepMind, NeurIPS 2023*)
10. **InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks** — Zhe Chen et al. (*OpenGVLab, CVPR 2024*)

### Video Deep Learning, 3D RoPE & Long-Context Architectures
11. **Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution** — Qwen Team (*Alibaba, 2024*)
12. **Gemini 1.5: Unlocking Multimodal Understanding Across Millions of Tokens of Context** — Gemini Team (*Google DeepMind, 2024*)
13. **Video-LLaVA: Learning United Visual Representation by Alignment Before Projection** — Bin Lin et al. (*CVPR 2024*)
14. **Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models** — Muhammad Maaz et al. (*ACL 2024*)
15. **RingAttention with Blockwise Transformers for Near-Infinite Context** — Hao Liu, Matei Zaharia, Pieter Abbeel (*ICLR 2024*)

### Real-Time Live Streaming, Neural Audio Codecs & WebSockets/WebRTC
16. **Gemini 2.0 Multimodal Live API Protocol & Systems Architecture** — Google DeepMind (2024–2025)
17. **OpenAI Realtime API Architecture & System Card** — OpenAI (2024)
18. **SoundStream: An End-to-End Neural Audio Codec** — Neil Zeghidour et al. (*Google Research, IEEE/ACM TASLP 2021*)
19. **High-Fidelity Audio Compression with Neural Networks (EnCodec)** — Alexandre Défossez et al. (*Meta AI, 2022*)
20. **Mimi: Streaming Neural Audio Codec with 12.5 Hz Framerate** — Kyutai Labs (2024)
21. **RFC 8825: Overview: Real-Time Communication in Browsers (WebRTC)** — IETF (2021)
22. **RFC 6455: The WebSocket Protocol** — IETF (2011)

