# Production Golang Microservices Architecture

> "Go is the de-facto lingua franca of cloud-native infrastructure (Kubernetes, Docker, Terraform, Prometheus). Writing production Go microservices requires mastering the Go runtime's M:N Goroutine scheduler, zero-allocation buffers, compile-time gRPC protocols, clean architecture boundaries, and graceful termination."  
> — *Synthesized from Cloud Native Go (Matthew Titmus), Go Systems Programming, and The Go Programming Language (Alan Donovan & Brian Kernighan)*

---

## 📚 Canonical Literature & Authoritative References
1. **Cloud Native Go: Building Reliable Services in Unreliable Environments** (Matthew Titmus / O'Reilly)
2. **The Go Programming Language** (Alan A. A. Donovan & Brian W. Kernighan / Addison-Wesley)
3. **100 Go Mistakes and How to Avoid Them** (Teiva Harsanyi / Manning)
4. **gRPC Go & Protocol Buffers Production Architecture Guides**

---

## 🏛️ Comprehensive Enterprise Go Architecture

```mermaid
flowchart TD
    Client["Client / External Ingress"] --> Ingress["Cloud Native Load Balancer (Envoy / K8s Service)"]

    subgraph GoMicroservice["Go Microservice Binary (Single Static ELF Binary)"]
        Server["Dual Protocol Server: gRPC (Port 9090) & HTTP/REST (Port 8080)"]
        Middleware["Interceptors / Middleware: OpenTelemetry Tracing, Prometheus Metrics, Auth"]
        Handler["Transport Layer: gRPC Server & Chi HTTP Router"]
        Usecase["Domain Service / Use Cases (Zero external dependencies)"]
        Repository["Data Access Layer: pgxpool (PostgreSQL) & go-redis"]
    end

    subgraph Infrastructure["External Infrastructure"]
        Postgres["PostgreSQL Cluster"]
        KafkaCluster["Apache Kafka (segmentio/kafka-go)"]
        Jaeger["Jaeger / OpenTelemetry Collector"]
    end

    Ingress --> Server
    Server --> Middleware
    Middleware --> Handler
    Handler --> Usecase
    Usecase --> Repository
    Repository --> Postgres
    Usecase --> KafkaCluster
    Middleware -.-> Jaeger
```

---

## 1. Go Runtime Internals: The M:N Scheduler (GMP Model)

Go does not map Goroutines 1:1 with operating system threads. It uses the **GMP Model**:

```mermaid
flowchart TD
    subgraph GMPModel["Go Runtime M:N Scheduler"]
        G1["Goroutine G1 (2 KB stack)"]
        G2["Goroutine G2"]
        G3["Goroutine G3"]

        P["Processor P (Logical Context, Run Queue)"]
        M["Machine M (Physical OS Kernel Thread)"]
        CPU["Physical CPU Core"]

        G1 --> P
        G2 --> P
        G3 --> P
        P --> M
        M --> CPU
    end
```

* **G (Goroutine)**: Lightweight user-space thread. Starts with a tiny **2 KB stack** that grows dynamically, allowing a single service to run **1,000,000+ active goroutines simultaneously**.
* **M (Machine)**: Operating system kernel thread managed by the OS scheduler.
* **P (Processor)**: Logical context representing execution resources. Exactly equal to `GOMAXPROCS` (number of CPU cores).
* **Work Stealing**: If a thread's local run-queue is empty, it steals half the goroutines from another processor's queue, balancing CPU load with zero lock contention.

---

## 2. Production Clean Architecture Project Structure in Go

Enterprise Go repositories adhere to the **Standard Go Project Layout** (`golang-standards/project-layout`):

```
my-enterprise-go-service/
├── cmd/
│   └── order-service/
│       └── main.go                             # Application entrypoint & dependency wiring
├── internal/                                   # Private code (Compiler prevents external imports)
│   ├── domain/                                 # Pure business entities & repository interfaces
│   │   ├── order.go
│   │   └── repository.go
│   ├── usecase/                                # Application business logic
│   │   └── order_usecase.go
│   ├── transport/
│   │   ├── grpc/                               # gRPC server & protobuf handlers
│   │   │   ├── handler.go
│   │   │   └── pb/                             # Generated protobuf code (*.pb.go)
│   │   └── http/                               # REST HTTP handlers (Chi / Gin)
│   │       └── router.go
│   └── platform/                               # Infrastructure & database adapters
│       ├── database/                           # pgx connection pool
│       │   └── postgres.go
│       └── kafka/                              # Kafka publisher
│           └── producer.go
├── api/
│   └── proto/                                  # Protobuf definition files (.proto)
│       └── order_service.proto
├── config/                                     # Configuration structs & environment loading
│   └── config.go
├── go.mod
├── go.sum
└── Dockerfile                                  # Multi-stage scratch Docker build (~15MB image)
```

---

## 3. High-Performance gRPC Service Implementation

### 3.1 Protobuf Contract (`api/proto/order_service.proto`)
```protobuf
syntax = "proto3";

package enterprise.order.v1;
option go_package = "internal/transport/grpc/pb";

service OrderService {
  rpc CreateOrder(CreateOrderRequest) returns (CreateOrderResponse);
}

message CreateOrderRequest {
  string customer_id = 1;
  int64 amount_cents = 2;
  repeated string item_ids = 3;
}

message CreateOrderResponse {
  string order_id = 1;
  string status = 2;
}
```

### 3.2 Production Go Implementation with Graceful Shutdown
```go
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	pb "my-enterprise-go-service/internal/transport/grpc/pb"
)

// 1. gRPC Server Handler
type OrderServer struct {
	pb.UnimplementedOrderServiceServer
	dbPool *pgxpool.Pool
}

func (s *OrderServer) CreateOrder(ctx context.Context, req *pb.CreateOrderRequest) (*pb.CreateOrderResponse, error) {
	if req.GetCustomerId() == "" || req.GetAmountCents() <= 0 {
		return nil, status.Error(codes.InvalidArgument, "invalid customer ID or amount")
	}

	orderID := fmt.Sprintf("ORD-%d", time.Now().UnixNano())

	// Fast binary database execution using pgxpool
	query := `INSERT INTO orders (id, customer_id, amount_cents, status) VALUES ($1, $2, $3, $4)`
	_, err := s.dbPool.Exec(ctx, query, orderID, req.GetCustomerId(), req.GetAmountCents(), "CONFIRMED")
	if err != nil {
		return nil, status.Errorf(codes.Internal, "database transaction failed: %v", err)
	}

	return &pb.CreateOrderResponse{
		OrderId: orderID,
		Status:  "CONFIRMED",
	}, nil
}

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 2. High-Performance PostgreSQL Pool
	dbConfig, _ := pgxpool.ParseConfig("postgres://user:secret@localhost:5432/orders_db?pool_max_conns=50")
	pool, err := pgxpool.NewWithConfig(ctx, dbConfig)
	if err != nil {
		log.Fatalf("Unable to connect to database: %v\n", err)
	}
	defer pool.Close()

	// 3. gRPC Listener
	lis, err := net.Listen("tcp", ":9090")
	if err != nil {
		log.Fatalf("Failed to listen on port 9090: %v\n", err)
	}

	grpcServer := grpc.NewServer()
	pb.RegisterOrderServiceServer(grpcServer, &OrderServer{dbPool: pool})

	// 4. Graceful Shutdown Coordinator
	go func() {
		log.Println("gRPC Server running on :9090...")
		if err := grpcServer.Serve(lis); err != nil && !errors.Is(err, grpc.ErrServerStopped) {
			log.Fatalf("gRPC Server failed: %v\n", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Received termination signal. Draining gRPC connections...")
	grpcServer.GracefulStop()
	log.Println("Server stopped gracefully.")
}
```

---

## 4. Do's, Don'ts & Production Gotchas

| Category | Rule | Deep Technical Rationale |
| :--- | :--- | :--- |
| **DON'T** | Never spawn goroutines without a deterministic lifecycle or cancellation context. | Spawning `go worker()` without passing a `context.Context` or managing a `sync.WaitGroup` creates **goroutine leaks**. If the worker blocks indefinitely on an unbuffered channel or slow socket, the goroutine remains alive in memory forever, eventually triggering an OOM crash. |
| **DO** | Use `pgxpool` instead of `database/sql` for PostgreSQL. | JackC's `pgx` bypasses `database/sql` reflection wrappers, speaking the native PostgreSQL binary wire protocol directly. It delivers **2x to 3x higher throughput** and native support for composite types and copy operations. |
| **GOTCHA** | Beware of closing channels from the receiver side. | In Go, sending to a closed channel causes a **fatal panic**. Closing an already closed channel causes a **fatal panic**. Always obey the channel idiom: **The sender that owns the channel is the only entity permitted to close it**. |
| **DO** | Build scratch or distroless Docker containers. | Go compiles into a single, statically linked ELF binary. Building with `CGO_ENABLED=0` allows running on `scratch`, producing an ultra-secure, minimal **15 MB Docker container** with zero shell utilities for attackers to exploit. |
| **DON'T** | Do not ignore errors returned by `defer resp.Body.Close()`. | Failing to read and close HTTP response bodies prevents the underlying TCP socket from returning to the `http.Transport` connection pool, forcing Go to open new TCP handshakes for every request. |
