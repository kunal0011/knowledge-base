# Part 2: Deep Next.js Architecture

Welcome to **Part 2** of the Production Web Engineering curriculum. This section covers the Next.js 15 enterprise web engine, from App Router runtime topologies, streaming SSR over HTTP/2, and the 4-tier caching model, to Route Handlers, Turbopack, and multi-tenant project structures.

---

## 🏛️ Comprehensive Chapter Syllabus

```
Projects/Production-React-and-Nextjs-Web-Development/Part 2 - Deep Nextjs Architecture/
├── 01. App Router Internals, Layouts & Server-Client Boundary.md
├── 02. Streaming SSR, Progressive Hydration & Suspense.md
├── 03. The Next.js 15 Multi-Tier Caching Pipeline.md
├── 04. Route Handlers, Server Middleware & Edge Runtime.md
├── 05. Enterprise Project Structure, Turbopack & Monorepo.md
└── 06. Security Hardening, XSS, CSRF & Enterprise CI-CD.md
```

### Module Breakdown & Coverage
* **01. App Router Internals & Server-Client Boundary**: Nested layouts, parallel routes (`@modal`), intercepting routes (`(.)photo`), root layouts, metadata generation, and mastering the `'use client'` boundary (composition patterns, passing Server Components as `children`).
* **02. Streaming SSR, Progressive Hydration & Suspense**: The mechanics of `ReadableStream`, chunked transfer encoding (`Transfer-Encoding: chunked`), out-of-order streaming HTML replacement (`$RC` template swapping), and selective client-side hydration.
* **03. The Next.js 15 Multi-Tier Caching Pipeline**: Deep dive into the 4 caching tiers: Request Memoization, Data Cache (`fetch` with tags), Full Route Cache (Static SSG vs Dynamic SSR), and Client-side Router Cache. Cache invalidation strategies via `revalidateTag` and `revalidatePath`.
* **04. Route Handlers & Edge Runtime**: Next.js Route Handlers (`GET`, `POST`, `PATCH`), NextRequest/NextResponse, Server Middleware execution order, the Edge Runtime (V8 isolates without Node.js APIs), Web Crypto, and rate limiting.
* **05. Enterprise Project Structure & Turbopack**: Production-grade directory layouts (`app/`, `src/components/`, `src/features/*`, `src/lib/`), Turbopack Rust incremental compiler architecture, and monorepo configurations with Turborepo.
* **06. Security Hardening, XSS, CSRF & CI-CD**: Sanitizing untrusted inputs, Server Action CSRF protections, Content Security Policy (CSP) headers, Server-Only package guards (`import 'server-only'`), and automated GitHub Actions with Vercel / Docker deployment.

Every chapter includes extensive **Do's, Don'ts & Gotchas** tables detailing subtle production pitfalls.
