# Compiler-Inspired Execution Planning for LLM Request Dispatch

> **Working title** — subject to revision  
> **Target venue** — ICSE 2027 (Technical Track)  
> **Paper type** — Benchmark + Empirical Study (this document covers Paper 1)

---

## 1. The Problem in One Paragraph

Every LLM serving system in production today makes the same silent assumption: that all requests should be executed the same way. The operator selects an execution strategy at deployment time — chunked prefill or full prefill, tensor parallelism degree, speculative decoding on or off — and that configuration is applied uniformly to every request for the lifetime of the deployment. A 32-token chat message and a 6,000-token document summarization request are processed identically. A simple factual lookup and a multi-step reasoning chain that will generate 1,000 tokens are handled by the same scheduler with the same parallelism and the same speculation policy. This is not a deliberate design choice. It is a blind spot — one the community has not yet named, let alone addressed.

---

## 2. Motivation

### 2.1 The Heterogeneity of Real LLM Workloads

Production LLM deployments serve radically heterogeneous request populations. Empirical analyses of serving traces from ShareGPT, LMSYS, and enterprise deployments consistently show:

- **Prompt length variance spans two orders of magnitude.** The same endpoint serves requests ranging from 50 tokens to 8,000+ tokens.
- **Output length is highly unpredictable and task-dependent.** A code generation request may produce 800 tokens; a sentiment classification request produces 3. Neither the operator nor the scheduler knows in advance.
- **Task types co-exist within the same deployment.** Interactive chat, document summarization, code generation, and chain-of-thought reasoning all arrive on the same queue, with fundamentally different compute and memory profiles.
- **Arrival patterns are bursty, not uniform.** Real traffic exhibits diurnal patterns, flash crowds, and priority-stratified SLOs that static configurations cannot accommodate.

This heterogeneity is not incidental — it is the defining characteristic of general-purpose LLM deployments.

### 2.2 The Cost of Strategy Mismatch

Each execution strategy has a distinct performance profile that interacts with request characteristics in non-trivial ways:

**Chunked prefill** reduces Time to First Token (TTFT) for concurrent long-prompt requests by interleaving prefill computation across scheduling steps. For short prompts, however, it adds unnecessary scheduling overhead with no benefit — and in some cases degrades TTFT by fragmenting what would be a single fast prefill pass.

**Speculative decoding** accelerates decode throughput when the draft model's acceptance rate is high — typically for repetitive or predictable output (code, structured data, short chat replies). For long reasoning chains with high token entropy, acceptance rates collapse and the speculative overhead becomes net-negative.

**Tensor parallelism** reduces per-request latency at high TP degrees by distributing attention computation across devices. For short sequences with small batch sizes, the inter-device communication overhead dominates and TP-2 is *slower* than TP-1. For long prompts at high concurrency, TP-2 and TP-4 provide meaningful latency reduction.

**Quantization** reduces memory bandwidth pressure and enables higher batch sizes, improving throughput. The latency tradeoff depends on whether the workload is compute-bound or memory-bound — which in turn depends on batch size and sequence length at the moment the request is scheduled.

The key insight: **the optimal strategy for a given request depends on that request's characteristics, not on a global deployment-time decision.** A deployment optimized for short interactive chat will systematically mis-serve long summarization requests, and vice versa. The magnitude of this mis-service — which we measure precisely in this paper — motivates a per-request execution planner.

### 2.3 The Missing Abstraction

The database community solved an analogous problem four decades ago. In the 1970s, relational database systems executed every query with a fixed access strategy: full table scans, in insertion order, with no awareness of data statistics. The invention of the query optimizer — a cost-based planner that selects a physical execution plan per query from a space of alternatives — transformed database performance from a configuration art into a principled engineering discipline. Today, no serious database system executes a query without consulting a cost model.

LLM serving in 2025 is where databases were in 1975. The execution strategy space exists and is well-understood. The request features that should drive strategy selection are measurable. The cost models can be fitted from profiling data. What is missing is the planning layer that connects them — and the empirical characterization of the strategy space that would justify building it.

**This paper provides that empirical characterization.** We define the execution strategy space formally, profile its cost landscape across a comprehensive workload matrix, and demonstrate that strategy selection is a well-posed optimization problem with measurable, learnable cost structure. The planner itself is Paper 2.

---

## 3. The Core Analogy — Precise, Not Decorative

The query optimizer analogy is central to this work. We use it precisely, not as a metaphor.

| Query Optimizer Concept | Our Instantiation |
|---|---|
| **Logical plan** | Abstract request representation: (prompt_len, task_type, priority_class) |
| **Physical operators** | Execution strategies: chunked prefill, speculative decoding, TP-N, quantization |
| **Operator algebra** | Formal strategy space with composability rules and mutual-exclusion constraints |
| **Cardinality estimation** | Output length prediction, KV cache hit probability, task type classification |
| **Operator cost model** | Parametric latency model per strategy, fitted from profiling data |
| **Plan enumeration** | Search over valid strategy combinations given request features |
| **Plan selection** | Minimum-cost strategy assignment at request admission time |
| **Plan cache** | Precomputed strategy assignments for common (prompt_len, output_len) buckets |
| **Statistics drift** | Workload distribution shift requiring cost model recalibration |

Each component in the right column has a direct, implementable counterpart. The analogy is productive — it imports four decades of cost-based optimization research into a new domain.

---

## 4. Research Questions

This paper is organized around five empirical research questions:

**RQ1 — Strategy Dominance**
> *Does any single execution strategy achieve the lowest TTFT P95 across all (prompt length, output length, task type) regimes? If not, which strategy dominates in each regime?*

We expect to show that no strategy dominates — that the optimal choice varies systematically with request characteristics. This is the primary finding that motivates Paper 2.

**RQ2 — Cost Separability**
> *Are strategy costs separable across request dimensions? Specifically, does prefill cost depend primarily on prompt length and decode cost on output length, independently of each other?*

Separability is the structural property that makes the cost model tractable. If costs are separable, a cost model with two independent terms (prefill cost + decode cost) is sufficient. If not, joint modeling is required. We quantify separability empirically using feature ablation R² analysis.

**RQ3 — Cost Model Accuracy**
> *How accurately can a fitted cost model predict per-strategy latency from request features? What is the per-strategy MAPE across prompt length and output length dimensions?*

This validates that the cost model component of a future planner is learnable and accurate enough to drive plan selection. If MAPE exceeds ~20%, the cost model is too noisy for reliable plan selection.

**RQ4 — Strategy Interaction Effects**
> *When execution strategies are combined (e.g., chunked prefill + speculative decoding), are their costs additive, subadditive (synergistic), or super-additive (interfering)?*

Interaction effects determine whether the strategy space can be treated as a product of independent dimensions or requires explicit modeling of combinations. We measure interaction effects as the deviation from the sum of individual effects.

**RQ5 — Misconfiguration Cost**
> *What is the latency penalty of static misconfiguration? Specifically, what is the P95 TTFT increase when a deployment is optimized for one workload type and presented with another?*

This quantifies the cost of the status quo — the headroom available to a per-request planner. If misconfiguration costs are small, the planning overhead is not justified. If they are large (which we expect), the planner is worth building.

---

## 5. Contributions

This paper makes four concrete contributions:

### Contribution 1 — Formal Execution Strategy Space (Operator Algebra)

We define the first formal operator algebra for LLM request execution strategies. The algebra specifies:

- The set of independently configurable execution dimensions (prefill mode, parallelism degree, speculation policy, quantization level)
- Composability rules: which dimension combinations are valid (e.g., speculative decoding requires a compatible draft model) and which are mutually exclusive
- A fractional factorial design over the strategy space that covers all meaningful combinations without requiring exhaustive enumeration

This formalization is the necessary precondition for cost-based planning. It establishes that LLM request dispatch has a well-defined, tractable search space — a fact that has been implicit in the community but never formalized.

### Contribution 2 — Comprehensive Cost Landscape Characterization

We profile 12 strategies across 6 workloads (covering 4 task types, 5 prompt length ranges, 4 output length ranges, and 3 arrival patterns) on two GPU hardware types (A100, L40S). For each (strategy, request) combination we measure TTFT P50/P95/P99, E2E latency, decode throughput, GPU utilization, KV cache pressure, and preemption count.

Key empirical findings (anticipated):
- No single strategy dominates: the optimal strategy shifts at predictable prompt/output length thresholds
- Costs are largely separable: prefill cost is prompt-length-driven; decode cost is output-length-driven
- Strategy interaction effects are measurable: chunked prefill + speculative decoding is subadditive for summarization but super-additive for reasoning workloads
- Misconfiguration cost is significant: static deployment for chat workloads degrades P95 TTFT by 2–4× on summarization traces

### Contribution 3 — Fitted Per-Strategy Cost Models

We fit parametric and learned (gradient-boosted) cost models for each strategy using request features (prompt length, output length, task type) as inputs. We report cross-validated MAPE for TTFT and E2E latency per strategy, and demonstrate that costs are predictable with <15% MAPE for 10 of 12 strategies. We further analyze which features drive prediction accuracy per strategy, providing the statistical foundation for the cardinality estimator in Paper 2.

### Contribution 4 — Open Benchmark Harness (Replication Package)

We release a fully open-source benchmark harness that:

- Launches, manages, and tears down vLLM instances with arbitrary strategy configurations
- Replays request traces with configurable arrival processes (Poisson, uniform, bursty)
- Measures TTFT via SSE streaming with millisecond resolution
- Samples GPU utilization, memory, and power at 1 Hz
- Generates paper-quality figures (TTFT CDF, dominance heatmap, interaction matrices) and LaTeX tables directly from results

The harness is designed to be model-agnostic and hardware-agnostic — any vLLM-compatible model on any CUDA-capable GPU can be evaluated. We intend this as a community standard for LLM serving strategy evaluation, analogous to TPC-H for database query optimization benchmarking.

---

## 6. What This Paper Is Not

To set expectations clearly:

- **This paper does not build a planner.** It characterizes the problem space that a planner must solve. The planner — including the plan selector, admission-time feature extractor, and vLLM integration — is Paper 2.
- **This paper does not claim that any strategy is universally better.** The central finding is the opposite: context-dependence of strategy optimality.
- **This paper does not address disaggregated prefill** (DistServe/Splitwise architecture) as a strategy dimension. Disaggregation requires a separate cluster topology and is orthogonal to the intra-instance strategy choices studied here. It is a natural extension for future work.
- **This paper does not study multi-tenant isolation or SLO enforcement.** Workloads are single-tenant. Multi-tenant strategy interaction is an open problem.

---

## 7. Relationship to Prior Work

### LLM Serving Systems

vLLM introduced continuous batching and PagedAttention, establishing the dominant serving architecture. Subsequent work — Sarathi-Serve (chunked prefill), DistServe (prefill-decode disaggregation), Medusa and EAGLE (speculative decoding), SGLang (RadixAttention for prefix caching) — each optimizes a single execution dimension in isolation. **No prior work characterizes the joint strategy space or studies how these optimizations interact.** This paper fills that gap.

### Query Optimization

The Selinger optimizer (System R, 1979) introduced cost-based plan selection for relational queries. The Volcano/Cascades framework (Graefe, 1993; 1995) generalized this to an extensible rule-based search over physical operators. Recent learned query optimizers (Bao, Neo, HybridQO) replace hand-crafted cost models with learned models — directly analogous to our GBT cost model fitting. We position our work as the first application of this intellectual tradition to LLM serving.

### Self-Adaptive Systems

MAPE-K (Kephart & Chess, 2003) provides the autonomic computing framework for systems that monitor and adapt their own configuration. Prior work (TAILOR, HASKI) applies MAPE-K to LLM serving at the request routing level. This paper provides the empirical foundation for applying MAPE-K at the finer granularity of per-request execution planning.

### Benchmarking LLM Serving

MLPerf Inference and LMBench measure throughput and latency of LLM serving systems but treat the serving system as a black box and evaluate a single fixed configuration. They do not vary execution strategies or characterize strategy-request interactions. This paper is the first benchmark that treats the execution strategy as the independent variable.

---

## 8. Anticipated Impact

If the empirical findings are as expected — no single strategy dominates, costs are separable, and misconfiguration penalties are significant — this paper makes three contributions to the SE community:

1. **It names the problem.** Static execution strategy selection in LLM serving is a form of configuration debt. Naming and quantifying it is the first step toward solving it.

2. **It provides the data.** The cost landscape characterization and fitted cost models are immediately usable by any team building a serving system optimizer. The benchmark harness makes the results reproducible and extensible.

3. **It establishes the intellectual bridge.** Importing query optimization concepts into LLM serving is a concrete, actionable research direction. This paper validates the analogy empirically; Paper 2 implements it.

---

## 9. Paper Outline (Planned)

```
1. Introduction
   1.1 The static configuration problem
   1.2 The query optimizer analogy
   1.3 Contributions and paper organization

2. Background
   2.1 LLM serving execution strategies
   2.2 Query optimization primer
   2.3 Motivating example: strategy × workload cost matrix

3. Execution Strategy Space
   3.1 Formal operator algebra
   3.2 Composability rules
   3.3 Benchmark strategy set (fractional factorial design)

4. Benchmark Design
   4.1 Workload characterization
   4.2 Metrics and measurement methodology
   4.3 Infrastructure and reproducibility

5. Empirical Results
   5.1 RQ1 — Strategy dominance
   5.2 RQ2 — Cost separability
   5.3 RQ3 — Cost model accuracy
   5.4 RQ4 — Interaction effects
   5.5 RQ5 — Misconfiguration cost

6. Cost Model
   6.1 Feature engineering
   6.2 Model selection and fitting
   6.3 Prediction accuracy and feature importance

7. Threats to Validity
   7.1 Internal validity
   7.2 External validity (model generalization, hardware generalization)
   7.3 Construct validity

8. Related Work
9. Conclusion and Future Work (Paper 2 preview)
```

---

## 10. Threats to Validity (Draft)

**Internal validity:** Benchmark results are sensitive to concurrent system processes and GPU thermal state. All experiments are run with exclusive GPU access, a 5-minute cooldown between strategy runs, and three repeated trials per (strategy, workload) combination. We report median across trials.

**External validity — model generalization:** We evaluate on Llama-3.1-8B. Cost model structure (prefill cost ∝ prompt_len², decode cost ∝ output_len) is derived from transformer arithmetic and is model-architecture-agnostic. We validate on a second model (Mistral-7B or DeepSeek-R1-Distill-8B) to confirm generalization.

**External validity — hardware generalization:** We profile on A100-40GB and L40S-48GB. These cover the two most common academic and cloud GPU types. H100 is left for future work; the cost model framework accommodates new hardware via re-profiling.

**Construct validity:** TTFT is the primary metric because it directly reflects user-perceived responsiveness for interactive workloads. For batch workloads, throughput (tokens/sec) is more appropriate. We report both and discuss the tradeoff.

---

*Document status: Working notes — to be converted to LaTeX after RQ1–RQ3 results are in hand.*
