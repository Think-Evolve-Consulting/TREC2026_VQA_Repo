# Session Documentation: Multi-Agent VQA Architecture

**Date:** August 14, 2026
**Topic:** Architecture and Execution Strategy for a Multi-Agent Video QA System
**Dataset Profile:** 2,443 queries across 2,411+ videos (primarily 30s reels, 15% >150s, 15 videos 250-386s in WebM/MP4 format).

## 1. Initial Architecture Proposal

The session began with designing a highly decoupled, multi-agent framework capable of handling multimodal inputs and producing 10 answer variations per query. The architecture targets a mix of short-form vertical reels and extended media without incurring heavy processing costs for every query.

### System Architecture Overview
```text
                    ┌─────────────────────────┐
                    │      Video Dataset      │
                    │ (.mp4, .webm, 9:16/Reels)│
                    └────────────┬────────────┘
                                 │
                   [Offline Ingestion Pipeline]
                                 │
           ┌─────────────────────┼─────────────────────┐
           ▼                     ▼                     ▼
    Visual Indexing      Audio/ASR Indexing    Metadata Indexing
 (Keyframes, OCR, Motion) (Whisper-v3, AudioSE)  (Formats, Durations)
           └─────────────────────┬─────────────────────┘
                                 │
                          [Query Entry Point]
                                 │
                     ┌───────────▼───────────┐
                     │    Orchestrator Agent │
                     └───────────┬───────────┘
                                 │ (Routes Sub-Tasks)
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
 ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
 │ Vision Agent  │       │ Audio Agent   │       │ Context Agent │
 └───────┬───────┘       └───────┬───────┘       └───────┬───────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    [Debate & Consensus Engine]
                                 │
                    [10-Variation Generator]
                                 │
                    ┌────────────▼────────────┐
                    │ 10 Scaled Output Variations │
                    └─────────────────────────┘
```

### Key Components:
* **Orchestrator Agent:** Deconstructs queries, determines necessary modalities, and routes tasks to specific micro-agents.
* **Worker Agents:** Specialized models dedicated to Vision & Motion, Audio & Acoustics, and Temporal Context tracking.
* **Debate & Consensus Engine:** A resolution mechanism triggered when confidence falls below 0.85 or when modalities present conflicting data (e.g., an audio transcript contradicts on-screen OCR).

### Debate & Consensus Protocol Flow
```text
                               ┌───────────────────────────┐
                               │     Orchestrator Agent    │
                               └─────────────┬─────────────┘
                                             │
                                   Initial Findings Sync
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  Confidence Score Check   │
                               └─────────────┬─────────────┘
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     │ High Confidence (>0.85)                       │ Low Confidence / Conflict
                     ▼                                               ▼
       ┌───────────────────────────┐                   ┌───────────────────────────┐
       │   Direct to Synthesizer   │                   │  Multi-Agent Debate Loop  │
       └───────────────────────────┘                   └─────────────┬─────────────┘
                                                                     │
                                                       ┌─────────────┴─────────────┐
                                                       ▼                           ▼
                                             ┌───────────────────┐       ┌───────────────────┐
                                             │ Vision Agent Argument │    │ Audio Agent Counter │
                                             └─────────┬─────────┘       └─────────┬─────────┘
                                                       │                           │
                                                       └─────────────┬─────────────┘
                                                                     │ Re-sample/Re-evaluate
                                                                     ▼
                                                       ┌───────────────────────────┐
                                                       │ Final Consensus Decision  │
                                                       └─────────────┬─────────────┘
                                                                     │
                                                                     ▼
                                                       ┌───────────────────────────┐
                                                       │   Direct to Synthesizer   │
                                                       └───────────────────────────┘
```

* **Variational Answer Synthesizer:** Generates 10 distinct formats ranging from concise factual statements to JSON schemas and chronologically detailed sequence breakdowns.

## 2. Query Routing Analysis

A secondary analysis was performed after reviewing the initial routing pass, revealing the following dataset characteristics:

| Metric / Category | Count | Percentage |
| :--- | :--- | :--- |
| **Total Queries Routed** | 2,443 | 100% |
| **Require Vision** | 2,259 | 92.5% |
| **Require Audio** | 576 | 23.6% |
| **Require OCR** | 57 | 2.3% |
| **Require Inference (Not Observed)** | 1,136 | **46.5%** |
| **Require Temporal Context** | 886 | **36.3%** |

## 3. Next Steps & Execution Strategy

Based on the routing statistics, the following actionable steps were formulated to optimize compute boundaries and prevent systemic bottlenecks:

* **Asymmetric Resource Allocation:** Heavy GPU/VRAM allocation must be prioritized for the Vision Agent, which exclusively handles 75.3% of the load. Conversely, the OCR agent should be lazy-loaded or managed via a lightweight API route since it is only needed for 2.3% of queries.
* **Inference LLM Layer:** Standard visual models cannot resolve the 46.5% of queries requiring "inference (not observed)." A central Context LLM must act as the primary deduction engine, fed purely by the extracted text/transcripts generated by the worker agents.
* **Strict Temporal Tracking:** To support the 36.3% of sequence-dependent queries, precise timestamp markers (`$t_{start}$` to `$t_{end}$`) must be rigorously applied during the offline extraction phase to anchor the consensus engine.

### Phased Execution Plan

1. **Phase 1 (The Baseline):** Run the 168 pure Audio and 10 pure OCR queries first. This validates the 10-variation output synthesizer without engaging the multi-agent debate loop.
2. **Phase 2 (The Heavy Lifter):** Process a segmented batch of pure Vision queries to profile system latency, specifically testing the threshold between the 30-second shorts and the extended >150-second videos.
3. **Phase 3 (The Debate Crucible):** Execute the complex, multi-modal (Vision + Audio) and inference-heavy queries to test cross-examination prompts and final resolution accuracy.
