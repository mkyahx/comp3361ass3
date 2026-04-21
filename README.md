# COMP3361: LLM Reasoning and Agent Implementation

This repository contains the implementation and evaluation of various Large Language Model (LLM) techniques, ranging from fundamental decoding strategies to advanced Agentic workflows using Chain-of-Thought (CoT) and ReAct.

---

## Project Overview

The assignment is divided into three key sections:

### Section 1: Comparative Analysis of Decoding Methods
We evaluate how different decoding strategies affect the quality and characteristics of generated text. We compare five distinct methods:
* **Greedy Search**: Deterministic selection of the highest-probability token.
* **Vanilla Sampling**: Unrestricted random sampling from the model's distribution.
* **Temperature Scaling**: Adjusting the confidence of the distribution to balance creativity and coherence.
* **Top-k Sampling**: Filtering the top $k$ most likely tokens.
* **Top-p (Nucleus) Sampling**: Dynamic filtering based on cumulative probability mass.

**Metrics:** Perplexity (PPL), Fluency, Diversity, and Repetition Rate.

### Section 2: Reasoning with Few-Shot Chain-of-Thought (CoT)
We implement a **Few-Shot CoT Agent** and evaluate its performance on the **ARC-Challenge** dataset. This section demonstrates how providing intermediate reasoning steps (rationales) enhances the model's ability to solve complex, grade-school-level science questions compared to direct prompting.

### Section 3: ReAct Agent Implementation
We develop a **ReAct (Reasoning and Acting)** Agent capable of using external tools to solve multi-step problems. 
* **Tools**: `Calculator` (for precise arithmetic) and `Wikipedia API` (for factual retrieval).
* **Benchmarks**: Evaluated on the **MATH** and **GAIA** datasets.
* **Comparison**: A comparative study between the "Vanilla" model (closed-book) and the tool-augmented "ReAct" agent.

---

## Experimental Results

The following table summarizes the performance across all three sections:

| Section | Task | Method | Metric | Score |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Decoding** | Greedy | PPL / Fluency / Diversity / Repetition | `1.92` / `1.000` / `0.045` / `0.755` |
| **1** | **Decoding** | Vanilla Sampling | PPL / Fluency / Diversity / Repetition | `50.08` / `0.280` / `0.992` / `0.005` |
| **1** | **Decoding** | Temperature (0.8) | PPL / Fluency / Diversity / Repetition | `13.98` / `0.800` / `0.977` / `0.003` |
| **1** | **Decoding** | Top-k (k=20) | PPL / Fluency / Diversity / Repetition | `11.69` / `0.880` / `0.969` / `0.006` |
| **1** | **Decoding** | Top-p (p=0.7) | PPL / Fluency / Diversity / Repetition | `11.53` / `0.800` / `0.960` / `0.019` |
| **2** | **ARC-Challenge** | Few-shot Direct | Accuracy | **0.28** |
| **2** | **ARC-Challenge** | Few-shot CoT | Accuracy | **0.82** |
| **3** | **MATH** | Vanilla / ReAct | Accuracy | **0.20 / 0.40** |
| **3** | **GAIA** | Vanilla / ReAct | Accuracy | **0.06 / 0.13** |

---

## Key Observations

1.  **Decoding**: While **Greedy Search** achieves high fluency, it suffers from a massive repetition rate (>75%). **Top-k** and **Top-p** provide a superior balance between coherence and creativity.
2.  **Reasoning**: Implementing **Chain-of-Thought** led to a nearly **3x improvement** in accuracy on the ARC-Challenge dataset, highlighting the power of explicit reasoning paths.
3.  **Agents**: The **ReAct** framework effectively doubled the performance on both MATH and GAIA benchmarks by mitigating the model's limitations in calculation and real-time knowledge retrieval.
