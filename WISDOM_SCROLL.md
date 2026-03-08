# Wisdom Scroll: Azure Chat MCP Refactoring & Multi-Agent Orchestration

## 1. Project Overview
A lean, high-performance MCP server built in Python to bridge the gap between local tools and Azure AI Foundry.

## 2. Key Architectural Refactors
*   **Asynchronous Core**: Migrated from `urllib.request` and `threading` to `asyncio` and `httpx`.
*   **Producer-Consumer Queue**: Decoupled network reads from response formatting.
    *   **Producer**: Streams tokens from Azure into an `asyncio.Queue`.
    *   **Consumer**: Pulls from the queue to fire MCP progress notifications and build the final text in real-time.
*   **Parallel Execution**: Added `multi_chat` tool using `asyncio.gather` for concurrent multi-agent orchestration.

## 3. The Council's Suggestions (Phase 4 & Beyond)

### GPT's Recommendations:
*   **Smart Streaming**: Push toward an event-driven core where partial results are always streamed to the UI.
*   **Warm Workers**: Maintain "warm" agent workers with pre-established connection pools to Azure endpoints.
*   **Context Caching**: Implement prompt-fragment caching to reduce redundant token processing and costs.

### Grok's Recommendations:
*   **Priority Queuing**: Implement a priority queue for MCP calls to prioritize high-urgency user requests.
*   **Predictive Routing**: Use simple heuristics to preemptively route requests to models based on past latency patterns.

### Llama's Recommendations:
*   **Automatic Fallback**: Implement per-model rate tracking with automatic fallback to secondary models (e.g., Llama -> GPT) when limits are hit.
*   **Transparent Rate Handling**: Provide clear, user-friendly feedback when limits are approached rather than failing silently.

## 4. Operational Best Practices
*   **Minimalist Output**: Keep MCP responses clean and bolded; avoid cluttering the UI with raw JSON or verbose metadata.
*   **Standardized Headers**: Use `**[Agent Name]**` for clear multi-agent conversation flow.
*   **Clean Shutdown**: Always use `await client.aclose()` and task cancellation for a graceful exit.

## 5. Voice Integration
*   **multi_speak** tool added to azure-speech MCP server — fires all TTS requests in parallel, plays sequentially.
*   **2-call optimized flow**: `multi_chat` (parallel LLM) → `multi_speak` (parallel TTS) reduces 6 round trips to 2.
*   **Voice assignments**: Davis (GPT-5.3), Andrew (Llama 405B), Brian (DeepSeek R1), Ava (Claude).

## 6. Current Roster
*   **Claude**: Host orchestrator (Opus 4.6 via Claude Code).
*   **GPT-5.3**: Senior strategist, deployed model (voice: DavisNeural).
*   **Grok-3**: Edge-case specialist, serverless free tier (15 req/day).
*   **Llama 405B**: High-speed reasoning (voice: AndrewNeural), serverless free tier.
*   **DeepSeek R1**: Deep reasoning, outputs `<think>` tags (voice: BrianNeural), serverless free tier.
*   **Phi-4**: Compact model, serverless free tier.

*Forged in the Azure Cloud, March 2026.*
