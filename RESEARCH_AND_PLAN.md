# Research and Plan: Azure Chat Assistant (MCP Server)

## 1. Current Architecture Overview
*   Single-file implementation: `mcp_voice_assistant.py`
*   Language: Python 3
*   Network: `urllib.request` (no dependencies)
*   Concurrency: Thread-based stdin/stdout reader
*   Configuration: `~/.config/azure-voice-assistant/config.json`

## 2. Latency Breakdown
*   Request serialization
*   Network RTT to Azure endpoint
*   Model inference (LLM delay)
*   Stream processing delay
*   Response formatting

## 3. Performance Baselines
*   Current TTFT (Time To First Token)
*   Total response latency
*   Token throughput

## 4. Caching Strategy
*   Prompt caching (if supported by Azure AI Foundry)
*   Response caching for repetitive queries
*   Embedding cache for context management

## 5. Streaming + Async Handling
*   Transition from thread-based to `asyncio` for better I/O concurrency
*   Ensure early token delivery to reduce perceived latency

## 6. Connection Management
*   Enable HTTP keep-alive
*   Implement connection pooling to minimize TCP/TLS handshakes

## 7. Prompt/Token Efficiency
*   System prompt optimization and reuse
*   Dynamic history trimming based on token counts
*   Context window management

## 8. Observability
*   Detailed logging of TTFT and total latency per request
*   Tracing for multi-step tool calls
*   Integration with telemetry collectors

## 9. Load Testing Plan
*   Simulated concurrent sessions
*   Stress testing tool call boundaries

## 10. Optimization Roadmap
*   Phase 1: Instrumentation and baseline measurement (Completed)
*   Phase 2: Quick wins (caching, connection pooling) (Completed)
*   Phase 3: Async refactoring with httpx + producer-consumer queue (Completed)
*   Phase 4: Multi-model parallel execution — `multi_chat` via `asyncio.gather` (Completed)
*   Phase 5: Voice integration — paired with azure-speech MCP for `multi_speak` (Completed)

## 11. Current Architecture (Post-Refactor)
*   **Async core**: `asyncio` + `httpx.AsyncClient` with connection pooling
*   **Streaming**: SSE with producer-consumer `asyncio.Queue` for real-time token delivery
*   **Parallel LLM**: `multi_chat` dispatches to N models concurrently
*   **Parallel TTS**: `multi_speak` (in azure-speech server) fires N TTS requests concurrently
*   **Optimized flow**: 2 MCP calls instead of 2N (multi_chat + multi_speak)

## 12. Known Constraints
*   GPT-5.3 rejects temperature != 1.0
*   DeepSeek R1 includes `<think>` tags regardless of system prompt
*   Serverless free tier: 15 requests/day per model
*   Azure Sponsorship blocks third-party marketplace models (no Claude on Azure)
