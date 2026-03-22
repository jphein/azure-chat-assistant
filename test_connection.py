#!/usr/bin/env python3
"""Test script for Azure AI Foundry model connections."""

import json
import os
import time
import urllib.request

CONFIG_PATH = os.path.expanduser("~/.config/cloud-chat-assistant/config.json")


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def test_deployed_model(cfg):
    """Test the deployed GPT model via OpenAI-compatible endpoint."""
    payload = json.dumps({
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_completion_tokens": 64
    }).encode()

    url = f"{cfg['endpoint']}/openai/deployments/{cfg['deployment']}/chat/completions?api-version=2024-12-01-preview"
    req = urllib.request.Request(url, data=payload, headers={
        "api-key": cfg["api_key"],
        "Content-Type": "application/json"
    })

    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=30)
    latency = (time.perf_counter() - t0) * 1000
    data = json.loads(resp.read().decode())

    text = data["choices"][0]["message"]["content"]
    model = data.get("model", "?")
    usage = data.get("usage", {})
    print(f"[Deployed] {cfg['deployment']}")
    print(f"  Model:   {model}")
    print(f"  Reply:   {text}")
    print(f"  Tokens:  {usage.get('prompt_tokens', '?')}\u2192{usage.get('completion_tokens', '?')}")
    print(f"  Latency: {latency:.0f}ms")


def test_serverless_model(cfg, model_name):
    """Test a serverless marketplace model via unified inference endpoint."""
    payload = json.dumps({
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "model": model_name,
        "max_tokens": 64
    }).encode()

    url = f"{cfg['endpoint']}/models/chat/completions?api-version=2024-05-01-preview"
    req = urllib.request.Request(url, data=payload, headers={
        "api-key": cfg["api_key"],
        "Content-Type": "application/json"
    })

    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=30)
    latency = (time.perf_counter() - t0) * 1000
    data = json.loads(resp.read().decode())

    text = data["choices"][0]["message"]["content"]
    model = data.get("model", "?")
    usage = data.get("usage", {})
    print(f"[Serverless] {model_name}")
    print(f"  Model:   {model}")
    print(f"  Reply:   {text}")
    print(f"  Tokens:  {usage.get('prompt_tokens', '?')}\u2192{usage.get('completion_tokens', '?')}")
    print(f"  Latency: {latency:.0f}ms")


def main():
    cfg = load_config()
    print("=== Azure AI Foundry Connection Test ===\n")

    # Test deployed model
    try:
        test_deployed_model(cfg)
    except Exception as e:
        print(f"[Deployed] {cfg.get('deployment', '?')}: FAILED - {e}")
    print()

    # Test serverless models
    for model in ["grok-3", "Meta-Llama-3.1-405B-Instruct", "Phi-4"]:
        try:
            test_serverless_model(cfg, model)
        except Exception as e:
            print(f"[Serverless] {model}: FAILED - {e}")
        print()


if __name__ == "__main__":
    main()
