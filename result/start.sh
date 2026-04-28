#!/bin/bash

export CUDA_VISIBLE_DEVICES=0


python -m vllm.entrypoints.openai.api_server \
    --model ../Qwen/Qwen3-4B \
    --enable-lora \
    --lora-modules trained_lora=./output \
    --port 8080 \
    --host 0.0.0.0 \
    --max-model-len 1024 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    2>&1 | tee server.log