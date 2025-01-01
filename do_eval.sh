#!/bin/bash
lm_eval --model hf \
    --model_args "pretrained=orionweller/test-flex-gpt,revision=a00859f3a08c2d59d07dc0ed55d13bf1c4e52b13" \
    --tasks lambada_openai,hellaswag,openbookqa,arc_easy,winogrande,arc_challenge,piqa,boolq \
    --device cuda:0 \
    --batch_size auto:4 \
    --log_samples \
    --output_path results \
    --trust_remote_code

# --model_args pretrained=EleutherAI/pythia-160m,dtype="float16" \
