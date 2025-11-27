python3 -m areal.launcher.local examples/math/deepscaler_grpo.py \
    --config examples/math/deepscaler_grpo.yaml \
    experiment_name=deepscaler-grpo-eagle3 \
    +sglang.speculative_algorithm=EAGLE3 \
    +sglang.speculative_draft_model_path=/home/test/testdata/models/qwen3_8b_eagle3 \
    +sglang.speculative_num_steps=1 \
    +sglang.speculative_eagle_topk=4 \
    +sglang.speculative_num_draft_tokens=4 \
    2>&1 | tee deepscaler_grpo.log