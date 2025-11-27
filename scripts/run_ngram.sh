python3 -m areal.launcher.local examples/math/deepscaler_grpo.py \
    --config examples/math/deepscaler_grpo.yaml \
    experiment_name=deepscaler-grpo-ngram \
    +sglang.speculative_algorithm=NGRAM \
    +sglang.speculative_num_draft_tokens=16 \
    +sglang.speculative_ngram_min_match_window_size=1 \
    +sglang.speculative_ngram_max_match_window_size=16 \
    +sglang.speculative_ngram_min_bfs_breadth=1 \
    +sglang.speculative_ngram_max_bfs_breadth=10 \
    +sglang.speculative_ngram_branch_length=18 \
    2>&1 | tee deepscaler_grpo.log