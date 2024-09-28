# How to Support a New Model

To support a new model in Coalinfer, you only need to add a single file under [Coalinfer Models Directory](https://github.com/sgl-project/coalinfer/tree/main/python/coalinfer/srt/models). You can learn from existing model implementations and create new files for the new models. Most models are based on the transformer architecture, making them very similar.

Another valuable resource is the [vLLM Models Directory](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models). vLLM has extensive coverage of models, and Coalinfer has reused vLLM for most parts of the model implementations. This similarity makes it easy to port many models from vLLM to Coalinfer.

To port a model from vLLM to Coalinfer, you can compare these two files [Coalinfer LLaMA Implementation](https://github.com/sgl-project/coalinfer/blob/main/python/coalinfer/srt/models/llama2.py) and [vLLM LLaMA Implementation](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py). This comparison will help you understand how to convert a model implementation from vLLM to Coalinfer. The major difference is the replacement of PagedAttention with RadixAttention. The other parts are almost identical. Specifically,
  - Replace vllm's `Attention` with `RadixAttention`. Note that you need to pass `layer_id` all the way to `RadixAttention`.
  - Replace vllm's `LogitsProcessor` with Coalinfer's `LogitsProcessor`.
  - Remove `Sample`.
  - Change `forward()` functions, and add `input_metadata`.
  - Add `EntryClass` at the end.
  - Test correctness by comparing the final logits and outputs of the two following commands:
    - `python3 scripts/playground/reference_hf.py --model [new model]`
    - `python3 -m coalinfer.bench_latency --model [new model] --correct --output-len 16 --trust-remote-code`
  - Update [Supported Models](https://github.com/sgl-project/coalinfer/tree/main?tab=readme-ov-file#supported-models) at [README](../README.md).
