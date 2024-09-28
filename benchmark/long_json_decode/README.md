## Run benchmark

### Benchmark coalinfer
```
python3 -m coalinfer.launch_server --model-path codellama/CodeLlama-7b-instruct-hf --port 30000
```

```
python3 bench_coalinfer.py --num-questions 5 --parallel 1
```


### Benchmark vllm
```
python3 -m vllm.entrypoints.api_server --tokenizer-mode auto --model codellama/CodeLlama-7b-instruct-hf  --disable-log-requests --port 21000 --gpu 0.97
```

```
python3 bench_other.py --backend vllm --num-questions 5
```


### Benchmark guidance
```
python3 bench_other.py --backend guidance --num-questions 5 --parallel 1 --n-ctx 11000 --model-path path/to/code-llama/gguf
```


### Build dataset
```
pip install wikipedia
python3 build_dataset.py
```
