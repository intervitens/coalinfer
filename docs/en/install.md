## Install Coalinfer

You can install Coalinfer using any of the methods below.

```
git clone https://github.com/sgl-project/coalinfer.git
cd coalinfer

pip install --upgrade pip
pip install -e "python[all]"

# Install FlashInfer CUDA kernels
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

### Common Notes
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) is currently one of the dependencies that must be installed for Coalinfer. It only supports sm75 and above. If you encounter any FlashInfer-related issues on sm75+ devices (e.g., T4, A10, A100, L4, L40S, H100), consider using Triton's kernel by `--disable-flashinfer --disable-flashinfer-sampling` and raise an issue.
