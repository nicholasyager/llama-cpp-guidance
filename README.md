# llama.cpp Guidance

[![pypi version shield](https://img.shields.io/pypi/v/llama-cpp-guidance)](https://img.shields.io/pypi/v/llama-cpp-guidance)

The `llama-cpp-guidance` package provides an LLM client compatibility layer between `llama-cpp-python` and `guidance`.

## Installation

The `llama-cpp-guidance` package can be installed using pip.

```console
pip install llama-cpp-guidance
```

## Basic Usage

Once installed, you can use the `LlamaCpp` class like any other guidance-compatible LLM class.

```python
from pathlib import Path
from llama_cpp_guidance.llm import LlamaCpp
import guidance

guidance.llm = LlamaCpp(
    model_path=Path("../path/to/llamacpp/model.gguf"),
    n_gpu_layers=1,
    n_threads=8
)

program = guidance(
    "The best thing about the beach is {{~gen 'best' temperature=0.7 max_tokens=10}}"
)
output = program()
print(output)
```

```console
The best thing about the beach is that there’s always something to do.
```
