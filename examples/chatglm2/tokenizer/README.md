# ChatGLM2 Tokenizer

This directory contains the ChatGLM2-6B tokenizer files downloaded from HuggingFace.

## Downloaded Files

- `tokenizer_config.json` - Tokenizer configuration
- `tokenizer.model` - SentencePiece model file
- `tokenization_chatglm.py` - Custom tokenizer implementation (requires `trust_remote_code=True`)

## Source

Downloaded from: https://huggingface.co/THUDM/chatglm2-6b

## Tokenizer Configuration

ChatGLM2-6B tokenizer properties:
- **vocab_size**: 65024 (padded)
- **eos_token_id**: 2
- **pad_token_id**: 0

## Usage

The tokenizer path is set in `env.sh` and defaults to this directory.
You can override it by setting the `TOKENIZER_MODEL` environment variable:

```bash
export TOKENIZER_MODEL=/path/to/your/tokenizer
```

## Re-download (if needed)

```bash
cd examples/chatglm2/tokenizer
python3 -c "
from huggingface_hub import hf_hub_download
repo_id = 'THUDM/chatglm2-6b'
files = ['tokenizer_config.json', 'tokenizer.model', 'tokenization_chatglm.py']
for f in files:
    hf_hub_download(repo_id=repo_id, filename=f, local_dir='.')
"
```
