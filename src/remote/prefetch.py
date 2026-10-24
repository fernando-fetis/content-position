from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    '''Downloads the models named on the command line (default: all of models.yaml) at their revisions into the Hugging Face cache (safetensors when available, else .bin) and checks that config and tokenizer load offline, so jobs can run with HF_HUB_OFFLINE=1.'''
    load_dotenv(ROOT / '.env')
    token = os.environ.get('HF_TOKEN') or sys.exit('HF_TOKEN missing: copy .env with HF_TOKEN=... to the checkout')
    registry = yaml.safe_load((ROOT / 'src' / 'models.yaml').read_text())
    for name in sys.argv[1:] or list(registry):
        spec = registry[name]
        files = HfApi().list_repo_files(spec['hf'], revision=spec['revision'], token=token)
        skip = ['original/*', '*.pth', '*.gguf', '*.msgpack', '*.h5', *(['*.bin'] if any(f.endswith('.safetensors') for f in files) else [])]
        path = snapshot_download(spec['hf'], revision=spec['revision'], token=token, ignore_patterns=skip)
        AutoConfig.from_pretrained(spec['hf'], revision=spec['revision'], local_files_only=True)
        AutoTokenizer.from_pretrained(spec['hf'], revision=spec['revision'], local_files_only=True)
        print(f'{name}: {sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 2 ** 30:.1f} GiB, loads offline', flush=True)


if __name__ == '__main__':
    main()
