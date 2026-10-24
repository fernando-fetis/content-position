from __future__ import annotations

import io
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/repo')
GPU = 'L40S'
app = modal.App('content-position-rope')
image = modal.Image.debian_slim(python_version='3.12').pip_install_from_requirements(str(ROOT / 'requirements.txt')).add_local_dir(str(ROOT), '/repo', ignore=['.venv', '.git', '.env', '.temp', 'logs', 'experiments/*/results', 'docs', '**/__pycache__'])
cache = modal.Volume.from_name('content-position-rope-hf', create_if_missing=True)
secrets = [modal.Secret.from_dotenv(str(ROOT))]


@app.function(image=image, cpu=2, memory=4096, timeout=3600, volumes={'/hf': cache}, secrets=secrets)
def prefetch(name: str) -> str:
    '''Downloads one model of models.yaml into the shared Hugging Face cache, as src/remote/prefetch.py does on the cluster, and checks that it loads offline.'''
    import os
    os.environ['HF_HOME'] = '/hf'
    import yaml
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoConfig, AutoTokenizer
    spec = yaml.safe_load((Path('/repo') / 'src' / 'models.yaml').read_text())[name]
    token = os.environ['HF_TOKEN']
    files = HfApi().list_repo_files(spec['hf'], revision=spec['revision'], token=token)
    skip = ['original/*', '*.pth', '*.gguf', '*.msgpack', '*.h5', *(['*.bin'] if any(f.endswith('.safetensors') for f in files) else [])]
    path = snapshot_download(spec['hf'], revision=spec['revision'], token=token, ignore_patterns=skip)
    AutoConfig.from_pretrained(spec['hf'], revision=spec['revision'], local_files_only=True)
    AutoTokenizer.from_pretrained(spec['hf'], revision=spec['revision'], local_files_only=True)
    cache.commit()
    return f'{name}: {sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 2 ** 30:.1f} GiB'


@app.function(image=image, gpu=GPU, cpu=2, memory=20480, timeout=24 * 3600, volumes={'/hf': cache}, secrets=secrets)
def compute(experiment: str, args: list[str], inputs: dict[str, bytes] | None = None) -> bytes:
    '''Runs the compute stage of one experiment offline on the shared cache and returns its results folder as a gzipped tar; inputs (paths relative to the repository and their bytes) are written first, for results the stage reads.'''
    import os
    os.environ.update(HF_HOME='/hf', HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1')
    for rel, data in (inputs or {}).items():
        (Path('/repo') / rel).parent.mkdir(parents=True, exist_ok=True)
        (Path('/repo') / rel).write_bytes(data)
    subprocess.run([sys.executable, 'src/run.py', experiment, 'compute', *args], cwd='/repo', check=True)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        tar.add(Path('/repo/experiments') / Path(experiment).name / 'results', arcname='results')
    return buf.getvalue()


@app.local_entrypoint()
def main(experiment: str, args: str = '', gpu: str = GPU, out: str = '', inputs: str = '') -> None:
    '''inputs: comma-separated globs of local files (relative to the repository) sent to the container.'''
    files = {str(f.relative_to(ROOT)): f.read_bytes() for g in inputs.split(',') if g for f in sorted(ROOT.glob(g))}
    data = compute.with_options(gpu=gpu).remote(experiment, shlex.split(args), files)
    dest = Path(out) if out else ROOT / '.temp' / 'pilot' / experiment
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
        tar.extractall(dest, filter='data')
    print(f'results in {dest}/results')


@app.local_entrypoint()
def download(models: str) -> None:
    '''Downloads the comma-separated models in parallel, one CPU container each.'''
    for line in prefetch.map(models.split(','), return_exceptions=True):
        print(line, flush=True)
