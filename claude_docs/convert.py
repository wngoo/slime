import os
import re
import json
import glob
import argparse
from typing import Any, Dict, Optional

# ---- thread env must be set BEFORE importing torch for best effect ----
def preset_thread_env(intra_op_threads: int):
    os.environ.setdefault("OMP_NUM_THREADS", str(intra_op_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(intra_op_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(intra_op_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(intra_op_threads))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(intra_op_threads))

preset_thread_env(1)

import torch
import torch.distributed as dist

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Export FSDP checkpoint to HF safetensors")

    parser.add_argument("--base-model", type=str, required=True,
                        help="Base HF model name or local path")
    parser.add_argument("--fsdp-ckpt-dir", type=str, required=True,
                        help="Directory containing FSDP checkpoint shards")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output directory for HF-compatible exported model")

    parser.add_argument("--shard-pattern", type=str, default="rank{rank}.pt",
                        help="Shard filename pattern. Example: rank{rank}.pt")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Execution device")
    parser.add_argument("--backend", type=str, default=None, choices=[None, "gloo", "nccl"],
                        help="Distributed backend. Defaults to nccl for cuda, gloo for cpu")
    parser.add_argument("--max-shard-size", type=str, default="10GB",
                        help="Max shard size for save_pretrained")
    parser.add_argument("--strict", action="store_true",
                        help="Use strict=True for load_state_dict")
    parser.add_argument("--cpu-ram-efficient-loading", action="store_true",
                        help="Enable low_cpu_mem_usage-like path where applicable")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Pass trust_remote_code=True to transformers")
    parser.add_argument("--save-tokenizer-only-if-exists", action="store_true",
                        help="Do not fail if tokenizer loading/saving is unavailable")
    parser.add_argument("--local-rank", type=int, default=None,
                        help="Optional local rank override. Usually set by torchrun")

    # CPU oversubscription controls
    parser.add_argument("--intra-op-threads", type=int, default=1,
                        help="torch.set_num_threads(...) and OMP/MKL/OpenBLAS threads")
    parser.add_argument("--inter-op-threads", type=int, default=1,
                        help="torch.set_num_interop_threads(...)")
    parser.add_argument("--allow-oversubscribe", action="store_true",
                        help="Allow world_size > physical/logical CPU count with warning")

    return parser.parse_args()


def infer_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def count_available_cpus() -> int:
    return os.cpu_count() or 1


def set_torch_threads(intra_op_threads: int, inter_op_threads: int):
    torch.set_num_threads(max(1, intra_op_threads))
    try:
        torch.set_num_interop_threads(max(1, inter_op_threads))
    except RuntimeError:
        # set_num_interop_threads may fail if called too late in some runtimes
        pass


def maybe_init_dist(device_mode: str, backend: Optional[str], local_rank_arg: Optional[int]) -> int:
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = local_rank_arg
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size_env > 1:
        if backend is None:
            backend = "nccl" if device_mode == "cuda" else "gloo"
        dist.init_process_group(backend=backend)

    if device_mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA mode requested, but torch.cuda.is_available() is False.")
        torch.cuda.set_device(local_rank)

    return local_rank


def load_shard(shard_path: str) -> Dict[str, Any]:
    shard = torch.load(shard_path, map_location="cpu")

    if isinstance(shard, dict):
        for key in ["model", "state_dict", "model_state_dict", "module", "weights"]:
            if key in shard and isinstance(shard[key], dict):
                return shard[key]
        return shard

    raise ValueError(f"Unsupported checkpoint object type in {shard_path}: {type(shard)}")


def detect_shard_files(fsdp_ckpt_dir: str) -> list[str]:
    candidates = []
    for ext in ("*.pt", "*.bin", "*.pth"):
        candidates.extend(glob.glob(os.path.join(fsdp_ckpt_dir, ext)))
    return sorted(candidates)


def materialize_shard_name(pattern: str, rank: int, local_rank: int, world_size: int) -> str:
    return pattern.format(rank=rank, local_rank=local_rank, world_size=world_size)


def warn_rank0(msg: str):
    if get_rank() == 0:
        print(f"[rank0][WARN] {msg}", flush=True)


def info_rank0(msg: str):
    if get_rank() == 0:
        print(f"[rank0] {msg}", flush=True)


def main():
    args = parse_args()

    # set thread limits as early as possible
    preset_thread_env(args.intra_op_threads)
    set_torch_threads(args.intra_op_threads, args.inter_op_threads)

    device_mode = infer_device(args.device)
    local_rank = maybe_init_dist(device_mode, args.backend, args.local_rank)

    rank = get_rank()
    world_size = get_world_size()

    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)

    cpu_count = count_available_cpus()
    if device_mode == "cpu" and world_size > cpu_count and not args.allow_oversubscribe:
        raise RuntimeError(
            f"world_size={world_size} > cpu_count={cpu_count}. "
            f"This can still run, but may be very slow. "
            f"Re-run with --allow-oversubscribe if intentional."
        )

    if device_mode == "cpu" and rank == 0:
        info_rank0(
            f"CPU mode: world_size={world_size}, cpu_count={cpu_count}, "
            f"intra_op_threads={args.intra_op_threads}, inter_op_threads={args.inter_op_threads}"
        )
        if world_size > cpu_count:
            warn_rank0(
                "Oversubscribed run detected. This is allowed, but performance may degrade sharply."
            )

    shard_files = detect_shard_files(args.fsdp_ckpt_dir)
    if rank == 0:
        info_rank0(f"Detected shard-like files in checkpoint dir: {len(shard_files)}")

    if shard_files and len(shard_files) != world_size and rank == 0:
        warn_rank0(
            f"Detected {len(shard_files)} shard-like files but current world_size is {world_size}. "
            f"For raw rank-sharded torch.save checkpoints, these usually need to match."
        )

    config = AutoConfig.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
    )

    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception:
        if not args.save_tokenizer_only_if_exists:
            raise
        warn_rank0("Tokenizer could not be loaded; continuing without tokenizer export.")

    model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=args.trust_remote_code,
    )

    if device_mode == "cuda":
        device = torch.device(f"cuda:{local_rank}")
        model = model.to(device)
        fsdp_model = FSDP(model, device_id=device)
    else:
        device = torch.device("cpu")
        model = model.to(device)
        fsdp_model = FSDP(model, device_id=None)

    shard_name = materialize_shard_name(
        args.shard_pattern, rank=rank, local_rank=local_rank, world_size=world_size
    )
    shard_path = os.path.join(args.fsdp_ckpt_dir, shard_name)

    if not os.path.exists(shard_path):
        raise FileNotFoundError(
            f"Shard file not found for rank {rank}: {shard_path}\n"
            f"Check --shard-pattern and world size.\n"
            f"Detected files: {[os.path.basename(x) for x in shard_files[:10]]}"
        )

    shard_state = load_shard(shard_path)

    load_result = fsdp_model.load_state_dict(shard_state, strict=args.strict)
    if isinstance(load_result, tuple):
        missing, unexpected = load_result
    else:
        missing = getattr(load_result, "missing_keys", [])
        unexpected = getattr(load_result, "unexpected_keys", [])

    if rank == 0:
        info_rank0("load_state_dict done.")
        if not args.strict:
            info_rank0(f"missing keys: {len(missing)}")
            info_rank0(f"unexpected keys: {len(unexpected)}")

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = fsdp_model.state_dict()

    if rank == 0:
        plain_model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=args.trust_remote_code,
        )
        plain_model.load_state_dict(cpu_state, strict=args.strict)

        plain_model.save_pretrained(
            args.out_dir,
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )
        config.save_pretrained(args.out_dir)

        if tokenizer is not None:
            tokenizer.save_pretrained(args.out_dir)

        meta = {
            "base_model": args.base_model,
            "device_mode": device_mode,
            "world_size": world_size,
            "cpu_count": cpu_count,
            "intra_op_threads": args.intra_op_threads,
            "inter_op_threads": args.inter_op_threads,
            "shard_pattern": args.shard_pattern,
        }
        with open(os.path.join(args.out_dir, "export_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        info_rank0(f"Export complete: {args.out_dir}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
