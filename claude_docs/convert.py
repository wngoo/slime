import os
import json
import argparse
from typing import Any, Dict

import torch
import torch.distributed as dist

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", type=str, required=True)
    p.add_argument("--ckpt-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--strict", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def infer_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def load_model_shard(ckpt_dir: str, world_size: int, rank: int) -> Dict[str, Any]:
    path = os.path.join(ckpt_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict):
        for key in ["model", "state_dict", "model_state_dict", "module", "weights"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        return obj

    raise ValueError(f"Unsupported shard object type: {type(obj)} from {path}")


def main():
    args = parse_args()
    device_mode = infer_device(args.device)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        backend = "nccl" if device_mode == "cuda" else "gloo"
        dist.init_process_group(backend=backend)

    if device_mode == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
    )

    model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=args.trust_remote_code,
    ).to(device)

    fsdp_model = FSDP(model, device_id=device if device.type == "cuda" else None)

    shard_state = load_model_shard(args.ckpt_dir, world_size=world_size, rank=rank)
    load_result = fsdp_model.load_state_dict(shard_state, strict=args.strict)

    if rank == 0:
        missing = getattr(load_result, "missing_keys", [])
        unexpected = getattr(load_result, "unexpected_keys", [])
        print(f"[rank0] missing={len(missing)} unexpected={len(unexpected)}")

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
            safe_serialization=True,   # -> safetensors
            max_shard_size="10GB",
        )
        tokenizer.save_pretrained(args.out_dir)
        config.save_pretrained(args.out_dir)

        with open(os.path.join(args.out_dir, "export_meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "base_model": args.base_model,
                    "world_size": world_size,
                    "device": device_mode,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"[rank0] done: {args.out_dir}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
