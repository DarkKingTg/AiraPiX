from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import torch

from airapix.training.config import load_training_config
from airapix.training.dataset import iter_jsonl


def build_prompt(row: dict) -> str | None:
    if row.get("input"):
        return row["input"]
    if row.get("category") in {"reasoning", "chat", "tool_use"}:
        return row.get("output", "")[:1000]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Missing dependency: transformers. Run `pip install -r requirements.txt`.") from exc

    config = load_training_config(args.config)
    distill_cfg = config["distillation"]
    model_name = distill_cfg.get("teacher_model", "Qwen/Qwen2.5-7B-Instruct")
    print(f"[distill] loading teacher: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model_kwargs = {
        "torch_dtype": compute_dtype,
        "device_map": distill_cfg.get("device_map", "auto"),
        "trust_remote_code": True,
    }
    if bool(distill_cfg.get("load_in_4bit", True)):
        try:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        except Exception:
            model_kwargs["load_in_4bit"] = True
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    target_device = next(model.parameters()).device
    print(
        json.dumps(
            {
                "teacher": model_name,
                "target_device": str(target_device),
                "compute_dtype": str(compute_dtype),
                "input": args.input,
                "output": args.output,
                "limit": args.limit,
            },
            indent=2,
        ),
        flush=True,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    started = time.time()
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        for idx, row in enumerate(iter_jsonl(args.input)):
            if args.limit is not None and idx >= args.limit:
                break
            prompt = build_prompt(row)
            if not prompt:
                continue
            messages = [
                {"role": "system", "content": "You are a clear, direct, helpful English AI companion teacher."},
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt")
            inputs = {key: value.to(target_device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=int(distill_cfg.get("max_new_tokens", 512)),
                    do_sample=True,
                    temperature=float(distill_cfg.get("temperature", 0.7)),
                    top_p=float(distill_cfg.get("top_p", 0.9)),
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = outputs[0][inputs.input_ids.shape[-1] :]
            answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
            if not answer:
                continue
            example = {
                "id": str(uuid.uuid4()),
                "source": model_name,
                "category": row.get("category", "chat"),
                "input": prompt,
                "output": answer,
                "meta": {
                    "lang": "en",
                    "teacher": model_name,
                    "distillation_type": "sequence",
                },
            }
            out.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")
            written += 1
            if written % max(1, args.log_every) == 0:
                elapsed = max(1e-6, time.time() - started)
                print(
                    f"[distill] wrote={written} elapsed_min={elapsed / 60:.1f} "
                    f"examples_per_min={written / elapsed * 60:.2f}",
                    flush=True,
                )
    print(f"[distill] wrote {written} examples to {out_path}", flush=True)


if __name__ == "__main__":
    main()
