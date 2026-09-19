from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent))
from common import (
    enabled_datasets,
    estimate_tokens,
    get_field,
    iter_jsonl,
    load_config,
    normalize_space,
    pipeline_path,
    stable_uuid,
)


def join_prompt(*parts: str) -> str:
    return "\n\n".join(part for part in (normalize_space(p) for p in parts) if part)


def normalize_record(row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any] | None:
    fmt = spec.get("format", "plain_text")
    fields = spec.get("fields", {})
    source = spec["name"]
    category = spec["category"]

    if fmt == "plain_text":
        input_text = ""
        output_text = get_field(row, fields.get("output", "text"))
    elif fmt == "instruction_response":
        input_text = join_prompt(
            get_field(row, fields.get("instruction", "instruction")),
            get_field(row, fields.get("context", "input")),
        )
        output_text = get_field(row, fields.get("output", "output"))
    elif fmt == "openorca":
        system = get_field(row, fields.get("system", "system_prompt"))
        question = get_field(row, fields.get("input", "question"))
        input_text = join_prompt(f"System: {system}" if system else "", f"User: {question}")
        output_text = get_field(row, fields.get("output", "response"))
    elif fmt == "qa":
        input_text = get_field(row, fields.get("input", "question"))
        output_text = get_field(row, fields.get("output", "answer"))
    elif fmt == "ultrachat":
        msgs = row.get("messages", [])
        if isinstance(msgs, list) and len(msgs) >= 2:
            input_parts = []
            for m in msgs[:-1]:
                role = str(m.get("role", "user")).capitalize()
                content = normalize_space(str(m.get("content", "")))
                if content:
                    input_parts.append(f"{role}: {content}")
            input_text = "\n".join(input_parts)
            output_text = normalize_space(str(msgs[-1].get("content", "")))
        else:
            input_text = get_field(row, fields.get("instruction", "instruction"))
            output_text = get_field(row, fields.get("output", "output"))
    elif fmt == "glaive_tool":
        system_text = normalize_space(get_field(row, fields.get("input", "system")))
        chat_text = normalize_space(get_field(row, fields.get("output", "chat")))
        if "ASSISTANT:" in chat_text:
            parts = chat_text.split("ASSISTANT:", 1)
            user_msg = parts[0].replace("USER:", "").strip()
            assist_msg = parts[1].strip()
            input_text = f"System: {system_text}\nUser: {user_msg}" if system_text else user_msg
            output_text = assist_msg
        else:
            input_text = system_text
            output_text = chat_text
    elif fmt == "hermes_tool":
        convs = row.get("conversations", [])
        if isinstance(convs, list) and len(convs) >= 2:
            input_parts = []
            for m in convs[:-1]:
                role = "User" if m.get("from") in {"human", "user"} else "Assistant"
                val = normalize_space(str(m.get("value", "")))
                if val:
                    input_parts.append(f"{role}: {val}")
            input_text = "\n".join(input_parts)
            output_text = normalize_space(str(convs[-1].get("value", "")))
        else:
            input_text = get_field(row, fields.get("input"))
            output_text = get_field(row, fields.get("output"))
    elif fmt == "auto_text":
        input_text = get_field(row, fields.get("input"))
        output_text = get_field(row, fields.get("output"))
        if not output_text:
            visible = {
                key: value
                for key, value in row.items()
                if not key.startswith("_") and isinstance(value, (str, int, float, bool, list, dict))
            }
            output_text = json.dumps(visible, ensure_ascii=False, sort_keys=True)
    elif fmt == "oasst1_flat":
        # OASST1 is a message tree. This fallback keeps individual assistant
        # messages only; use a dedicated reconstruction pass before enabling it
        # for higher-quality chat examples.
        if row.get("role") not in {"assistant", "prompter"}:
            return None
        input_text = ""
        output_text = row.get("text", "")
    else:
        input_text = get_field(row, fields.get("input"))
        output_text = get_field(row, fields.get("output"))

    input_text = normalize_space(input_text)
    output_text = normalize_space(output_text)
    if not output_text:
        return None

    text_for_id = {"input": input_text, "output": output_text, "source": source}
    combined = join_prompt(input_text, output_text)
    return {
        "id": stable_uuid(source, text_for_id),
        "source": source,
        "category": category,
        "input": input_text,
        "output": output_text,
        "meta": {
            "lang": "unknown",
            "n_tokens_est": estimate_tokens(combined),
            "format": fmt,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dataset_builder/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = pipeline_path(config, "raw_dir")
    interim_dir = pipeline_path(config, "interim_dir")
    interim_dir.mkdir(parents=True, exist_ok=True)
    out_path = interim_dir / "02_normalized.jsonl"

    counts: dict[str, int] = {}
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        for spec in enabled_datasets(config):
            raw_path = raw_dir / f"{spec['name']}.jsonl"
            if not raw_path.exists():
                print(f"[normalize] skip missing {raw_path}")
                continue
            for row in iter_jsonl(raw_path):
                example = normalize_record(row, spec)
                if example is None:
                    continue
                out.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")
                counts[spec["name"]] = counts.get(spec["name"], 0) + 1

    print(f"[normalize] wrote {sum(counts.values())} examples to {out_path}")


if __name__ == "__main__":
    main()
