#!/usr/bin/env python3
"""
B.I.R.D. fine-tuning with Unsloth (run on the NVIDIA GPU box, not the Mac).

Combines the adversarial dataset (bird_datagen.py) with the Security+ knowledge
dataset (bird_secplus_gen.py), trains a LoRA adapter, scores the model on the
held-out Security+ multiple-choice set before and after training, and can export
a GGUF file you can run in Ollama on your Mac.

Setup (Ubuntu + CUDA GPU):
  python3 -m venv ~/bird-env && source ~/bird-env/bin/activate
  pip install unsloth

Usage:
  python bird_train.py --data bird_train.jsonl bird_data/secplus_train.jsonl \
                       --eval bird_data/secplus_eval.jsonl --gguf
  python bird_train.py --eval-only --adapter bird-lora      # score a saved adapter
  python bird_train.py --check-data --data ...                # validate files, no GPU needed

v3 (Qwen3-4B-Instruct-2507, multi-format log data):
  python bird_train.py --data bird_train_clean.jsonl bird_data/train_run2/secplus_train.jsonl bird_data/logs_v3.jsonl \
      --eval bird_data/eval_run/secplus_eval.jsonl --base-model unsloth/Qwen3-4B-Instruct-2507 --chat-template native \
      --max-seq-len 4096 --batch-size 1 --grad-accum 16 --epochs 1 --adapter bird-v3-lora --gguf
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROLES = {"system", "user", "assistant"}

MCQ_PROMPT = ("Answer the following CompTIA Security+ question. Reply with only the letter of the "
              "correct option.\n\n{question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}\n\nAnswer:")


# --------------------------------------------------------------------------- #
# Data (pure Python, testable without a GPU)
# --------------------------------------------------------------------------- #

def load_conversations(paths: list[str], max_per_file: int | None, seed: int) -> list[list[dict]]:
    convs: list[list[dict]] = []
    rng = random.Random(seed)
    for p in paths:
        rows = []
        with open(p, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                if not line.strip():
                    continue
                msgs = json.loads(line).get("messages")
                err = check_messages(msgs)
                if err:
                    print(f"  skip {p}:{ln}: {err}")
                    continue
                rows.append([{"role": m["role"], "content": m["content"]} for m in msgs])
        if max_per_file and len(rows) > max_per_file:
            rows = rng.sample(rows, max_per_file)
        print(f"  {p}: {len(rows)} conversations")
        convs += rows
    rng.shuffle(convs)
    return convs


def check_messages(msgs) -> str:
    if not isinstance(msgs, list) or len(msgs) < 2:
        return "messages missing or too short"
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") not in ROLES or not isinstance(m.get("content"), str):
            return "bad message entry"
    roles = [m["role"] for m in msgs if m["role"] != "system"]
    if roles[0] != "user" or roles[-1] != "assistant":
        return "must start with user and end with assistant"
    if any(a == b for a, b in zip(roles, roles[1:])):
        return "roles must alternate"
    return ""


def load_mcq(path: str, limit: int | None) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    rows = [r for r in rows if set(r.get("options", {})) == {"A", "B", "C", "D"} and r.get("answer") in "ABCD"]
    return rows[:limit] if limit else rows


def extract_letter(text: str) -> str | None:
    m = re.search(r"\b([ABCD])\b", text.strip().upper())
    return m.group(1) if m else None


def score(results: list[tuple[str, bool]]) -> dict:
    by = defaultdict(lambda: [0, 0])
    for dom, ok in results:
        by[dom][0] += ok
        by[dom][1] += 1
    total = sum(v[0] for v in by.values()), sum(v[1] for v in by.values())
    out = {d: round(100 * c / n, 1) for d, (c, n) in sorted(by.items())}
    out["OVERALL"] = round(100 * total[0] / max(total[1], 1), 1)
    return out


# --------------------------------------------------------------------------- #
# GPU parts
# --------------------------------------------------------------------------- #

def evaluate(model, tokenizer, mcq: list[dict], label: str, max_new_tokens: int = 4) -> dict:
    import torch
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    results = []
    t0 = time.time()
    for q in mcq:
        prompt = MCQ_PROMPT.format(question=q["question"], **q["options"])
        ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                            add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        reply = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        results.append((q.get("domain", "unknown"), extract_letter(reply) == q["answer"]))
    s = score(results)
    print(f"\n[{label}] Security+ MCQ accuracy on {len(mcq)} held-out questions ({time.time()-t0:.0f}s)")
    for k, v in s.items():
        print(f"  {k:52} {v:5.1f}%")
    return s


def build_trainer(model, tokenizer, dataset, a):
    from trl import SFTConfig, SFTTrainer
    cfg = SFTConfig(
        output_dir=a.output_dir, dataset_text_field="text", max_seq_length=a.max_seq_len,
        per_device_train_batch_size=a.batch_size, gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=a.epochs, learning_rate=a.lr, warmup_ratio=0.05, lr_scheduler_type="cosine",
        logging_steps=10, save_strategy="epoch", optim="adamw_8bit", weight_decay=0.01,
        seed=a.seed, report_to="none", packing=False,
    )
    kwargs = dict(model=model, train_dataset=dataset, args=cfg)
    # TRL renamed `tokenizer` to `processing_class`; support both.
    if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    return SFTTrainer(**kwargs)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fine-tune B.I.R.D. with Unsloth LoRA.")
    p.add_argument("--data", nargs="+", default=["bird_train.jsonl", "bird_data/secplus_train.jsonl"])
    p.add_argument("--eval", default="bird_data/secplus_eval.jsonl")
    p.add_argument("--eval-limit", type=int, default=300)
    p.add_argument("--base-model", default="unsloth/Qwen2.5-1.5B-Instruct",
                   help="e.g. unsloth/Qwen2.5-3B-Instruct for more knowledge if the GPU has 12GB+")
    p.add_argument("--chat-template", default="qwen-2.5",
                   help="Unsloth template name, or 'native' to keep the model's own (use native for Qwen3-*-Instruct-2507)")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--max-per-file", type=int, default=None, help="cap conversations taken from each file")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--output-dir", default="bird-outputs")
    p.add_argument("--adapter", default="bird-lora", help="where to save / load the LoRA adapter")
    p.add_argument("--gguf", action="store_true", help="also export a q4_k_m GGUF for Ollama")
    p.add_argument("--skip-baseline", action="store_true", help="don't score the base model first")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--check-data", action="store_true", help="validate data files and exit (no GPU)")
    a = p.parse_args(argv)

    print("Loading data")
    convs = load_conversations(a.data, a.max_per_file, a.seed) if not a.eval_only else []
    mcq = load_mcq(a.eval, a.eval_limit) if Path(a.eval).exists() else []
    print(f"  total: {len(convs)} training conversations, {len(mcq)} eval questions")
    if a.check_data:
        return 0 if (convs or a.eval_only) else 1
    if not convs and not a.eval_only:
        sys.exit("No valid training data.")

    from unsloth import FastLanguageModel                      # import before transformers/trl
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from datasets import Dataset

    model_name = a.adapter if a.eval_only else a.base_model
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=model_name, max_seq_length=a.max_seq_len,
                                                         load_in_4bit=True, dtype=None)
    if a.chat_template != "native":
        tokenizer = get_chat_template(tokenizer, chat_template=a.chat_template)

    if a.eval_only:
        if not mcq:
            sys.exit(f"No eval file at {a.eval}")
        evaluate(model, tokenizer, mcq, f"adapter {a.adapter}")
        return 0

    baseline = evaluate(model, tokenizer, mcq, "base model") if mcq and not a.skip_baseline else None

    model = FastLanguageModel.get_peft_model(
        model, r=a.lora_r, lora_alpha=a.lora_r, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=a.seed)

    texts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False) for c in convs]
    too_long = sum(len(tokenizer(t).input_ids) > a.max_seq_len for t in texts[:500])
    if too_long:
        print(f"  note: ~{too_long/5:.0f}% of samples exceed {a.max_seq_len} tokens and will be truncated")
    dataset = Dataset.from_dict({"text": texts})

    trainer = build_trainer(model, tokenizer, dataset, a)
    # Learn only from B.I.R.D.'s replies, not from the attacker's messages.
    trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n",
                                      response_part="<|im_start|>assistant\n")
    print(f"\nTraining on {len(dataset)} conversations")
    stats = trainer.train()
    print(f"  train loss {stats.training_loss:.4f}, {stats.metrics.get('train_runtime', 0)/60:.1f} min")

    model.save_pretrained(a.adapter)
    tokenizer.save_pretrained(a.adapter)
    print(f"Saved LoRA adapter to {a.adapter}/")

    if mcq:
        after = evaluate(model, tokenizer, mcq, "fine-tuned B.I.R.D.")
        if baseline:
            print(f"\n  change: {after['OVERALL'] - baseline['OVERALL']:+.1f} points overall "
                  f"(aim for 80%+; the real exam's 750/900 is a scaled score, not a percentage)")

    if a.gguf:
        print("\nExporting GGUF (q4_k_m) for Ollama...")
        model.save_pretrained_gguf(a.adapter + "-gguf", tokenizer, quantization_method="q4_k_m")
        print(f"Done. Copy {a.adapter}-gguf/*.gguf to your Mac and create an Ollama model from it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
