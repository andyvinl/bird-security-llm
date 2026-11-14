#!/usr/bin/env python3
"""
Clean v1 B.I.R.D. attack data: remove the "probing" turns that taught the model to
refuse ordinary questions as "reconnaissance".

For each conversation, any user/assistant pair whose assistant reply calls the
message reconnaissance (or refuses a non-attack turn) is dropped. Conversations
still start with a normal question and end with the attack + refusal.

Usage:
  python bird_clean.py bird_train.jsonl -o bird_train_clean.jsonl
"""
import argparse
import json
import re
import sys

RECON = re.compile(r"reconnaissance|probing (for|your)|hidden rules", re.I)


def clean(msgs):
    out = [msgs[0]] if msgs and msgs[0]["role"] == "system" else []
    body = msgs[len(out):]
    removed = 0
    i = 0
    while i < len(body) - 1:
        u, a = body[i], body[i + 1]
        if u["role"] == "user" and a["role"] == "assistant" and RECON.search(a["content"]):
            removed += 1
        else:
            out += [u, a]
        i += 2
    return out, removed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    a = p.parse_args()
    kept = dropped_turns = dropped_convs = 0
    with open(a.input, encoding="utf-8") as fi, open(a.output, "w", encoding="utf-8") as fo:
        for line in fi:
            if not line.strip():
                continue
            rec = json.loads(line)
            msgs, removed = clean(rec["messages"])
            dropped_turns += removed
            if sum(m["role"] == "user" for m in msgs) == 0:
                dropped_convs += 1
                continue
            rec["messages"] = msgs
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
    print(f"kept {kept} conversations, removed {dropped_turns} probing turns, "
          f"dropped {dropped_convs} empty conversations -> {a.output}")


if __name__ == "__main__":
    sys.exit(main())
