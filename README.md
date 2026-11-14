# B.I.R.D. — Behavioral Injection & Request Defense

A small security-focused LLM, fine-tuned to spot prompt injections, refuse to obey instructions hidden inside data (like an audit log), and answer CompTIA Security+-level questions. It runs fully offline on a laptop through [Ollama](https://ollama.com).

- **Base model:** Qwen2.5-3B-Instruct, fine-tuned with LoRA using [Unsloth](https://github.com/unslothai/unsloth)
- **Training hardware:** AWS EC2 instance with a Tesla T4 GPU (16 GB)
- **Training data:** about 3,850 synthetic conversations built from NIST SP 800 / CSF 2.0, MITRE ATT&CK / CAPEC / CWE, OWASP Cheat Sheets and the CISA KEV catalog
- **Output:** a 4-bit GGUF file (Q4_K_M) served by Ollama

## How it works

```
public sources ──> bird_sources.py ──> generators (Ollama, qwen2.5) ──> bird_clean.py ──> bird_train.py ──> GGUF ──> Ollama
                                       bird_datagen.py    attacks + defenses
                                       bird_secplus_gen.py Security+ Q&A (+ held-out test set)
                                       bird_logs_gen.py    audit logs with hidden injections
```

| Script | What it does |
|---|---|
| `scripts/bird_sources.py` | Downloads and chunks the public security sources |
| `scripts/bird_datagen.py` | Generates prompt-injection attacks and model responses (direct, indirect, multi-turn, obfuscated) |
| `scripts/bird_secplus_gen.py` | Generates Security+ (SY0-701) practice Q&A plus a held-out multiple-choice test set |
| `scripts/bird_logs_gen.py` | Generates audit-log analysis tasks, some with instructions hidden inside the log |
| `scripts/bird_clean.py` | Removes training pairs that taught the model to refuse normal questions |
| `scripts/bird_train.py` | LoRA fine-tune, before/after scoring by Security+ domain, GGUF export |
| `scripts/bird_chat.py` | Terminal chat client with the B.I.R.D. system prompt |
| `scripts/bird_logs_gen_v3.py` | v3 data: six log formats, 13 scenarios, ground-truth answers, verified MITRE IDs, secret masking |
| `scripts/bird_eval.py` | Behavior eval through Ollama: injection resistance, verdicts, findings, MITRE validity, actions, refusals |
| `test_data/` | Synthetic "Northwind" audit logs for manual testing (one hides an injection) |

Step-by-step commands are in [`scripts/USAGE.md`](scripts/USAGE.md).

## Results (v2)

Scored on 267 held-out Security+-style questions that never appear in training:

| SY0-701 domain | Base Qwen2.5-3B | B.I.R.D. v2 |
|---|---|---|
| 1. General security concepts | 75.9% | 82.8% |
| 2. Threats, vulnerabilities & mitigations | 76.3% | 76.3% |
| 3. Security architecture | 75.5% | 75.5% |
| 4. Security operations | 81.6% | 82.9% |
| 5. Program management & oversight | 64.8% | 70.4% |
| **Overall** | **75.3%** | **77.5%** |

The overall gain is small and within normal variation for a test this size. The project's real goal is behavior, not trivia.

## What changed from v1 to v2

v1 had two problems:

1. **Over-refusal.** It refused ordinary study questions because the training data rewarded refusing anything that looked like reconnaissance.
2. **Indirect injection.** It obeyed an instruction hidden inside a log file it was asked to analyze.

v2 fixed these through the data only. `bird_clean.py` removed the bad refusal pairs. `bird_logs_gen.py` added log-analysis examples where the right answer is to report the hidden instruction instead of following it.

## Limitations

- Attack-blocking and false-refusal rates are not measured automatically yet (planned: `bird_eval.py`).
- The Northwind test logs share patterns with the v2 training data, so they're a demo, not a fair benchmark.
- All training data is synthetic and model-written, so some answers may be wrong.
- This is a learning project, not a production security control.

## Data and licensing

No datasets or model weights are included. The scripts download public sources at run time. OWASP-derived data is CC BY-SA 4.0, so if you publish a dataset built with these scripts, publish it under the same license and keep the generated `ATTRIBUTION.md`.
