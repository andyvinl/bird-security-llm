# B.I.R.D. training pipeline

| Step | Script | Where to run | Output |
|---|---|---|---|
| 1 | `bird_sources.py` | GPU box (or Mac) | `bird_data/chunks.jsonl`, `ATTRIBUTION.md` |
| 2 | `bird_secplus_gen.py` | Mac (Ollama) | `bird_data/secplus_train.jsonl`, `secplus_eval.jsonl` |
| 3 | `bird_datagen.py` | Mac (Ollama) | `bird_train.jsonl` (prompt-injection defense) |
| 4 | `bird_train.py` | GPU box (CUDA) | `bird-lora/`, optional `bird-lora-gguf/` |

## 1. Download sources
```bash
pip install requests pypdf
python bird_sources.py            # ~100 MB download, cached in bird_data/raw/
```
Sources: NIST SP 800 series + CSF 2.0, MITRE ATT&CK / CAPEC / CWE, OWASP Cheat Sheets, CISA KEV.
A source that fails to download is skipped; the rest still run.

## 2 + 3. Generate data on the Mac
```bash
pip install ollama
ollama pull qwen2.5:3b
python bird_secplus_gen.py -n 2000 --eval-num 300
python bird_datagen.py -n 1000 -o bird_train.jsonl
```
Copy `bird_data/chunks.jsonl` from the GPU box first if you ran step 1 there:
`scp -i aws-gpu-key.pem ubuntu@<IP>:~/bird/bird_data/chunks.jsonl bird_data/`

## 3b. v2 additions (fix over-refusal and hidden injections)
```bash
python bird_clean.py bird_train.jsonl -o bird_train_clean.jsonl     # drop v1 'reconnaissance' refusal turns
python bird_logs_gen.py -n 800 -o bird_data/logs_train.jsonl       # log analysis + injections hidden in data
```

## 4. Train on the GPU box
```bash
scp -i aws-gpu-key.pem bird_train.jsonl bird_data/secplus_*.jsonl ubuntu@<IP>:~/bird/   # from the Mac
# on the GPU box, inside tmux:
pip install unsloth
python bird_train.py --check-data          # sanity check, no GPU
python bird_train.py --data bird_train_clean.jsonl bird_data/train_run2/secplus_train.jsonl bird_data/logs_train.jsonl \
  --eval bird_data/eval_run/secplus_eval.jsonl --base-model unsloth/Qwen2.5-3B-Instruct \
  --max-seq-len 4096 --batch-size 1 --grad-accum 16 --gguf
```
It scores the base model and the fine-tuned model on the held-out Security+ questions,
broken down by SY0-701 domain.

## Run the result on your Mac
```bash
scp -i aws-gpu-key.pem -r ubuntu@<IP>:~/bird/bird-lora-gguf .
cd bird-lora-gguf && printf 'FROM ./%s\n' "$(ls *.gguf | head -1)" > Modelfile
ollama create bird -f Modelfile && ollama run bird
```

## Licensing
Keep `ATTRIBUTION.md` with the dataset. OWASP-derived data is CC BY-SA 4.0: if you publish
the dataset, publish it under the same license.

## Chat with it
`python bird_chat.py` (raven banner + system prompt). Test with the Northwind audit log in `bird_test/`.

## v3: multi-format logs + behavior eval
```bash
# on the GPU box (Ollama running), from ~/bird
python bird_logs_gen_v3.py --attack-json bird_data/raw/attack/enterprise-attack.json -n 2400 -o bird_data/logs_v3.jsonl
python bird_eval.py --models bird --attack-json bird_data/raw/attack/enterprise-attack.json --out eval_v2.jsonl   # baseline
python bird_train.py --data bird_train_clean.jsonl bird_data/train_run2/secplus_train.jsonl bird_data/logs_v3.jsonl \
  --eval bird_data/eval_run/secplus_eval.jsonl --base-model unsloth/Qwen3-4B-Instruct-2507 --chat-template native \
  --max-seq-len 4096 --batch-size 1 --grad-accum 16 --epochs 1 --adapter bird-v3-lora --gguf
# put Modelfile.v3 next to the exported GGUF, then:
ollama create bird-v3 -f Modelfile.v3
python bird_eval.py --models bird bird-v3 --attack-json bird_data/raw/attack/enterprise-attack.json --out eval_v3.jsonl
```
`bird_eval.py` scores verdicts, injection resistance, key findings, MITRE ID validity, block-vs-isolate actions,
secret masking, over-refusal and refusal of harmful requests on held-out cases (different seed, prompts and
injection wording than training, plus a fixed demo firewall log).
