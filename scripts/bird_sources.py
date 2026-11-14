#!/usr/bin/env python3
"""
B.I.R.D. source downloader.

Downloads openly licensed security references, extracts text, splits it into
chunks, and tags each chunk with the CompTIA Security+ SY0-701 domain it fits best.

Sources (all free to use for training; keep ATTRIBUTION.md with anything you share):
  - NIST SP 800 series + CSF 2.0   public domain (US Government work)
  - MITRE ATT&CK, CAPEC, CWE       free use with attribution (MITRE terms of use)
  - OWASP Cheat Sheet Series       CC BY-SA 4.0 (share-alike if you redistribute derived data)
  - CISA KEV catalog               public domain

Output (in --out, default ./bird_data):
  raw/<source files>           cached downloads (re-runs skip them)
  chunks.jsonl                 {"id","source","title","url","license","domain","text"}
  ATTRIBUTION.md               licenses and links

Usage:
  pip install requests pypdf
  python bird_sources.py                      # everything
  python bird_sources.py --only nist owasp    # subset
  python bird_sources.py --list
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# --------------------------------------------------------------------------- #
# Source catalog
# --------------------------------------------------------------------------- #

NIST_DOCS = {
    "NIST CSF 2.0": "https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf",
    "NIST SP 800-53 Rev. 5 (Security and Privacy Controls)": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf",
    "NIST SP 800-61 Rev. 3 (Incident Response)": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r3.pdf",
    "NIST SP 800-63-4 (Digital Identity)": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63-4.pdf",
    "NIST SP 800-207 (Zero Trust Architecture)": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-207.pdf",
    "NIST SP 800-30 Rev. 1 (Risk Assessments)": "https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-30r1.pdf",
    "NIST SP 800-34 Rev. 1 (Contingency Planning)": "https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-34r1.pdf",
    "NIST SP 800-41 Rev. 1 (Firewalls)": "https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-41r1.pdf",
    "NIST SP 800-115 (Security Testing)": "https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-115.pdf",
    "NIST SP 800-175B Rev. 1 (Cryptography)": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-175Br1.pdf",
    "NIST SP 800-40 Rev. 4 (Patch Management)": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-40r4.pdf",
    "NIST SP 800-92 (Log Management)": "https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-92.pdf",
}

URLS = {
    "attack": "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json",
    "capec": "https://raw.githubusercontent.com/mitre/cti/master/capec/2.1/stix-capec.json",
    "cwe": "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip",
    "owasp": "https://github.com/OWASP/CheatSheetSeries/archive/refs/heads/master.zip",
    "kev": "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
}

LICENSES = {
    "nist": "Public domain (U.S. Government work), NIST",
    "attack": "MITRE ATT&CK, (c) The MITRE Corporation, used under the ATT&CK Terms of Use",
    "capec": "CAPEC, (c) The MITRE Corporation, used under the CAPEC Terms of Use",
    "cwe": "CWE, (c) The MITRE Corporation, used under the CWE Terms of Use",
    "owasp": "OWASP Cheat Sheet Series, CC BY-SA 4.0",
    "kev": "CISA Known Exploited Vulnerabilities Catalog, public domain",
}

ALL_SOURCES = ["nist", "attack", "capec", "cwe", "owasp", "kev"]

# --------------------------------------------------------------------------- #
# SY0-701 domain tagging (keyword scoring; good enough for routing prompts)
# --------------------------------------------------------------------------- #

DOMAINS = {
    "1.0 General Security Concepts": [
        "confidentiality", "integrity", "availability", "non-repudiation", "aaa", "authentication",
        "authorization", "accounting", "zero trust", "gap analysis", "change management", "physical security",
        "deception", "honeypot", "encryption", "hashing", "salting", "digital signature", "pki", "certificate",
        "key exchange", "tpm", "hsm", "obfuscation", "steganography", "tokenization", "blockchain",
        "symmetric", "asymmetric", "control categor", "compensating control", "deterrent",
    ],
    "2.0 Threats, Vulnerabilities, and Mitigations": [
        "threat actor", "phishing", "social engineering", "malware", "ransomware", "trojan", "worm",
        "rootkit", "keylogger", "spyware", "vulnerability", "injection", "cross-site", "xss", "csrf",
        "buffer overflow", "race condition", "privilege escalation", "supply chain", "zero-day", "exploit",
        "brute force", "password spray", "replay", "on-path", "man-in-the-middle", "ddos", "denial of service",
        "indicator of compromise", "attack surface", "misconfiguration", "segmentation", "patching", "hardening",
        "cwe", "capec", "att&ck", "technique", "tactic",
    ],
    "3.0 Security Architecture": [
        "architecture", "cloud", "iaas", "paas", "saas", "virtualization", "container", "microservice",
        "serverless", "iot", "ics", "scada", "embedded", "network infrastructure", "firewall", "ids", "ips",
        "vpn", "ipsec", "tls", "proxy", "load balancer", "sd-wan", "sase", "data classification", "data at rest",
        "data in transit", "resilience", "high availability", "backup", "recovery site", "replication",
        "redundan", "contingency", "power", "capacity planning",
    ],
    "4.0 Security Operations": [
        "baseline", "secure configuration", "mobile device", "wireless", "wpa3", "asset management",
        "inventory", "vulnerability scan", "penetration test", "monitoring", "siem", "log", "alert",
        "edr", "xdr", "dlp", "email security", "dmarc", "dkim", "spf", "identity and access", "iam",
        "provisioning", "single sign-on", "sso", "federation", "mfa", "multifactor", "privileged access",
        "automation", "orchestration", "incident response", "containment", "eradication", "forensic",
        "chain of custody", "playbook", "threat hunting", "patch",
    ],
    "5.0 Security Program Management and Oversight": [
        "governance", "policy", "procedure", "standard", "guideline", "compliance", "regulation", "audit",
        "risk management", "risk assessment", "risk register", "risk appetite", "ale", "sle", "aro",
        "business impact", "bia", "rto", "rpo", "third-party", "vendor", "sla", "mou", "nda",
        "privacy", "gdpr", "awareness training", "attestation", "data owner", "data custodian", "framework",
        "csf", "control assessment",
    ],
}


def tag_domain(text: str, source: str) -> str:
    t = text.lower()
    scores = {d: sum(t.count(k) for k in kws) for d, kws in DOMAINS.items()}
    # source priors: ATT&CK/CAPEC/CWE/KEV are mostly threats; NIST leans ops/governance
    if source in ("attack", "capec", "cwe", "kev"):
        scores["2.0 Threats, Vulnerabilities, and Mitigations"] += 3
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "1.0 General Security Concepts"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

@dataclass
class Doc:
    source: str
    title: str
    url: str
    text: str


def download(url: str, dest: Path, retries: int = 3) -> Path:
    import requests
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "BIRD-dataset-builder/1.0 (educational research)"}
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with tmp.open("wb") as f:
                    for block in r.iter_content(1 << 16):
                        f.write(block)
                tmp.rename(dest)
            print(f"  downloaded {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
            return dest
        except Exception as e:
            print(f"  attempt {attempt} failed for {url}: {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"could not download {url}")


def clean(text: str) -> str:
    text = text.replace("­", "").replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"-\n(?=[a-z])", "", text)                  # de-hyphenate line breaks
    text = re.sub(r"This publication is available free of charge from:?\s*\S+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, max_words: int = 350, overlap: int = 50) -> Iterator[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.split()) >= 8]
    buf: list[str] = []
    count = 0
    for p in paras:
        words = p.split()
        if len(words) > max_words:                               # split very long paragraphs
            for i in range(0, len(words), max_words - overlap):
                yield " ".join(words[i:i + max_words])
            continue
        if count + len(words) > max_words and buf:
            yield "\n\n".join(buf)
            tail = " ".join(" ".join(buf).split()[-overlap:])
            buf, count = [tail], len(tail.split())
        buf.append(p)
        count += len(words)
    if buf and count >= 25:
        yield "\n\n".join(buf)


def _stix_external_url(obj: dict) -> str:
    for ref in obj.get("external_references", []):
        if ref.get("url"):
            return ref["url"]
    return ""


def _strip_md(s: str) -> str:
    s = re.sub(r"```.*?```", " ", s, flags=re.S)                # drop code blocks
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", s)                 # images
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)              # links -> text
    s = re.sub(r"\(Citation:[^)]*\)", "", s)                    # ATT&CK citations
    s = re.sub(r"<[^>]+>", " ", s)
    return s


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def load_nist(raw: Path) -> Iterator[Doc]:
    from pypdf import PdfReader
    for title, url in NIST_DOCS.items():
        try:
            path = download(url, raw / "nist" / url.rsplit("/", 1)[-1])
        except RuntimeError as e:
            print(f"  skipping {title}: {e}")
            continue
        try:
            reader = PdfReader(str(path))
            pages = [(p.extract_text() or "") for p in reader.pages]
        except Exception as e:
            print(f"  could not read {path.name}: {e}")
            continue
        # drop front matter (roughly the first 6% of pages) and reference/appendix tails are kept
        start = max(1, len(pages) // 16)
        yield Doc("nist", title, url, clean("\n\n".join(pages[start:])))


def load_stix(raw: Path, key: str, types: tuple[str, ...]) -> Iterator[Doc]:
    path = download(URLS[key], raw / key / URLS[key].rsplit("/", 1)[-1])
    data = json.loads(path.read_text(encoding="utf-8"))
    for obj in data.get("objects", []):
        if obj.get("type") not in types or obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        name, desc = obj.get("name", ""), obj.get("description", "")
        extra = []
        if obj.get("kill_chain_phases"):
            extra.append("Tactics: " + ", ".join(p["phase_name"] for p in obj["kill_chain_phases"]))
        if obj.get("x_mitre_detection"):
            extra.append("Detection: " + obj["x_mitre_detection"])
        for k in ("x_capec_prerequisites", "x_capec_mitigations"):
            if obj.get(k):
                extra.append(k.split("_")[-1].title() + ": " + " ".join(obj[k]))
        text = _strip_md(f"{name}\n\n{desc}\n\n" + "\n\n".join(extra))
        if len(text.split()) >= 30:
            yield Doc(key, f"{key.upper()}: {name}", _stix_external_url(obj), clean(text))


def load_cwe(raw: Path) -> Iterator[Doc]:
    import xml.etree.ElementTree as ET
    path = download(URLS["cwe"], raw / "cwe" / "cwec_latest.xml.zip")
    with zipfile.ZipFile(path) as z:
        xml_name = next(n for n in z.namelist() if n.endswith(".xml"))
        root = ET.fromstring(z.read(xml_name))
    ns = {"c": root.tag.split("}")[0].strip("{")}
    for w in root.iterfind(".//c:Weakness", ns):
        if w.get("Status") == "Deprecated":
            continue
        parts = [f"CWE-{w.get('ID')}: {w.get('Name')}"]
        for tag in ("Description", "Extended_Description"):
            el = w.find(f"c:{tag}", ns)
            if el is not None:
                parts.append(" ".join(el.itertext()))
        mit = w.find("c:Potential_Mitigations", ns)
        if mit is not None:
            parts.append("Mitigations: " + " ".join(" ".join(m.itertext()) for m in mit)[:2000])
        text = clean("\n\n".join(parts))
        if len(text.split()) >= 30:
            yield Doc("cwe", f"CWE-{w.get('ID')}: {w.get('Name')}",
                      f"https://cwe.mitre.org/data/definitions/{w.get('ID')}.html", text)


def load_owasp(raw: Path) -> Iterator[Doc]:
    path = download(URLS["owasp"], raw / "owasp" / "CheatSheetSeries.zip")
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if "/cheatsheets/" not in name or not name.endswith(".md"):
                continue
            md = z.read(name).decode("utf-8", "ignore")
            title = Path(name).stem.replace("_", " ")
            slug = Path(name).stem
            yield Doc("owasp", f"OWASP {title}",
                      f"https://cheatsheetseries.owasp.org/cheatsheets/{slug}.html", clean(_strip_md(md)))


def load_kev(raw: Path, limit: int = 400) -> Iterator[Doc]:
    path = download(URLS["kev"], raw / "kev" / "kev.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    vulns = sorted(data.get("vulnerabilities", []), key=lambda v: v.get("dateAdded", ""), reverse=True)[:limit]
    for v in vulns:
        text = (f"{v['cveID']}: {v.get('vendorProject')} {v.get('product')} - {v.get('vulnerabilityName')}\n\n"
                f"{v.get('shortDescription','')}\n\nRequired action: {v.get('requiredAction','')}\n"
                f"Known ransomware use: {v.get('knownRansomwareCampaignUse','Unknown')}. "
                f"Related weaknesses: {', '.join(v.get('cwes', [])) or 'n/a'}.")
        yield Doc("kev", v["cveID"], f"https://nvd.nist.gov/vuln/detail/{v['cveID']}", clean(text))


LOADERS = {
    "nist": load_nist,
    "attack": lambda raw: load_stix(raw, "attack", ("attack-pattern", "course-of-action", "malware", "tool")),
    "capec": lambda raw: load_stix(raw, "capec", ("attack-pattern",)),
    "cwe": load_cwe,
    "owasp": load_owasp,
    "kev": load_kev,
}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def write_attribution(out: Path, used: list[str]) -> None:
    lines = ["# B.I.R.D. training data attribution\n",
             "This dataset was derived from the following sources:\n"]
    for s in used:
        lines.append(f"- **{s}**: {LICENSES[s]}")
    lines += ["", "OWASP-derived content is CC BY-SA 4.0: if you publish data or text derived from it, "
              "publish it under the same license with attribution.",
              "MITRE content: (c) The MITRE Corporation. Reproduced and distributed with the permission of "
              "The MITRE Corporation under its terms of use."]
    (out / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download and chunk open security references for B.I.R.D.")
    p.add_argument("--out", default="bird_data")
    p.add_argument("--only", nargs="+", choices=ALL_SOURCES, default=ALL_SOURCES)
    p.add_argument("--max-words", type=int, default=350, help="words per chunk")
    p.add_argument("--kev-limit", type=int, default=400, help="most recent KEV entries to include")
    p.add_argument("--list", action="store_true", help="show sources and exit")
    a = p.parse_args(argv)

    if a.list:
        for s in ALL_SOURCES:
            print(f"{s:8} {LICENSES[s]}")
        for t, u in NIST_DOCS.items():
            print(f"         {t}: {u}")
        return 0

    out = Path(a.out).expanduser()
    raw = out / "raw"
    out.mkdir(parents=True, exist_ok=True)
    chunks_path = out / "chunks.jsonl"
    seen: set[str] = set()
    counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    used: list[str] = []

    with chunks_path.open("w", encoding="utf-8") as f:
        for src in a.only:
            print(f"[{src}]")
            try:
                loader = LOADERS[src]
                docs = loader(raw) if src != "kev" else load_kev(raw, a.kev_limit)
                n = 0
                for doc in docs:
                    for piece in chunk_text(doc.text, a.max_words):
                        piece = f"{doc.title}\n\n{piece}"      # keep context (CVE/CWE ids, section)
                        h = hashlib.sha1(piece.encode()).hexdigest()
                        if h in seen:
                            continue
                        seen.add(h)
                        dom = tag_domain(piece, src)
                        rec = {"id": f"{src}-{h[:12]}", "source": src, "title": doc.title, "url": doc.url,
                               "license": LICENSES[src], "domain": dom, "text": piece}
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n += 1
                        domain_counts[dom] = domain_counts.get(dom, 0) + 1
                counts[src] = n
                used.append(src)
                print(f"  {n} chunks")
            except Exception as e:
                print(f"  FAILED: {e}  (continuing with other sources)")

    write_attribution(out, used)
    print(f"\nWrote {sum(counts.values())} chunks to {chunks_path}")
    for d in sorted(domain_counts):
        print(f"  {d}: {domain_counts[d]}")
    return 0 if counts else 1


if __name__ == "__main__":
    sys.exit(main())
