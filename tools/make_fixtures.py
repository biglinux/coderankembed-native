"""Write tests/fixtures/reference.json: token ids from Hugging Face's
tokenizer and embeddings from sentence-transformers running CodeRankEmbed's
own PyTorch code in float32, for the parity tests to compare against.

Needs torch, transformers 4 (the model's remote code calls
`get_extended_attention_mask`, which transformers 5 removed),
sentence-transformers and einops, and downloads nomic-ai/CodeRankEmbed and
its modeling code from the Hugging Face Hub:

    python -m venv venv
    venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
    venv/bin/pip install "transformers<5" "sentence-transformers<6" einops
    venv/bin/python -I tools/make_fixtures.py
"""

import json
import random
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

QUERY_PREFIX = "Represent this query for searching relevant code: "
OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "reference.json"

QUERIES = [
    "Calculate the n-th factorial",
    "parse a JSON file and return the list of users",
    "Função que remove acentos de uma string",
]

PYTHON = '''def binary_search(items, target):
    """Return the index of target in sorted items, or -1."""
    lo, hi = 0, len(items) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
'''

RUST = """/// Café, naïve, Ελληνικά, 中文注释 and an emoji 🚀 in a doc comment.
pub fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() { return None; }
    Some(values.iter().sum::<f64>() / values.len() as f64) // [SEP] in a comment
}
"""


def long_code(lines: int) -> str:
    """A function of `lines` random assignments: 250 lines pass 2000 tokens,
    1200 pass the 8192 the model cuts texts to."""
    rng = random.Random(7)
    names = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "theta"]
    out = []
    for i in range(lines):
        a, b = rng.choice(names), rng.choice(names)
        out.append(
            f"    {a}_{i} = compute({b}, {rng.randint(0, 999)}) * {rng.random():.4f}  # step {i}"
        )
    return "def generated():\n" + "\n".join(out) + "\n    return locals()\n"


def tokenizer_cases() -> list[str]:
    rng = random.Random(11)
    pools = [
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        " \t\n\r\x0b\x0c 　",
        "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~«»¡¿–—…‘’“”",
        "áàâãäåçéèêëíìîïñóòôõöúùûüýÿÁÀÂÃÄÅÇÉÈÊËÍÌÎÏÑÓÒÔÕÖÚÙÛÜΣσςİıßẞ",
        "中文日本語한국어漢字ｶﾀｶﾅ",
        "́̈\u200b‍﻿\x00\x07�\U0001f680\U0001f600",
        "αβγδεζηθλμπρστφχψωДЖЗИЙКЛМНОПабвгд",
    ]
    cases = [
        "",
        " ",
        "[CLS][SEP]",
        "x[MASK]y",
        "[UNK",
        "a" * 100,
        "a" * 101,
        "Supercalifragilisticexpialidocious",
    ]
    for _ in range(300):
        pool = rng.choice(pools)
        cases.append("".join(rng.choice(pool) for _ in range(rng.randint(1, 40))))
    return cases


def main() -> None:
    model = SentenceTransformer(
        "nomic-ai/CodeRankEmbed", trust_remote_code=True, device="cpu"
    )
    model = model.to(torch.float32).eval()
    tokenizer = model.tokenizer
    max_tokens = model.max_seq_length
    documents = ["", PYTHON, RUST, long_code(250), long_code(1200)]
    texts = [(QUERY_PREFIX + q, "query") for q in QUERIES] + [
        (d, "document") for d in documents
    ]
    with torch.inference_mode():
        embeddings = [
            model.encode([text], convert_to_numpy=True)[0].tolist() for text, _ in texts
        ]

    def ids(text: str) -> list[int]:
        return tokenizer(text, truncation=True, max_length=max_tokens)["input_ids"]

    fixtures = {
        "max_tokens": max_tokens,
        "embeddings": [
            {
                "text": text.removeprefix(QUERY_PREFIX) if role == "query" else text,
                "role": role,
                "ids": ids(text),
                "embedding": embedding,
            }
            for (text, role), embedding in zip(texts, embeddings, strict=True)
        ],
        "tokens": [{"text": case, "ids": ids(case)} for case in tokenizer_cases()],
    }
    OUT.write_text(json.dumps(fixtures, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
