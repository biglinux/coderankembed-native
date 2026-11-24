# CodeRankEmbed native (coderankembed-native)

Search code in plain language on your own machine: an engine that runs
the CodeRankEmbed code retrieval model in plain Rust, on the CPU or a Vulkan
GPU, without Python.

This repository holds the inference engine only, not the model: its weights
are Nomic AI's [nomic-ai/CodeRankEmbed](https://huggingface.co/nomic-ai/CodeRankEmbed),
which you download from Hugging Face once ([Install](#install)).

- Embeds code and questions with the same numbers as the model's own PyTorch
  code (cosine similarity above 0.99999 on every reference text), and its
  tokenizer gives the same ids as Hugging Face's.
- Indexes a source tree once; a search then loads the model in under a
  second and embeds the query in milliseconds. Indexing again embeds only
  the files that changed.
- Runs 2 to 9 times faster than PyTorch on the same CPU, and up to 6 times
  faster again on a discrete GPU with a Vulkan 1.2 driver.
- Reads the model's files as Hugging Face publishes them: no conversion step.

## Install

It runs on Linux, on x86-64 with its vector kernels, and on a GPU through
the system's Vulkan loader. There is no package in a repository yet;
`makepkg -si` in [`pkgbuild/`](pkgbuild/PKGBUILD) builds and installs one
from GitHub. Or build it with Rust 1.100 or newer and `glslc`, from
shaderc, which compiles the GPU shaders:

```sh
sudo pacman -S rust shaderc vulkan-icd-loader
cargo build --release
```

The binary is `target/release/coderankembed`. The model does not come with
it: download its files, `config.json`, `model.safetensors` and
`tokenizer.json` (547 MB, MIT license), from
[nomic-ai/CodeRankEmbed](https://huggingface.co/nomic-ai/CodeRankEmbed):

```sh
hf download nomic-ai/CodeRankEmbed config.json model.safetensors tokenizer.json
```

The program finds them in the Hugging Face cache; `--model DIR` or
`CODERANKEMBED_MODEL` points at another directory.

## Use

Index a source tree, then ask it questions:

```sh
coderankembed index ~/src/project project.idx
coderankembed search project.idx "compute mean reciprocal rank of reranked results"
```

```text
0.5206  src/rerank.py:1-85
0.3391  src/rerank.py:86-161
0.3368  src/rerank.py:162-173
```

Each line is the cosine similarity and a run of lines of a file; `--top K`
prints K of them (default 10). The index holds the files git tracks or
would track (outside a git work tree, every file and directory not hidden),
in chunks of whole lines of up to 1024 tokens (`--chunk-tokens`). Binary
files, files over 1 MiB, files not in UTF-8 and links that lead out of the
directory are left out.

Embed texts for another program, one JSON array of 768 floats per text:

```sh
coderankembed embed src/main.rs src/lib.rs
printf 'parse a JSON file\nopen a socket\n' | coderankembed embed --query --lines
```

`--query` puts the prefix the model was trained with before each question;
code goes in without it. `--raw` writes little-endian `f32` instead of JSON,
and `--max-tokens N` cuts each text to N tokens (default 8192).
`coderankembed bench --tokens 2048 --batch 4` measures the speed of this
machine on 4 texts of 2048 tokens, best of `--rounds` (default 3).

`--device gpu` runs on the best Vulkan GPU, `--device cpu` on the CPU, and
the default, `auto`, takes the GPU when there is one and says why when it
falls back. `coderankembed --help` (or `-h`) lists every command, option,
variable and exit status.

## How fast is it?

Tokens embedded per second, best of three runs. PyTorch is 2.15 on the
CPU, with sentence-transformers 5.7 and the model's own code in float32. A
query is one text of 21 tokens. The first three machines were idle; the last
was in use (10 to 20% CPU pressure, 3 GB of memory free), so PyTorch did not
run there.

| Machine | | 32 texts of 512 tokens | 4 of 2048 | 1 of 8192 | One query |
|---|---|---:|---:|---:|---:|
| Core i5-13400, GTX 1660 Ti | GPU | **17261** | **12354** | **5678** | **8 ms** |
| | CPU | 2693 | 2546 | 1334 | 14 ms |
| | PyTorch | 1042 | 463 | 146 | 110 ms |
| Ryzen 7 5700U, Radeon (Vega 8) | GPU | **3419** | **2139** | **904** | 38 ms |
| | CPU | 1114 | 902 | 394 | **30 ms** |
| | PyTorch | 547 | 316 | 111 | 63 ms |
| Ryzen 5 5600H, Radeon (Vega 7) | GPU | **3277** | **2049** | **814** | 51 ms |
| | CPU | 1984 | 1610 | 774 | **17 ms** |
| | PyTorch | 832 | 454 | 166 | 46 ms |
| Core Ultra 5 225H, Intel Arc (integrated) | GPU | **4423** | **2276** | **1097** | **29 ms** |
| | CPU | 1685 | 1387 | 777 | 33 ms |

Peak resident memory on the first machine is 680 MiB on the GPU path,
whatever the batch, and on the CPU, which keeps the weights in floats, from
600 MiB for one query to 930 MiB for batches of 8192-token texts.

How the encoder is built, and why, is in
[docs/architecture.md](docs/architecture.md).

## Help

Report a problem or ask a question in the
[issues](https://github.com/biglinux/coderankembed-native/issues).

## Contributing

```sh
cargo test --release
cargo test --release -- --include-ignored
cargo clippy --all-targets --config 'build.warnings="deny"'
ast-grep scan
```

The ignored tests need the model's files and, for those named `gpu_`, a
Vulkan GPU (`--skip gpu_` leaves them out). The parity tests compare
against `tests/fixtures/reference.json`, which
[`tools/make_fixtures.py`](tools/make_fixtures.py) writes from the model's
own PyTorch code. `ast-grep scan` runs the project's rules in
[`rules/`](rules).

The readers of the model's files and the tokenizer have fuzz targets in
`fuzz/` (`safetensors`, `tokenizer_json`, `tokenize`), run with the nightly
toolchain and `cargo-fuzz`:

```sh
cd fuzz
cargo +nightly fuzz run safetensors corpus/safetensors seeds/safetensors -- -max_total_time=600
```

## License

`MIT AND BSD-3-Clause`: the code is MIT ([LICENSE](LICENSE)); the exponential
follows XNNPACK's, under BSD-3-Clause. [NOTICE](NOTICE) credits both and the
model, which is MIT; its weights are not included.
