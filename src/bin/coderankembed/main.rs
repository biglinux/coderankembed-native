//! CodeRankEmbed code search from the command line. Its commands, options,
//! variables and exit statuses are [`USAGE`] and [`HELP`], which `--help`
//! prints.

mod index;

use std::ffi::OsString;
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use coderankembed_native::{
    DIMENSION, Device, Encoder, Error, MAX_TOKENS, Role, default_model_dir,
};

const USAGE: &str = "usage: coderankembed embed [--query] [--lines] [--raw] [FILE...]
       coderankembed index [--chunk-tokens N] DIR INDEX
       coderankembed search [--top K] INDEX QUERY...
       coderankembed bench [--tokens N] [--batch N] [--rounds N]
       coderankembed --help | -h";

const HELP: &str = "
Commands:
  embed   Write one JSON array of 768 floats per text: each FILE is a text,
          or standard input is.
    --query           put the prefix the model was trained with before each
                      text, for questions
    --lines           each line of the input is a text
    --raw             write little-endian f32 instead of JSON
  index   Embed the text files of DIR (what git tracks or would track, in a
          work tree) in chunks of whole lines into the file INDEX; run again,
          it embeds only the files that changed.
    --chunk-tokens N  tokens a chunk holds, 16 to 8192 (default 1024)
  search  Print the chunks of INDEX closest to QUERY: cosine similarity,
          then path:first-last.
    --top K           chunks to print (default 10)
  bench   Embed synthetic code and print the tokens per second of the best
          round.
    --tokens N        tokens a text holds at least, 2 to 8192 (default 512)
    --batch N         texts a round embeds (default 16)
    --rounds N        rounds after one warm-up (default 3)

Options of every command:
  --model DIR         the files of nomic-ai/CodeRankEmbed (default:
                      CODERANKEMBED_MODEL, else the Hugging Face cache)
  --device D          gpu, the best Vulkan GPU; cpu; or auto (default), the
                      GPU when there is one, else the CPU, saying why
  --threads N         CPU threads (default: the CPUs this process may use)
  --max-tokens N      tokens a text is cut to, 2 to 8192 (default 8192);
                      not for index, which takes --chunk-tokens

Environment:
  CODERANKEMBED_MODEL the model's directory when --model is not given
  CODERANKEMBED_TIER  0, 1 or 2: run the CPU kernels for SSE2, AVX or AVX2
                      at most

Exit status: 0 done, 1 failed, 2 wrong command line.";

/// What to run, with its operands.
enum Command {
    /// The files to embed, or standard input when there are none.
    Embed(Vec<PathBuf>),
    Index {
        dir: PathBuf,
        index: PathBuf,
    },
    Search {
        index: PathBuf,
        query: String,
    },
    Bench,
}

/// The options, each at its default unless given.
struct Args {
    model: Option<PathBuf>,
    /// None for `auto`.
    device: Option<Device>,
    threads: usize,
    max_tokens: Option<usize>,
    query: bool,
    lines: bool,
    raw: bool,
    chunk_tokens: u32,
    top: usize,
    tokens: usize,
    batch: usize,
    rounds: usize,
}

impl Default for Args {
    /// The defaults `HELP` states.
    fn default() -> Self {
        Self {
            model: None,
            device: None,
            threads: 0,
            max_tokens: None,
            query: false,
            lines: false,
            raw: false,
            chunk_tokens: 1024,
            top: 10,
            tokens: 512,
            batch: 16,
            rounds: 3,
        }
    }
}

impl Args {
    /// Refuse a number outside the range its option takes.
    fn check(&self) -> Result<(), String> {
        // The library panics on any other value; the command line refuses it.
        if std::env::var_os("CODERANKEMBED_TIER")
            .is_some_and(|tier| !["0", "1", "2"].contains(&tier.to_string_lossy().as_ref()))
        {
            return Err("CODERANKEMBED_TIER takes 0, 1 or 2".to_string());
        }
        if self
            .max_tokens
            .is_some_and(|n| !(2..=MAX_TOKENS).contains(&n))
        {
            return Err("--max-tokens takes 2 to 8192".to_string());
        }
        if !(16..=MAX_TOKENS).contains(&(self.chunk_tokens as usize)) {
            return Err("--chunk-tokens takes 16 to 8192".to_string());
        }
        if !(2..=MAX_TOKENS).contains(&self.tokens) {
            return Err("--tokens takes 2 to 8192".to_string());
        }
        if self.batch == 0 || self.rounds == 0 {
            return Err("--batch and --rounds need at least 1".to_string());
        }
        Ok(())
    }
}

fn number<T: std::str::FromStr>(name: &str, value: Option<OsString>) -> Result<T, String> {
    value
        .and_then(|v| v.into_string().ok())
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| format!("{name} needs a number"))
}

/// The command and options of `argv`, or `None` for `--help`.
fn parse(mut argv: impl Iterator<Item = OsString>) -> Result<Option<(Command, Args)>, String> {
    let name = argv
        .next()
        .and_then(|c| c.into_string().ok())
        .ok_or("no command")?;
    if matches!(name.as_str(), "--help" | "-h") {
        return Ok(None);
    }
    if !matches!(name.as_str(), "embed" | "index" | "search" | "bench") {
        return Err(format!("unknown command {name}"));
    }
    let mut args = Args::default();
    let mut operands = Vec::new();
    while let Some(arg) = argv.next() {
        match arg.to_str() {
            Some("--help" | "-h") => return Ok(None),
            Some("--model") => {
                args.model = Some(argv.next().ok_or("--model needs a directory")?.into())
            }
            Some("--device") => {
                args.device = match argv.next().as_ref().and_then(|d| d.to_str()) {
                    Some("auto") => None,
                    Some("gpu") => Some(Device::Gpu),
                    Some("cpu") => Some(Device::Cpu),
                    _ => return Err("--device takes auto, gpu or cpu".to_string()),
                }
            }
            Some("--threads") => args.threads = number("--threads", argv.next())?,
            // A command refuses the options of another, which it would ignore.
            Some("--max-tokens") if name != "index" => {
                args.max_tokens = Some(number("--max-tokens", argv.next())?)
            }
            Some("--query") if name == "embed" => args.query = true,
            Some("--lines") if name == "embed" => args.lines = true,
            Some("--raw") if name == "embed" => args.raw = true,
            Some("--chunk-tokens") if name == "index" => {
                args.chunk_tokens = number("--chunk-tokens", argv.next())?
            }
            Some("--top") if name == "search" => args.top = number("--top", argv.next())?,
            Some("--tokens") if name == "bench" => args.tokens = number("--tokens", argv.next())?,
            Some("--batch") if name == "bench" => args.batch = number("--batch", argv.next())?,
            Some("--rounds") if name == "bench" => args.rounds = number("--rounds", argv.next())?,
            Some("--") => operands.extend(argv.by_ref()),
            Some(flag) if flag.starts_with("--") => {
                return Err(format!("{name} has no option {flag}"));
            }
            _ => operands.push(arg),
        }
    }
    args.check()?;
    Ok(Some((command(&name, &operands)?, args)))
}

/// The command `name` with its operands, which it checks.
fn command(name: &str, operands: &[OsString]) -> Result<Command, String> {
    Ok(match (name, operands) {
        ("embed", files) => Command::Embed(files.iter().map(PathBuf::from).collect()),
        ("index", [dir, index]) => Command::Index {
            dir: dir.into(),
            index: index.into(),
        },
        ("search", [index, query @ ..]) if !query.is_empty() => {
            let words: Vec<String> = query
                .iter()
                .map(|q| q.to_string_lossy().into_owned())
                .collect();
            Command::Search {
                index: index.into(),
                query: words.join(" "),
            }
        }
        ("bench", []) => Command::Bench,
        ("index", _) => return Err("index takes DIR and INDEX".to_string()),
        ("search", _) => return Err("search takes INDEX and a query".to_string()),
        ("bench", _) => return Err("bench takes no operands".to_string()),
        _ => return Err(format!("unknown command {name}")),
    })
}

fn main() -> ExitCode {
    let (command, args) = match parse(std::env::args_os().skip(1)) {
        Ok(Some(parsed)) => parsed,
        Ok(None) => {
            return match written(writeln!(io::stdout(), "{USAGE}\n{HELP}")) {
                Ok(()) => ExitCode::SUCCESS,
                Err(message) => {
                    eprintln!("coderankembed: {message}");
                    ExitCode::FAILURE
                }
            };
        }
        Err(message) => {
            eprintln!("coderankembed: {message}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    match run(&command, &args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("coderankembed: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run(command: &Command, args: &Args) -> Result<(), String> {
    match command {
        Command::Embed(files) => embed(args, files),
        Command::Index { dir, index } => {
            index::build(&mut open(args)?, dir, index, args.chunk_tokens)
        }
        Command::Search { index, query } => search(args, index, query),
        Command::Bench => bench(&mut open(args)?, args),
    }
}

fn open(args: &Args) -> Result<Encoder, String> {
    let dir = args.model.clone().or_else(default_model_dir).ok_or(
        "no CodeRankEmbed model: download nomic-ai/CodeRankEmbed with the Hugging Face CLI, or pass --model DIR",
    )?;
    let started = Instant::now();
    let mut encoder = match args.device {
        Some(device) => Encoder::open(&dir, device, args.threads),
        None => match Encoder::open(&dir, Device::Gpu, args.threads) {
            Err(Error::Gpu(e)) => {
                eprintln!("coderankembed: no GPU ({e}), running on the CPU");
                Encoder::open(&dir, Device::Cpu, args.threads)
            }
            opened => opened,
        },
    }
    .map_err(|e| e.to_string())?;
    if let Some(n) = args.max_tokens {
        encoder.set_max_tokens(n);
    }
    eprintln!(
        "loaded {} in {:.2} s, on {}",
        dir.display(),
        started.elapsed().as_secs_f64(),
        encoder.device_name()
    );
    Ok(encoder)
}

/// Read the texts, then load the model: a wrong FILE fails at once.
fn embed(args: &Args, files: &[PathBuf]) -> Result<(), String> {
    let mut texts = Vec::new();
    if files.is_empty() {
        let mut text = String::new();
        io::stdin()
            .read_to_string(&mut text)
            .map_err(|e| format!("standard input: {e}"))?;
        texts.push(text);
    } else {
        for path in files {
            texts.push(
                std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?,
            );
        }
    }
    if args.lines {
        texts = texts
            .iter()
            .flat_map(|t| t.lines().map(str::to_string))
            .collect();
    }
    let role = if args.query {
        Role::Query
    } else {
        Role::Document
    };
    let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
    let embeddings = open(args)?.embed(&texts, role).map_err(|e| e.to_string())?;
    let mut out = BufWriter::new(io::stdout().lock());
    let output: io::Result<()> = (|| {
        for embedding in embeddings.as_chunks::<DIMENSION>().0 {
            if args.raw {
                for v in embedding {
                    out.write_all(&v.to_le_bytes())?;
                }
            } else {
                let values: Vec<String> = embedding.iter().map(f32::to_string).collect();
                writeln!(out, "[{}]", values.join(","))?;
            }
        }
        out.flush()
    })();
    written(output)
}

fn search(args: &Args, path: &Path, query: &str) -> Result<(), String> {
    let index = index::Index::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut encoder = open(args)?;
    let started = Instant::now();
    let embedding = encoder
        .embed(&[query], Role::Query)
        .map_err(|e| e.to_string())?;
    let hits = index::search(&index, &embedding, args.top);
    eprintln!(
        "searched {} in {:.0} ms",
        index::printable(&index.root.to_string_lossy()),
        started.elapsed().as_secs_f64() * 1000.0
    );
    let mut out = io::stdout().lock();
    written(hits.iter().try_for_each(|(score, file, chunk)| {
        writeln!(
            out,
            "{score:.4}  {}:{}-{}",
            index::printable(&file.path),
            chunk.lines.0,
            chunk.lines.1
        )
    }))
}

/// The outcome of writing to standard output: a reader that stopped early,
/// as `head` does, wanted no more and is no failure.
fn written(result: io::Result<()>) -> Result<(), String> {
    match result {
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        other => other.map_err(|e| format!("standard output: {e}")),
    }
}

/// Embed `batch` texts of about `tokens` tokens each, `rounds` times after
/// one warm-up, and report the best round.
fn bench(encoder: &mut Encoder, args: &Args) -> Result<(), String> {
    let mut texts = Vec::with_capacity(args.batch);
    for t in 0..args.batch {
        let mut text = String::new();
        let mut tokens = 2;
        let mut i = 0;
        while tokens < args.tokens {
            let line = format!("let value_{t}_{i} = compute(input[{i}], {t}) * {i};\n");
            tokens += encoder.count_tokens(&line);
            text.push_str(&line);
            i += 1;
        }
        texts.push(text);
    }
    let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
    let tokens: usize = texts.iter().map(|t| encoder.tokenize(t).len()).sum();
    encoder
        .embed(&texts, Role::Document)
        .map_err(|e| e.to_string())?;
    let mut best = f64::INFINITY;
    for _ in 0..args.rounds {
        let started = Instant::now();
        encoder
            .embed(&texts, Role::Document)
            .map_err(|e| e.to_string())?;
        best = best.min(started.elapsed().as_secs_f64());
    }
    println!(
        "{} texts, {tokens} tokens: {:.1} ms, {:.0} tokens/s",
        texts.len(),
        best * 1000.0,
        tokens as f64 / best
    );
    Ok(())
}
