//! A searchable index of a source tree: its text files cut into chunks of
//! whole lines, each with its normalized embedding.
//!
//! A chunk is embedded as its file's path, a space, then its lines, as the
//! CoRNStack evaluation joins a file's title and text. Indexing again only
//! embeds the files whose size or modification time changed.
//!
//! The index file is this program's own cache: `CRE-IDX1`, then the indexed
//! directory, the chunk budget in tokens and the files, each its path, size,
//! modification time and chunks, each chunk its first and last line and its
//! embedding, all little-endian. It is replaced atomically, and a file this
//! version cannot read is refused, never half read.

use std::borrow::Cow;
use std::collections::HashMap;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, IsTerminal, Read, Write};
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, UNIX_EPOCH};

use coderankembed_native::{DIMENSION, Encoder, Role};

const MAGIC: &[u8; 8] = b"CRE-IDX1";
/// Files past this are generated or minified more often than written.
const MAX_FILE_BYTES: u64 = 1 << 20;
/// Chunks embedded between two progress reports.
const PROGRESS_STEP: usize = 256;

pub(crate) struct Chunk {
    /// First and last line, counted from 1.
    pub(crate) lines: (u32, u32),
    pub(crate) embedding: Vec<f32>,
}

pub(crate) struct IndexedFile {
    /// Relative to the indexed directory, `/`-separated.
    pub(crate) path: String,
    size: u64,
    modified: u128,
    pub(crate) chunks: Vec<Chunk>,
}

pub(crate) struct Index {
    pub(crate) root: PathBuf,
    chunk_tokens: u32,
    pub(crate) files: Vec<IndexedFile>,
}

impl Index {
    pub(crate) fn read(path: &Path) -> io::Result<Self> {
        let mut r = BufReader::new(File::open(path)?);
        // A file shorter than the magic is not an index either.
        let mut magic = Vec::new();
        r.by_ref()
            .take(MAGIC.len() as u64)
            .read_to_end(&mut magic)?;
        if magic != MAGIC {
            return Err(invalid("not a coderankembed index of this version"));
        }
        let root = PathBuf::from(OsString::from_vec(read_bytes(&mut r)?));
        let chunk_tokens = read_u32(&mut r)?;
        let mut files = Vec::new();
        for _ in 0..read_u32(&mut r)? {
            let path = read_string(&mut r)?;
            let size = read_u64(&mut r)?;
            let modified = u128::from(read_u64(&mut r)?) | (u128::from(read_u64(&mut r)?) << 64);
            let mut chunks = Vec::new();
            for _ in 0..read_u32(&mut r)? {
                let lines = (read_u32(&mut r)?, read_u32(&mut r)?);
                let mut raw = [0u8; DIMENSION * 4];
                r.read_exact(&mut raw)?;
                let embedding = raw
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                chunks.push(Chunk { lines, embedding });
            }
            files.push(IndexedFile {
                path,
                size,
                modified,
                chunks,
            });
        }
        if r.read(&mut [0u8; 1])? != 0 {
            return Err(invalid("trailing bytes"));
        }
        Ok(Self {
            root,
            chunk_tokens,
            files,
        })
    }

    /// Replace the file at `path` with this index, atomically, through the
    /// temp file of [`create_temp`].
    fn write(&self, path: &Path) -> io::Result<()> {
        let (temp, file) = create_temp(path)?;
        let mut w = BufWriter::new(file);
        let written = (|| {
            w.write_all(MAGIC)?;
            write_bytes(&mut w, self.root.as_os_str().as_bytes())?;
            w.write_all(&self.chunk_tokens.to_le_bytes())?;
            w.write_all(&count(self.files.len())?.to_le_bytes())?;
            for file in &self.files {
                write_bytes(&mut w, file.path.as_bytes())?;
                w.write_all(&file.size.to_le_bytes())?;
                w.write_all(&file.modified.to_le_bytes())?;
                w.write_all(&count(file.chunks.len())?.to_le_bytes())?;
                for chunk in &file.chunks {
                    w.write_all(&chunk.lines.0.to_le_bytes())?;
                    w.write_all(&chunk.lines.1.to_le_bytes())?;
                    for v in &chunk.embedding {
                        w.write_all(&v.to_le_bytes())?;
                    }
                }
            }
            w.flush()?;
            fs::rename(&temp, path)
        })();
        if written.is_err() {
            // The write's error is the one to report; a temp file that
            // cannot be removed either is left as it is.
            let _ = fs::remove_file(&temp);
        }
        written
    }
}

/// Index the text files of `root` into `index_path`, reusing the
/// embeddings of files unchanged since the index there was written.
pub(crate) fn build(
    encoder: &mut Encoder,
    root: &Path,
    index_path: &Path,
    chunk_tokens: u32,
) -> Result<(), String> {
    let started = Instant::now();
    // A line longer than the budget is a chunk of its own, cut to it.
    encoder.set_max_tokens(chunk_tokens as usize);
    let root = fs::canonicalize(root).map_err(|e| format!("{}: {e}", root.display()))?;
    temp_is_free(index_path)?;
    let old = reusable_files(index_path, &root, chunk_tokens)?;
    let old = old.into_iter().map(|f| (f.path.clone(), f)).collect();
    let (mut files, pending) = scan(encoder, &root, old, chunk_tokens)?;
    let total = pending.len();
    let progress = io::stderr().is_terminal();
    for (done, batch) in pending.chunks(PROGRESS_STEP).enumerate() {
        let texts: Vec<&str> = batch.iter().map(|(_, _, text)| text.as_str()).collect();
        let embeddings = encoder
            .embed(&texts, Role::Document)
            .map_err(|e| e.to_string())?;
        for ((file, lines, _), embedding) in batch.iter().zip(embeddings.as_chunks::<DIMENSION>().0)
        {
            files[*file].chunks.push(Chunk {
                lines: *lines,
                embedding: normalized(embedding),
            });
        }
        if progress {
            let done = (done * PROGRESS_STEP + batch.len()).min(total);
            eprint!("\rembedded {done} of {total} chunks");
        }
    }
    if progress && total > 0 {
        eprintln!();
    }
    let index = Index {
        root,
        chunk_tokens,
        files,
    };
    index
        .write(index_path)
        .map_err(|e| format!("{}: {e}", index_path.display()))?;
    let chunks: usize = index.files.iter().map(|f| f.chunks.len()).sum();
    eprintln!(
        "{} files, {chunks} chunks ({total} embedded) in {:.1} s",
        index.files.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Refuse, before any embedding, what would stop `Index::write` once the
/// work is done: a temp file taken, or a directory that is missing or
/// cannot be written. The temp file made to find out is removed at once.
fn temp_is_free(index_path: &Path) -> Result<(), String> {
    let (temp, _file) = create_temp(index_path).map_err(|e| e.to_string())?;
    fs::remove_file(&temp).map_err(|e| format!("{}: {e}", temp.display()))
}

/// `INDEX.tmp` beside `index_path`, made new: whatever is there already, a
/// file of the user's or another run's, is in the way, never emptied or
/// followed. Errors name the temp file.
fn create_temp(index_path: &Path) -> io::Result<(PathBuf, File)> {
    let temp = index_path.with_added_extension("tmp");
    let made = File::options().write(true).create_new(true).open(&temp);
    let file = made.map_err(|e| {
        let message = match e.kind() {
            io::ErrorKind::AlreadyExists => format!(
                "{} is in the way: another index run writes it, or a killed one left it; \
                 move or remove it to index",
                temp.display()
            ),
            _ => format!("{}: {e}", temp.display()),
        };
        io::Error::new(e.kind(), message)
    })?;
    Ok((temp, file))
}

/// The files of the index at `index_path` when it indexed `root` with
/// `chunk_tokens`, none when there is no index there or it indexed something
/// else. A file there that is not an index is refused, never replaced,
/// except an empty regular file, which holds nothing to lose (a `mktemp`).
fn reusable_files(
    index_path: &Path,
    root: &Path,
    chunk_tokens: u32,
) -> Result<Vec<IndexedFile>, String> {
    if fs::metadata(index_path).is_ok_and(|meta| meta.is_file() && meta.len() == 0) {
        return Ok(Vec::new());
    }
    match Index::read(index_path) {
        Ok(index) if index.root == root && index.chunk_tokens == chunk_tokens => Ok(index.files),
        Ok(_) => Ok(Vec::new()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(format!("{}: {e}", index_path.display())),
    }
}

/// A chunk still to embed: its file's index, its lines and its text.
type Pending = (usize, (u32, u32), String);

/// The files under `root`: those unchanged since `old` with their chunks,
/// the others without, their chunks returned to embed.
fn scan(
    encoder: &Encoder,
    root: &Path,
    mut old: HashMap<String, IndexedFile>,
    chunk_tokens: u32,
) -> Result<(Vec<IndexedFile>, Vec<Pending>), String> {
    let mut files = Vec::new();
    let mut pending = Vec::new();
    for path in list_files(root)? {
        let (full, meta) = match file_in_tree(root, &path) {
            Ok(Some(found)) => found,
            Ok(None) => continue,
            Err(e) => {
                eprintln!("coderankembed: skipping {}: {e}", printable(&path));
                continue;
            }
        };
        if meta.len() > MAX_FILE_BYTES {
            continue;
        }
        let modified = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos());
        if let Some(previous) = old.remove(&path)
            && previous.size == meta.len()
            && previous.modified == modified
        {
            files.push(previous);
            continue;
        }
        let bytes = match fs::read(&full) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!("coderankembed: skipping {}: {e}", printable(&path));
                continue;
            }
        };
        // Binary files, and text in other encodings, are left out.
        let Ok(text) = String::from_utf8(bytes) else {
            continue;
        };
        if text.contains('\0') || text.trim().is_empty() {
            continue;
        }
        let index = files.len();
        for (lines, chunk) in chunks(encoder, &path, &text, chunk_tokens) {
            pending.push((index, lines, chunk));
        }
        files.push(IndexedFile {
            path,
            size: meta.len(),
            modified,
            chunks: Vec::new(),
        });
    }
    Ok((files, pending))
}

/// The canonical path and metadata of `path` under `root`, which is
/// canonical, when it is a regular file inside `root`: a link, or a
/// directory replaced by one, that leads out of the tree is not followed.
fn file_in_tree(root: &Path, path: &str) -> io::Result<Option<(PathBuf, fs::Metadata)>> {
    let full = match fs::canonicalize(root.join(path)) {
        Ok(full) => full,
        // Tracked by git but deleted from the work tree.
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    if !full.starts_with(root) {
        return Ok(None);
    }
    let meta = fs::metadata(&full)?;
    Ok(meta.is_file().then_some((full, meta)))
}

/// The `k` chunks most similar to `query`: score, file and chunk.
pub(crate) fn search<'a>(
    index: &'a Index,
    query: &[f32],
    k: usize,
) -> Vec<(f32, &'a IndexedFile, &'a Chunk)> {
    let query = &normalized(query);
    let mut hits: Vec<(f32, &IndexedFile, &Chunk)> = index
        .files
        .iter()
        .flat_map(|file| {
            file.chunks
                .iter()
                .map(move |chunk| (dot(query, &chunk.embedding), file, chunk))
        })
        .collect();
    let k = k.min(hits.len());
    if k > 0 {
        hits.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
    }
    hits.truncate(k);
    hits.sort_by(|a, b| b.0.total_cmp(&a.0));
    hits
}

/// `git -C root`, kept from running what the repository's own config
/// names: a `core.fsmonitor` command, or a lazy fetch from a promisor remote
/// whose URL can be a command. An unpacked archive can carry such a `.git`.
/// Its messages are in English, which `list_files` matches.
fn git(root: &Path) -> Command {
    let mut git = Command::new("git");
    git.env("GIT_NO_LAZY_FETCH", "1")
        .env("LC_ALL", "C")
        .args(["-c", "core.fsmonitor=false", "-C"])
        .arg(root);
    git
}

/// The text files under `root`, relative paths: what git tracks or would
/// track when `root` is in a work tree, else every file outside hidden files
/// and directories. In a work tree a failure of git, or git missing, is an
/// error, never a walk that would take in what git ignores.
fn list_files(root: &Path) -> Result<Vec<String>, String> {
    // Unlike `ls-files`, this does not read the index, so a damaged index
    // still counts as a work tree. Only git's own "no repository here" leads
    // to the walk: a repository it refuses, such as one of another owner,
    // is an error, never a walk into what it ignores.
    let probe = match git(root)
        .args(["rev-parse", "--is-inside-work-tree"])
        .output()
    {
        Ok(probe) => probe,
        Err(e) if e.kind() == io::ErrorKind::NotFound && in_work_tree(root) => {
            return Err(format!(
                "{} is in a git work tree, which needs git to list it",
                printable(&root.to_string_lossy())
            ));
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => return walk(root),
        Err(e) => return Err(format!("git: {e}")),
    };
    if !probe.status.success() {
        let stderr = String::from_utf8_lossy(&probe.stderr);
        if stderr.contains("not a git repository (or any") {
            return walk(root);
        }
        return Err(format!(
            "git in {}: {}",
            printable(&root.to_string_lossy()),
            printable(stderr.trim())
        ));
    }
    if !probe.stdout.starts_with(b"true") {
        return walk(root);
    }
    let out = git(root)
        .args([
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ])
        .output()
        .map_err(|e| format!("git: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "git ls-files in {}: {}",
            printable(&root.to_string_lossy()),
            printable(String::from_utf8_lossy(&out.stderr).trim())
        ));
    }
    let mut files: Vec<String> = out
        .stdout
        .split(|&b| b == 0)
        .filter(|p| !p.is_empty())
        .filter_map(|p| String::from_utf8(p.to_vec()).ok())
        .collect();
    files.sort();
    files.dedup();
    Ok(files)
}

/// Whether `root` or a directory above it holds `.git`, as the top of a
/// work tree does: what tells a work tree when git cannot.
fn in_work_tree(root: &Path) -> bool {
    root.ancestors()
        .any(|dir| dir.join(".git").symlink_metadata().is_ok())
}

/// Every file under `root` outside hidden files and directories, relative
/// paths; links are left out. A directory below `root` that cannot be read
/// is skipped with a message.
fn walk(root: &Path) -> Result<Vec<String>, String> {
    let mut files = Vec::new();
    let mut dirs = vec![PathBuf::new()];
    while let Some(dir) = dirs.pop() {
        let at = root.join(&dir);
        let shown = || printable(&at.to_string_lossy()).into_owned();
        let entries = match fs::read_dir(&at) {
            Ok(entries) => entries,
            Err(e) if !dir.as_os_str().is_empty() => {
                eprintln!("coderankembed: skipping {}: {e}", shown());
                continue;
            }
            Err(e) => return Err(format!("{}: {e}", shown())),
        };
        for entry in entries {
            let entry = entry.map_err(|e| format!("{}: {e}", shown()))?;
            let name = entry.file_name();
            if name.to_string_lossy().starts_with('.') {
                continue;
            }
            let path = dir.join(&name);
            let kind = entry
                .file_type()
                .map_err(|e| format!("{}: {e}", printable(&path.to_string_lossy())))?;
            if kind.is_dir() {
                dirs.push(path);
            } else if kind.is_file()
                && let Some(path) = path.to_str()
            {
                files.push(path.replace(std::path::MAIN_SEPARATOR, "/"));
            }
        }
    }
    files.sort();
    Ok(files)
}

/// `text` with what a terminal acts on escaped, so that a hostile file name
/// prints as itself: control and bidirectional formatting characters, and
/// the backslash that would make an escape ambiguous.
pub(crate) fn printable(text: &str) -> Cow<'_, str> {
    let acts = |c: char| {
        c.is_control()
            || c == '\\'
            || matches!(c, '\u{61c}' | '\u{200e}' | '\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}')
    };
    if !text.contains(acts) {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len() + 8);
    for c in text.chars() {
        if c == '\\' {
            out.push_str("\\\\");
        } else if acts(c) {
            out.extend(c.escape_unicode());
        } else {
            out.push(c);
        }
    }
    Cow::Owned(out)
}

/// `text` cut into runs of whole lines whose tokens, with the path's, stay
/// within `budget`; a single longer line is a chunk of its own, which the
/// encoder cuts to the budget.
fn chunks(encoder: &Encoder, path: &str, text: &str, budget: u32) -> Vec<((u32, u32), String)> {
    let budget = budget as usize;
    let overhead = encoder.count_tokens(path) + 2;
    let mut out = Vec::new();
    let mut chunk = String::new();
    let (mut first, mut last, mut tokens) = (1u32, 0u32, overhead);
    for (n, line) in (1u32..).zip(text.lines()) {
        let cost = encoder.count_tokens(line);
        if tokens > overhead && tokens + cost > budget {
            out.push(((first, n - 1), format!("{path} {chunk}")));
            chunk.clear();
            (first, tokens) = (n, overhead);
        }
        chunk.push_str(line);
        chunk.push('\n');
        tokens += cost;
        last = n;
    }
    if tokens > overhead {
        out.push(((first, last), format!("{path} {chunk}")));
    }
    out
}

fn normalized(v: &[f32]) -> Vec<f32> {
    let norm = dot(v, v).sqrt();
    let scale = if norm > 0.0 { 1.0 / norm } else { 0.0 };
    v.iter().map(|x| x * scale).collect()
}

/// Dot product in eight interleaved lanes, which LLVM keeps in one vector.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    let (a8, a_rest) = a.as_chunks::<8>();
    let (b8, b_rest) = b.as_chunks::<8>();
    for (x, y) in a8.iter().zip(b8) {
        for ((l, p), q) in lanes.iter_mut().zip(x).zip(y) {
            *l += p * q;
        }
    }
    let rest: f32 = a_rest.iter().zip(b_rest).map(|(p, q)| p * q).sum();
    lanes.iter().sum::<f32>() + rest
}

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

fn count(n: usize) -> io::Result<u32> {
    u32::try_from(n).map_err(|_| invalid("too many entries"))
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_bytes(r: &mut impl Read) -> io::Result<Vec<u8>> {
    let len = read_u32(r)? as usize;
    let mut bytes = Vec::new();
    r.take(len as u64).read_to_end(&mut bytes)?;
    if bytes.len() != len {
        return Err(io::ErrorKind::UnexpectedEof.into());
    }
    Ok(bytes)
}

fn read_string(r: &mut impl Read) -> io::Result<String> {
    String::from_utf8(read_bytes(r)?).map_err(|_| invalid("a path is not UTF-8"))
}

fn write_bytes(w: &mut impl Write, bytes: &[u8]) -> io::Result<()> {
    w.write_all(&count(bytes.len())?.to_le_bytes())?;
    w.write_all(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Index {
        let embedding = |seed: f32| (0..DIMENSION).map(|i| (i as f32 * seed).sin()).collect();
        Index {
            root: PathBuf::from("/src/ação"),
            chunk_tokens: 1024,
            files: vec![
                IndexedFile {
                    path: "a/b.rs".to_string(),
                    size: 12,
                    modified: u128::MAX - 5,
                    chunks: vec![
                        Chunk {
                            lines: (1, 40),
                            embedding: embedding(0.1),
                        },
                        Chunk {
                            lines: (41, 41),
                            embedding: embedding(0.2),
                        },
                    ],
                },
                IndexedFile {
                    path: "c.py".to_string(),
                    size: 0,
                    modified: 0,
                    chunks: Vec::new(),
                },
            ],
        }
    }

    /// A new, empty, canonical directory for test `name`.
    fn scratch_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("coderankembed-index-{name}-{}", std::process::id()));
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }
        fs::create_dir_all(&dir).unwrap();
        fs::canonicalize(dir).unwrap()
    }

    fn scratch(name: &str) -> PathBuf {
        scratch_dir(name).join("test.idx")
    }

    fn git(root: &Path, args: &[&str]) {
        let status = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(args)
            .status()
            .expect("git runs");
        assert!(status.success(), "git {args:?}");
    }

    /// A directory whose name is not UTF-8 is still the same directory when
    /// read back, so its unchanged files are reused.
    #[test]
    fn root_keeps_its_bytes() {
        let path = scratch("root-bytes");
        let mut index = sample();
        index.root = std::ffi::OsString::from_vec(b"/src/\xff".to_vec()).into();
        index.write(&path).unwrap();
        assert_eq!(Index::read(&path).unwrap().root, index.root);
        fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn index_round_trips() {
        let path = scratch("round-trip");
        let index = sample();
        index.write(&path).unwrap();
        let back = Index::read(&path).unwrap();
        assert_eq!(
            (&back.root, back.chunk_tokens),
            (&index.root, index.chunk_tokens)
        );
        assert_eq!(back.files.len(), index.files.len());
        for (a, b) in back.files.iter().zip(&index.files) {
            assert_eq!((&a.path, a.size, a.modified), (&b.path, b.size, b.modified));
            for (x, y) in a.chunks.iter().zip(&b.chunks) {
                assert_eq!((x.lines, &x.embedding), (y.lines, &y.embedding));
            }
            assert_eq!(a.chunks.len(), b.chunks.len());
        }
        assert!(!path.with_added_extension("tmp").exists());
        fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn temp_file_is_never_replaced() {
        use std::os::unix::fs::symlink;
        let dir = scratch_dir("temp");
        let path = dir.join("test.idx");
        let temp = path.with_added_extension("tmp");
        fs::write(&temp, "my notes\n").unwrap();
        assert!(sample().write(&path).is_err());
        assert_eq!(fs::read_to_string(&temp).unwrap(), "my notes\n");
        let victim = dir.join("victim.txt");
        fs::write(&victim, "kept\n").unwrap();
        fs::remove_file(&temp).unwrap();
        symlink(&victim, &temp).unwrap();
        assert!(sample().write(&path).is_err());
        assert_eq!(fs::read_to_string(&victim).unwrap(), "kept\n");
        assert!(!path.exists());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn search_returns_at_most_k() {
        let index = sample();
        let query = &index.files[0].chunks[1].embedding;
        assert!(search(&index, query, 0).is_empty());
        let mut index = index;
        let third = (0..DIMENSION).map(|i| (i as f32 * 0.3).cos()).collect();
        index.files[1].chunks.push(Chunk {
            lines: (1, 9),
            embedding: third,
        });
        let query = &index.files[0].chunks[1].embedding;
        let all = search(&index, query, 5);
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].2.lines, (41, 41));
        assert!(all.windows(2).all(|pair| pair[0].0 >= pair[1].0));
    }

    /// A write that fails takes away the temp file it made.
    #[test]
    fn failed_write_removes_its_temp() {
        let dir = scratch_dir("failed-write");
        let path = dir.join("test.idx");
        fs::create_dir_all(path.join("full")).unwrap();
        assert!(sample().write(&path).is_err());
        assert!(!path.with_added_extension("tmp").exists());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn taken_temp_stops_early() {
        let dir = scratch_dir("in-the-way");
        let path = dir.join("test.idx");
        assert!(temp_is_free(&path).is_ok());
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
        assert!(temp_is_free(&dir.join("missing/test.idx")).is_err());
        std::os::unix::fs::symlink("/nonexistent", path.with_added_extension("tmp")).unwrap();
        assert!(temp_is_free(&path).is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    /// Without git, a tree inside a work tree is refused rather than walked
    /// into what git ignores.
    #[test]
    fn work_tree_is_found_without_git() {
        let dir = scratch_dir("no-git");
        fs::create_dir_all(dir.join("repo/.git")).unwrap();
        fs::create_dir_all(dir.join("repo/src/deep")).unwrap();
        fs::create_dir_all(dir.join("worktree")).unwrap();
        fs::write(dir.join("worktree/.git"), "gitdir: /elsewhere\n").unwrap();
        assert!(in_work_tree(&dir.join("repo")));
        assert!(in_work_tree(&dir.join("repo/src/deep")));
        assert!(in_work_tree(&dir.join("worktree")));
        // Unlike a scratch directory, whose ancestors may be in a work tree.
        assert!(!in_work_tree(Path::new("/")));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn damaged_index_is_refused() {
        let path = scratch("damaged");
        sample().write(&path).unwrap();
        let bytes = fs::read(&path).unwrap();
        for damaged in [
            &bytes[..bytes.len() - 1],
            &bytes[..9],
            b"CRE-IDX0".as_slice(),
        ] {
            fs::write(&path, damaged).unwrap();
            assert!(Index::read(&path).is_err());
        }
        let mut longer = bytes;
        longer.push(0);
        fs::write(&path, longer).unwrap();
        assert!(Index::read(&path).is_err());
        fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn non_index_file_is_kept() {
        let dir = scratch_dir("reuse");
        let notes = dir.join("notes.txt");
        fs::write(&notes, "my notes\n").unwrap();
        assert!(reusable_files(&notes, &dir, 1024).is_err());
        assert_eq!(fs::read_to_string(&notes).unwrap(), "my notes\n");
        let empty = dir.join("empty.idx");
        fs::write(&empty, "").unwrap();
        assert!(reusable_files(&empty, &dir, 1024).unwrap().is_empty());
        let not_index = Index::read(&empty).err().map(|e| e.kind());
        assert_eq!(not_index, Some(io::ErrorKind::InvalidData));
        let missing = dir.join("missing.idx");
        assert!(reusable_files(&missing, &dir, 1024).unwrap().is_empty());
        let path = dir.join("test.idx");
        let index = sample();
        index.write(&path).unwrap();
        assert_eq!(reusable_files(&path, &index.root, 1024).unwrap().len(), 2);
        assert!(reusable_files(&path, &index.root, 512).unwrap().is_empty());
        assert!(reusable_files(&path, &dir, 1024).unwrap().is_empty());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn links_out_of_tree_are_skipped() {
        use std::os::unix::fs::symlink;
        let dir = scratch_dir("links");
        let (root, outside) = (dir.join("root"), dir.join("outside"));
        fs::create_dir_all(root.join("d")).unwrap();
        fs::create_dir(&outside).unwrap();
        fs::write(root.join("a.txt"), "a").unwrap();
        fs::write(outside.join("secret"), "s").unwrap();
        symlink("../outside/secret", root.join("leaf")).unwrap();
        symlink("a.txt", root.join("inner")).unwrap();
        // A tracked `d/secret` whose directory became a link out of the tree.
        fs::remove_dir(root.join("d")).unwrap();
        symlink("../outside", root.join("d")).unwrap();
        let found = |path: &str| file_in_tree(&root, path).unwrap().map(|(full, _)| full);
        assert_eq!(found("a.txt"), Some(root.join("a.txt")));
        assert_eq!(found("inner"), Some(root.join("a.txt")));
        assert_eq!(found("leaf"), None);
        assert_eq!(found("d/secret"), None);
        assert_eq!(found("gone.txt"), None);
        assert_eq!(walk(&root).unwrap(), ["a.txt"]);
        fs::remove_dir_all(dir).unwrap();
    }

    /// A repository's own config may name commands; listing it runs none.
    #[test]
    fn git_config_runs_nothing() {
        let root = scratch_dir("fsmonitor");
        git(&root, &["init", "-q"]);
        fs::write(root.join("a.txt"), "a").unwrap();
        fs::write(root.join(".gitignore"), "ignored/\n").unwrap();
        fs::create_dir(root.join("ignored")).unwrap();
        fs::write(root.join("ignored/b.txt"), "b").unwrap();
        git(&root, &["add", "a.txt", ".gitignore"]);
        let marker = root.join("ran");
        let hook = format!("touch '{}'; false", marker.display());
        git(&root, &["config", "core.fsmonitor", &hook]);
        assert_eq!(list_files(&root).unwrap(), [".gitignore", "a.txt"]);
        assert!(!marker.exists(), "core.fsmonitor ran");
        // A damaged index is git's error, not a walk into ignored files.
        fs::write(root.join(".git/index"), "garbage").unwrap();
        assert!(list_files(&root).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    /// A repository git refuses to read is an error, not a walk.
    #[test]
    fn refused_repository_is_an_error() {
        let root = scratch_dir("refused");
        fs::write(root.join("a.txt"), "a").unwrap();
        fs::write(root.join(".git"), "gitdir: /nonexistent/repository\n").unwrap();
        assert!(list_files(&root).is_err());
        fs::remove_file(root.join(".git")).unwrap();
        assert_eq!(list_files(&root).unwrap(), ["a.txt"]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn printable_escapes_terminal_controls() {
        for (name, shown) in [
            ("src/ação.rs", "src/ação.rs"),
            ("x\u{1b}]0;title\u{7}", "x\\u{1b}]0;title\\u{7}"),
            ("a\nfake 0.9999  b", "a\\u{a}fake 0.9999  b"),
            ("a\\b", "a\\\\b"),
            ("\u{202e}txt.exe", "\\u{202e}txt.exe"),
        ] {
            assert_eq!(printable(name), shown, "{name:?}");
        }
    }
}
