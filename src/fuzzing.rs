//! The bodies of the fuzz targets in `fuzz/`, compiled only by `cargo fuzz`,
//! which sets `cfg(fuzzing)`: the readers of the model's files and the
//! tokenizer, fed bytes they did not write. Malformed input must come back
//! as an error; the assertions are what a well-formed result guarantees.

use std::io::Cursor;
use std::path::Path;
use std::sync::OnceLock;

use crate::safetensors::SafeTensors;
use crate::tokenizer::Tokenizer;

/// Inputs past this are cut, so that a run spends its time on structure.
const MAX_INPUT_BYTES: usize = 1 << 20;

/// A `.safetensors` file: its header, then each tensor it lists, read as the
/// shape it claims.
pub fn safetensors(bytes: &[u8]) {
    let bytes = &bytes[..bytes.len().min(MAX_INPUT_BYTES)];
    let len = u64::try_from(bytes.len()).expect("a slice length fits u64");
    let Ok(mut file) = SafeTensors::read(Cursor::new(bytes), len, Path::new("fuzz")) else {
        return;
    };
    // Every tensor read: together no more floats than the data holds.
    let header: [u8; 8] = bytes[..8]
        .try_into()
        .expect("an opened file has a header length");
    let header = usize::try_from(u64::from_le_bytes(header)).expect("a header within the input");
    let data_len = bytes.len() - 8 - header;
    let mut read = 0;
    for (name, shape) in file.listed() {
        if let Ok(values) = file.floats(&name, &shape) {
            assert_eq!(values.len(), shape.iter().product::<usize>());
            read += 4 * values.len();
        }
    }
    assert!(read <= data_len, "more floats than the data holds");
}

/// A `tokenizer.json`, then a few texts through the tokenizer it describes.
pub fn tokenizer_json(bytes: &[u8]) {
    let bytes = &bytes[..bytes.len().min(MAX_INPUT_BYTES)];
    let Ok(tokenizer) = Tokenizer::parse(bytes) else {
        return;
    };
    for text in ["", "def f(x): return x", "Ünïcödé 中文 [SEP]x"] {
        let ids = tokenizer.encode(text, 16);
        assert!((2..=16).contains(&ids.len()), "{} ids", ids.len());
        assert!(ids.iter().all(|&id| id <= tokenizer.max_id()));
    }
}

/// Text through a small BERT tokenizer: cut to the limit, `[CLS]` and
/// `[SEP]` around it, and as many tokens split at a space as whole.
pub fn tokenize(bytes: &[u8]) {
    static TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();
    let Ok(text) = std::str::from_utf8(&bytes[..bytes.len().min(MAX_INPUT_BYTES)]) else {
        return;
    };
    let tokenizer = TOKENIZER.get_or_init(|| {
        Tokenizer::parse(include_bytes!("../fuzz/seeds/tokenizer_json/bert.json"))
            .expect("the seed tokenizer parses")
    });
    let max = 64;
    let ids = tokenizer.encode(text, max);
    let whole = tokenizer.count(text);
    assert_eq!(ids.len(), (whole + 2).min(max));
    assert_eq!((ids.first(), ids.last()), (Some(&2), Some(&3)));
    if let Some(at) = text.find(' ') {
        let (left, right) = text.split_at(at);
        assert_eq!(tokenizer.count(left) + tokenizer.count(right), whole);
    }
}
