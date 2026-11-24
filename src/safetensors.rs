//! Reading the float tensors of a `.safetensors` file: eight bytes giving the
//! length of a JSON header, the header naming each tensor's type, shape and
//! byte range, then the data.
//!
//! The file is untrusted input: the header length is capped, the tensors'
//! ranges must tile the data, so that reading them all allocates at most the
//! file's length, and each range is checked against its tensor's shape before
//! anything is allocated for it.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::Error;

/// Largest header read: the model's is 12 KB, and each of the loader's
/// threads parses it into about 20 times its size.
const MAX_HEADER_BYTES: u64 = 64 << 10;

struct Entry {
    dtype: String,
    shape: Vec<usize>,
    start: u64,
    end: u64,
}

pub(crate) struct SafeTensors<R = File> {
    /// The file's path, for messages.
    path: PathBuf,
    reader: R,
    entries: HashMap<String, Entry>,
    /// Where the data starts: tensor ranges count from here.
    data: u64,
}

impl SafeTensors {
    pub(crate) fn open(path: &Path) -> Result<Self, Error> {
        let io = |e| Error::Io(path.to_path_buf(), e);
        let file = File::open(path).map_err(io)?;
        let len = file.metadata().map_err(io)?.len();
        Self::read(file, len, path)
    }
}

impl<R: Read + Seek> SafeTensors<R> {
    /// The header of the `len` bytes of `reader`, named `path` in messages.
    pub(crate) fn read(mut reader: R, len: u64, path: &Path) -> Result<Self, Error> {
        let io = |e| Error::Io(path.to_path_buf(), e);
        let mut prefix = [0u8; 8];
        reader.read_exact(&mut prefix).map_err(io)?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len > MAX_HEADER_BYTES || header_len > len.saturating_sub(8) {
            return Err(Error::model(
                path,
                format!("a header of {header_len} bytes"),
            ));
        }
        let mut header = vec![0u8; usize::try_from(header_len).map_err(|e| Error::model(path, e))?];
        reader.read_exact(&mut header).map_err(io)?;
        let header: serde_json::Value = serde_json::from_slice(&header)
            .map_err(|e| Error::model(path, format!("header: {e}")))?;
        let object = header
            .as_object()
            .ok_or_else(|| Error::model(path, "the header is not an object"))?;
        let mut entries = HashMap::with_capacity(object.len());
        for (name, value) in object {
            if name == "__metadata__" {
                continue;
            }
            let entry = parse_entry(value)
                .ok_or_else(|| Error::model(path, format!("tensor {name} is malformed")))?;
            entries.insert(name.clone(), entry);
        }
        let data_len = len - 8 - header_len;
        let mut ranges: Vec<(u64, u64)> = entries.values().map(|e| (e.start, e.end)).collect();
        ranges.sort_unstable();
        let mut at = 0;
        for (start, end) in ranges {
            if start != at || end < start {
                return Err(Error::model(path, "the tensors do not tile the data"));
            }
            at = end;
        }
        if at != data_len {
            return Err(Error::model(path, "the tensors do not tile the data"));
        }
        Ok(Self {
            path: path.to_path_buf(),
            reader,
            entries,
            data: 8 + header_len,
        })
    }

    /// How many tensors the header lists.
    pub(crate) fn tensor_count(&self) -> usize {
        self.entries.len()
    }

    /// Each tensor the header lists and its shape, by name.
    #[cfg(fuzzing)]
    pub(crate) fn listed(&self) -> Vec<(String, Vec<usize>)> {
        let mut listed: Vec<_> = self
            .entries
            .iter()
            .map(|(name, entry)| (name.clone(), entry.shape.clone()))
            .collect();
        listed.sort();
        listed
    }

    /// The 32-bit float tensor `name`, which must have shape `shape`.
    pub(crate) fn floats(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, Error> {
        let path = &self.path;
        let entry = self
            .entries
            .get(name)
            .ok_or_else(|| Error::model(path, format!("no tensor {name}")))?;
        if entry.dtype != "F32" || entry.shape != shape {
            return Err(Error::model(
                path,
                format!(
                    "{name} is {} {:?}, not F32 {shape:?}",
                    entry.dtype, entry.shape
                ),
            ));
        }
        let too_large = || Error::model(path, format!("{name} is too large"));
        let count = shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(too_large)?;
        let bytes = count
            .checked_mul(4)
            .and_then(|b| u64::try_from(b).ok())
            .ok_or_else(too_large)?;
        // `read` checked that the ranges tile the data.
        if entry.end.checked_sub(entry.start) != Some(bytes) {
            return Err(Error::model(
                path,
                format!("{name} spans other than its shape's bytes"),
            ));
        }
        let io = |e| Error::Io(path.clone(), e);
        self.reader
            .seek(SeekFrom::Start(self.data + entry.start))
            .map_err(io)?;
        let mut values = vec![0.0f32; count];
        let mut chunk = vec![0u8; 1 << 20];
        for out in values.chunks_mut(chunk.len() / 4) {
            let raw = &mut chunk[..out.len() * 4];
            self.reader.read_exact(raw).map_err(io)?;
            for (v, b) in out.iter_mut().zip(raw.as_chunks::<4>().0) {
                *v = f32::from_le_bytes(*b);
            }
        }
        Ok(values)
    }
}

fn parse_entry(value: &serde_json::Value) -> Option<Entry> {
    let numbers = |key: &str| -> Option<Vec<u64>> {
        value
            .get(key)?
            .as_array()?
            .iter()
            .map(serde_json::Value::as_u64)
            .collect()
    };
    let shape = numbers("shape")?
        .into_iter()
        .map(|d| usize::try_from(d).ok())
        .collect::<Option<Vec<_>>>()?;
    let offsets = numbers("data_offsets")?;
    let [start, end] = offsets.as_slice() else {
        return None;
    };
    Some(Entry {
        dtype: value.get("dtype")?.as_str()?.to_string(),
        shape,
        start: *start,
        end: *end,
    })
}

#[cfg(test)]
mod tests {
    use std::assert_matches;

    use super::*;

    fn write(dir: &Path, header: &str, data: &[u8]) -> PathBuf {
        let path = dir.join("t.safetensors");
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(data);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("coderankembed-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn tensor_read_refuses_bad_ranges() {
        let dir = scratch_dir("safetensors");
        let tensor_bytes: Vec<u8> = [1.5f32, -2.0, 3.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let good = r#"{"__metadata__":{"format":"pt"},"w":{"dtype":"F32","shape":[3],"data_offsets":[0,12]}}"#;
        let mut file = SafeTensors::open(&write(&dir, good, &tensor_bytes)).unwrap();
        assert_eq!(file.floats("w", &[3]).unwrap(), [1.5, -2.0, 3.25]);
        assert!(file.floats("w", &[1, 3]).is_err());
        let short = r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,12]}}"#;
        let mut file = SafeTensors::open(&write(&dir, short, &tensor_bytes)).unwrap();
        assert!(file.floats("w", &[2]).is_err());
        assert!(file.floats("x", &[3]).is_err());
        let past_end = r#"{"w":{"dtype":"F32","shape":[3],"data_offsets":[0,16]}}"#;
        assert!(SafeTensors::open(&write(&dir, past_end, &tensor_bytes)).is_err());
        let huge = r#"{"w":{"dtype":"F32","shape":[4611686018427387904,4],"data_offsets":[0,12]}}"#;
        let mut file = SafeTensors::open(&write(&dir, huge, &tensor_bytes)).unwrap();
        assert!(file.floats("w", &[4_611_686_018_427_387_904, 4]).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// Tensors sharing bytes would let a small file ask for one allocation
    /// per tensor of the same data; the format tiles its data exactly.
    #[test]
    fn tensors_must_tile_the_data() {
        let dir = scratch_dir("tiling");
        let zeros = [0u8; 12];
        let tensor = |start: u32, end: u32| {
            format!(
                r#"{{"dtype":"F32","shape":[{}],"data_offsets":[{start},{end}]}}"#,
                (end - start) / 4
            )
        };
        let header = |a: String, b: String| format!(r#"{{"a":{a},"b":{b}}}"#);
        let tiled = header(tensor(0, 4), tensor(4, 12));
        assert!(SafeTensors::open(&write(&dir, &tiled, &zeros)).is_ok());
        for refused in [
            header(tensor(0, 12), tensor(0, 12)),
            header(tensor(0, 8), tensor(4, 12)),
            header(tensor(0, 4), tensor(8, 12)),
            header(tensor(0, 4), tensor(4, 8)),
        ] {
            let opened = SafeTensors::open(&write(&dir, &refused, &zeros));
            assert_matches!(opened.err(), Some(Error::Model(_)), "{refused}");
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// At the cap the header is read, and found short; past it, refused
    /// before anything is allocated for it.
    #[test]
    fn header_length_is_capped() {
        let read = |header_len: u64| {
            let prefix = std::io::Cursor::new(header_len.to_le_bytes());
            SafeTensors::read(prefix, u64::MAX, Path::new("t")).err()
        };
        assert_matches!(read(MAX_HEADER_BYTES), Some(Error::Io(..)));
        assert_matches!(read(MAX_HEADER_BYTES + 1), Some(Error::Model(_)));
    }

    #[test]
    fn refuses_malformed_headers() {
        let dir = scratch_dir("header");
        let path = dir.join("t.safetensors");
        std::fs::write(&path, 1000u64.to_le_bytes()).unwrap();
        assert!(SafeTensors::open(&path).is_err());
        std::fs::write(&path, [0u8; 3]).unwrap();
        assert!(SafeTensors::open(&path).is_err());
        for header in [
            "{\"w\": [",
            "[1, 2]",
            "{\"w\": {\"dtype\": \"F32\", \"shape\": [-1], \"data_offsets\": [0, 0]}}",
        ] {
            write(&dir, header, &[]);
            assert!(SafeTensors::open(&path).is_err(), "{header}");
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}
