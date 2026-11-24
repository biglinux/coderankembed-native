//! The BERT WordPiece tokenizer of `tokenizer.json`, as Hugging Face's
//! `tokenizers` runs it: special tokens matched in the raw text, the BERT
//! normalizer, splitting on whitespace and punctuation, then greedy
//! longest-match WordPiece, between `[CLS]` and `[SEP]`.
//!
//! The Unicode categories and canonical decompositions come from the crates
//! `tokenizers` itself uses, so the ids match its own.

use std::collections::HashMap;
use std::path::Path;

use unicode_categories::UnicodeCategories;
use unicode_normalization::char::decompose_canonical;

use crate::Error;

pub(crate) struct Tokenizer {
    vocab: HashMap<String, u32>,
    /// Tokens matched in the text before normalization, as written.
    specials: Vec<(String, u32)>,
    unk: u32,
    cls: u32,
    sep: u32,
    prefix: String,
    max_word_chars: usize,
    clean_text: bool,
    chinese_chars: bool,
    strip_accents: bool,
    lowercase: bool,
}

impl Tokenizer {
    pub(crate) fn load(path: &Path) -> Result<Self, Error> {
        let text = std::fs::read(path).map_err(|e| Error::Io(path.to_path_buf(), e))?;
        Self::parse(&text).map_err(|msg| Error::model(path, msg))
    }

    pub(crate) fn parse(text: &[u8]) -> Result<Self, String> {
        let json: serde_json::Value = serde_json::from_slice(text).map_err(|e| e.to_string())?;
        let kind = |key: &str| {
            json.get(key)
                .and_then(|v| v.get("type"))
                .and_then(|v| v.as_str())
        };
        if kind("normalizer") != Some("BertNormalizer")
            || kind("pre_tokenizer") != Some("BertPreTokenizer")
            || kind("model") != Some("WordPiece")
        {
            return Err("not a BERT WordPiece tokenizer".to_string());
        }
        let normalizer = &json["normalizer"];
        let flag = |key: &str| normalizer.get(key).and_then(serde_json::Value::as_bool);
        let lowercase = flag("lowercase").ok_or("normalizer.lowercase is missing")?;
        let model = &json["model"];
        let text_of = |key: &str| -> Result<String, String> {
            Ok(model
                .get(key)
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("model.{key} is missing"))?
                .to_string())
        };
        let mut vocab = HashMap::new();
        for (token, id) in model
            .get("vocab")
            .and_then(|v| v.as_object())
            .ok_or("model.vocab is missing")?
        {
            let id = id
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| format!("token {token} has no id"))?;
            vocab.insert(token.clone(), id);
        }
        let id_of = |token: &str| {
            vocab
                .get(token)
                .copied()
                .ok_or_else(|| format!("{token} is not in the vocabulary"))
        };
        let unk = id_of(&text_of("unk_token")?)?;
        let (cls, sep) = (id_of("[CLS]")?, id_of("[SEP]")?);
        let mut specials = Vec::new();
        for token in json
            .get("added_tokens")
            .and_then(|v| v.as_array())
            .ok_or("added_tokens is missing")?
        {
            let content = token.get("content").and_then(|v| v.as_str());
            let id = token.get("id").and_then(serde_json::Value::as_u64);
            let special = token.get("special").and_then(serde_json::Value::as_bool);
            let normalized = token.get("normalized").and_then(serde_json::Value::as_bool);
            // Matching that strips whitespace or wants whole words is not run here.
            let stripping = ["lstrip", "rstrip", "single_word"]
                .iter()
                .any(|key| token.get(key).and_then(serde_json::Value::as_bool) == Some(true));
            match (content, id, special, normalized) {
                (Some(content), Some(id), Some(true), Some(false))
                    if !content.is_empty() && !stripping =>
                {
                    let id = u32::try_from(id).map_err(|e| e.to_string())?;
                    specials.push((content.to_string(), id));
                }
                _ => return Err(format!("unsupported added token {token}")),
            }
        }
        // The longest match wins where one special token starts another.
        specials.sort_by_key(|(content, _)| std::cmp::Reverse(content.len()));
        Ok(Self {
            vocab,
            specials,
            unk,
            cls,
            sep,
            prefix: text_of("continuing_subword_prefix")?,
            max_word_chars: model
                .get("max_input_chars_per_word")
                .and_then(serde_json::Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or("model.max_input_chars_per_word is missing")?,
            clean_text: flag("clean_text").unwrap_or(true),
            chinese_chars: flag("handle_chinese_chars").unwrap_or(true),
            strip_accents: flag("strip_accents").unwrap_or(lowercase),
            lowercase,
        })
    }

    /// The largest id the tokenizer gives.
    pub(crate) fn max_id(&self) -> u32 {
        let vocab = self.vocab.values().copied();
        vocab
            .chain(self.specials.iter().map(|(_, id)| *id))
            .max()
            .unwrap_or(0)
    }

    /// The ids of `text` between `[CLS]` and `[SEP]`, its tokens cut after
    /// the first `max_tokens - 2`.
    ///
    /// # Panics
    /// When `max_tokens` is below 2.
    pub(crate) fn encode(&self, text: &str, max_tokens: usize) -> Vec<u32> {
        assert!(max_tokens >= 2, "no room for [CLS] and [SEP]");
        let mut ids = vec![self.cls];
        self.tokenize(text, max_tokens - 1, &mut ids);
        ids.push(self.sep);
        ids
    }

    /// Tokens of `text`, without `[CLS]` and `[SEP]`. Text split at
    /// whitespace has as many tokens as its parts together.
    pub(crate) fn count(&self, text: &str) -> usize {
        let mut ids = Vec::new();
        self.tokenize(text, usize::MAX, &mut ids);
        ids.len()
    }

    /// Append the ids of `text` to `ids`, stopping once it holds `limit`.
    fn tokenize(&self, text: &str, limit: usize, ids: &mut Vec<u32>) {
        let mut word = String::new();
        let mut rest = text;
        while !rest.is_empty() && ids.len() < limit {
            let (plain, special) = self.next_special(rest);
            self.split(&rest[..plain], &mut word, limit, ids);
            let Some((len, id)) = special else { break };
            if ids.len() < limit {
                ids.push(id);
            }
            rest = &rest[plain + len..];
        }
        ids.truncate(limit);
    }

    /// Bytes of `text` before its first special token, and that token's
    /// length and id.
    fn next_special(&self, text: &str) -> (usize, Option<(usize, u32)>) {
        for (at, _) in text.char_indices() {
            for (content, id) in &self.specials {
                if text[at..].starts_with(content.as_str()) {
                    return (at, Some((content.len(), *id)));
                }
            }
        }
        (text.len(), None)
    }

    /// Normalize `text` and split it into words at whitespace and around
    /// punctuation, appending each word's pieces.
    fn split(&self, text: &str, word: &mut String, limit: usize, ids: &mut Vec<u32>) {
        word.clear();
        for c in text.chars() {
            if ids.len() >= limit {
                return;
            }
            if self.clean_text {
                if c == '\0'
                    || c == '\u{fffd}'
                    || (c.is_other() && !matches!(c, '\t' | '\n' | '\r'))
                {
                    continue;
                }
                if c.is_whitespace() {
                    self.flush(word, ids);
                    continue;
                }
            }
            if self.chinese_chars && is_chinese_char(c) {
                self.flush(word, ids);
                self.push_normalized(c, word);
                self.flush(word, ids);
                continue;
            }
            self.push_normalized(c, word);
        }
        self.flush(word, ids);
    }

    /// Append `c` as the normalizer leaves it: without its accents and
    /// lowercased, when the tokenizer asks for that.
    fn push_normalized(&self, c: char, word: &mut String) {
        let mut push = |c: char| {
            if self.lowercase {
                word.extend(c.to_lowercase());
            } else {
                word.push(c);
            }
        };
        if self.strip_accents {
            decompose_canonical(c, |d| {
                if !d.is_mark_nonspacing() {
                    push(d);
                }
            });
        } else {
            push(c);
        }
    }

    /// Split the normalized `word` at whitespace and around punctuation and
    /// append the pieces of each part.
    fn flush(&self, word: &mut String, ids: &mut Vec<u32>) {
        let mut start = 0;
        for (at, c) in word.char_indices() {
            if c.is_whitespace() || c.is_ascii_punctuation() || c.is_punctuation() {
                self.pieces(&word[start..at], ids);
                if !c.is_whitespace() {
                    self.pieces(&word[at..at + c.len_utf8()], ids);
                }
                start = at + c.len_utf8();
            }
        }
        self.pieces(&word[start..], ids);
        word.clear();
    }

    /// Greedy longest-match-first WordPiece of one word: `[UNK]` for the
    /// whole word when a part matches nothing.
    fn pieces(&self, word: &str, ids: &mut Vec<u32>) {
        if word.is_empty() {
            return;
        }
        if word.chars().count() > self.max_word_chars {
            ids.push(self.unk);
            return;
        }
        let first = ids.len();
        let mut piece = String::new();
        let mut start = 0;
        while start < word.len() {
            let mut end = word.len();
            let found = loop {
                piece.clear();
                if start > 0 {
                    piece.push_str(&self.prefix);
                }
                piece.push_str(&word[start..end]);
                if let Some(&id) = self.vocab.get(&piece) {
                    break Some(id);
                }
                end = word.floor_char_boundary(end - 1);
                if end <= start {
                    break None;
                }
            };
            let Some(id) = found else {
                ids.truncate(first);
                ids.push(self.unk);
                return;
            };
            ids.push(id);
            start = end;
        }
    }
}

/// The CJK blocks BERT puts spaces around.
fn is_chinese_char(c: char) -> bool {
    matches!(
        u32::from(c),
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B920..=0x2CEAF
            | 0xF900..=0xFAFF
            | 0x2F800..=0x2FA1F
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_json() -> String {
        let vocab = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "def", "add", "##er", "(", ")", ":", "a", "##b",
            "e", "中", "σ",
        ];
        let vocab: Vec<String> = vocab
            .iter()
            .enumerate()
            .map(|(i, t)| format!("\"{t}\": {i}"))
            .collect();
        format!(
            r###"{{"added_tokens": [{{"id": 2, "content": "[CLS]", "special": true, "normalized": false}},
                {{"id": 3, "content": "[SEP]", "special": true, "normalized": false}}],
              "normalizer": {{"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true,
                "strip_accents": null, "lowercase": true}},
              "pre_tokenizer": {{"type": "BertPreTokenizer"}},
              "model": {{"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 8, "vocab": {{{}}}}}}}"###,
            vocab.join(",")
        )
    }

    fn tiny() -> Tokenizer {
        Tokenizer::parse(tiny_json().as_bytes()).unwrap()
    }

    /// An empty special token would match at every position of a text, and
    /// stripping or whole-word matching is not run.
    #[test]
    fn unsupported_added_tokens_are_refused() {
        let sep = r#""content": "[SEP]""#;
        let json = tiny_json();
        assert!(json.contains(sep));
        for other in [
            r#""content": """#,
            r#""content": "[SEP]", "lstrip": true"#,
            r#""content": "[SEP]", "rstrip": true"#,
            r#""content": "[SEP]", "single_word": true"#,
        ] {
            let refused = json.replace(sep, other);
            assert!(Tokenizer::parse(refused.as_bytes()).is_err(), "{other}");
        }
        let plain = json.replace(sep, r#""content": "[SEP]", "lstrip": false"#);
        assert!(Tokenizer::parse(plain.as_bytes()).is_ok());
    }

    #[test]
    fn splits_normalizes_and_matches_pieces() {
        let t = tiny();
        // DEF ADDER(Ab): lowercased, split at punctuation, "adder" as two pieces.
        assert_eq!(
            t.encode("DEF ADDER(Ab):", 64),
            [2, 4, 5, 6, 7, 10, 11, 8, 9, 3]
        );
        // Accents stripped, a control character dropped, CJK isolated.
        assert_eq!(t.encode("É\u{7}中a", 64), [2, 12, 13, 10, 3]);
        // A word with no full match, and one past the length limit.
        assert_eq!(t.encode("abz aaaaaaaaa", 64), [2, 1, 1, 3]);
        // Specials in the text, final sigma lowercased one character at a time.
        assert_eq!(t.encode("a[SEP]Σ", 64), [2, 10, 3, 14, 3]);
    }

    #[test]
    fn truncates_and_counts() {
        let t = tiny();
        assert_eq!(t.encode("def add def add", 4), [2, 4, 5, 3]);
        assert_eq!(t.count("def adder\n(a)"), 6);
        assert_eq!(t.count("def adder") + t.count("(a)"), 6);
    }
}
