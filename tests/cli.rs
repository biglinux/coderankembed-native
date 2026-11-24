//! The command line's contract: `--help` names every option and variable,
//! and each exit status the help lists comes back for its case. None of
//! these runs loads the model but the ignored one.

use std::process::{Command, Output};

fn coderankembed(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_coderankembed"))
        .args(args)
        .env_remove("CODERANKEMBED_MODEL")
        .output()
        .expect("the binary runs")
}

#[test]
fn help_lists_every_option() {
    for flag in ["--help", "-h"] {
        let out = coderankembed(&[flag]);
        assert_eq!(out.status.code(), Some(0), "{flag}");
        let help = String::from_utf8(out.stdout).expect("UTF-8 help");
        // Every option `parse` in src/bin/coderankembed/main.rs accepts.
        for name in [
            "embed",
            "index",
            "search",
            "bench",
            "--query",
            "--lines",
            "--raw",
            "--chunk-tokens",
            "--top",
            "--tokens",
            "--batch",
            "--rounds",
            "--model",
            "--device",
            "--threads",
            "--max-tokens",
            "CODERANKEMBED_MODEL",
            "CODERANKEMBED_TIER",
        ] {
            assert!(help.contains(name), "{flag} does not name {name}");
        }
    }
    assert_eq!(coderankembed(&["embed", "--help"]).status.code(), Some(0));
}

#[test]
fn wrong_command_lines_exit_2() {
    for args in [
        &[][..],
        &["frobnicate"],
        &["embed", "--bogus"],
        &["embed", "--threads", "x"],
        &["embed", "--max-tokens", "1"],
        &["search"],
        &["search", "an.idx"],
        &["index", "only-a-dir"],
        &["bench", "extra"],
        &["bench", "--tokens", "8193"],
        // Options of another command, which this one would ignore.
        &["embed", "--top", "3"],
        &["search", "--chunk-tokens", "64", "an.idx", "query"],
        &["index", "--query", "a-dir", "an.idx"],
        &["index", "--max-tokens", "64", "a-dir", "an.idx"],
        &["bench", "--raw"],
    ] {
        let out = coderankembed(args);
        assert_eq!(out.status.code(), Some(2), "{args:?}");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("usage:"), "{args:?}: {stderr}");
    }
}

#[test]
fn bad_tier_exits_2() {
    let out = Command::new(env!("CARGO_BIN_EXE_coderankembed"))
        .args(["embed", "--model", "/nonexistent/model"])
        .env("CODERANKEMBED_TIER", "avx")
        .output()
        .expect("the binary runs");
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn unknown_command_is_named() {
    let out = coderankembed(&["frobnicate", "--top", "3"]);
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("unknown command frobnicate"), "{stderr}");
}

/// A file that cannot be read fails before the model loads.
#[test]
fn missing_file_fails_before_the_model() {
    let out = coderankembed(&[
        "embed",
        "--model",
        "/nonexistent/model",
        "/nonexistent/text",
    ]);
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("/nonexistent/text"), "{stderr}");
}

#[test]
fn missing_index_exits_1() {
    let out = coderankembed(&["search", "/nonexistent/coderankembed.idx", "query"]);
    assert_eq!(out.status.code(), Some(1));
}

#[test]
fn missing_model_skips_cpu_fallback() {
    let out = coderankembed(&["embed", "--model", "/nonexistent/model"]);
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    // Only a GPU failure moves `--device auto` to the CPU.
    assert!(!stderr.contains("on the CPU"), "{stderr}");
    assert_eq!(stderr.matches("config.json").count(), 1, "{stderr}");
}

/// A reader that stops early, as `head` does, is no failure.
#[test]
fn closed_output_exits_0() {
    let (reader, writer) = std::io::pipe().expect("a pipe");
    drop(reader);
    let status = Command::new(env!("CARGO_BIN_EXE_coderankembed"))
        .arg("--help")
        .stdout(writer)
        .status()
        .expect("the binary runs");
    assert_eq!(status.code(), Some(0));
}

/// Without git, a tree in a work tree is refused, never walked into what
/// git ignores.
#[test]
#[ignore = "needs the CodeRankEmbed weights"]
fn gitless_index_refuses_work_tree() {
    let dir = std::env::temp_dir().join(format!("coderankembed-cli-no-git-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("tree/.git")).expect("a scratch tree");
    std::fs::write(dir.join("tree/a.txt"), "a\n").expect("a scratch file");
    let out = Command::new(env!("CARGO_BIN_EXE_coderankembed"))
        .args(["index", "--device", "cpu"])
        .arg(dir.join("tree"))
        .arg(dir.join("tree.idx"))
        .env("PATH", "")
        .output()
        .expect("the binary runs");
    std::fs::remove_dir_all(&dir).expect("the scratch tree goes");
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("needs git"), "{stderr}");
}
