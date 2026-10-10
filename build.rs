//! Build script: prepares the learning guide for rustdoc.
//!
//! The guide chapters in `documentation/guide/` use plain LaTeX maths so that
//! they render in Markdown previews, on GitHub and in the mdBook (`book/`).
//! Rustdoc parses Markdown before KaTeX runs and would mangle that LaTeX, so
//! this script writes a copy of each chapter to `$OUT_DIR/guide/` with every
//! maths span escaped for rustdoc (see `build_support/rustdoc_math.rs`), and
//! with links between chapters (`02_quantisation_dither.md`, which work in a
//! Markdown viewer and in the book) turned into intra-doc links
//! (`crate::guide::c02_quantisation_dither`).
//! `src/guide.rs` includes those copies. Nothing is done unless the `guide`
//! module is being built (features `educational` and `full_no_plotting`).

#[path = "build_support/rustdoc_math.rs"]
#[allow(dead_code)]
mod rustdoc_math;

use std::path::PathBuf;
use std::{env, fs};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build_support/rustdoc_math.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_EDUCATIONAL");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_FULL_NO_PLOTTING");
    if env::var_os("CARGO_FEATURE_EDUCATIONAL").is_none()
        || env::var_os("CARGO_FEATURE_FULL_NO_PLOTTING").is_none()
    {
        return;
    }

    let src_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"))
        .join("documentation")
        .join("guide");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR")).join("guide");
    fs::create_dir_all(&out_dir).expect("create $OUT_DIR/guide");
    println!("cargo:rerun-if-changed={}", src_dir.display());

    let mut chapters: Vec<PathBuf> = fs::read_dir(&src_dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", src_dir.display()))
        .map(|entry| entry.expect("read guide directory entry").path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("md"))
        .collect();
    chapters.sort();
    assert!(
        !chapters.is_empty(),
        "no guide chapters found in {}",
        src_dir.display()
    );
    let links: Vec<(String, String)> = chapters
        .iter()
        .map(|p| {
            let file = p
                .file_name()
                .and_then(|f| f.to_str())
                .expect("UTF-8 file name");
            (format!("]({file}"), format!("]({}", rustdoc_path(file)))
        })
        .collect();

    for path in &chapters {
        println!("cargo:rerun-if-changed={}", path.display());
        let text =
            fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let mut escaped =
            rustdoc_math::escape_math(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        for (from, to) in &links {
            escaped = escaped.replace(from.as_str(), to);
        }
        let name = path.file_name().expect("file name");
        fs::write(out_dir.join(name), escaped)
            .unwrap_or_else(|e| panic!("write {}: {e}", out_dir.display()));
    }
}

/// The rustdoc path of the guide page made from the chapter file `file`:
/// `README.md` is the `guide` module itself, `FOR_INSTRUCTORS.md` is
/// `guide::for_instructors` and `NN_topic.md` is `guide::cNN_topic`
/// (the module names in `src/guide.rs`).
fn rustdoc_path(file: &str) -> String {
    let stem = file.strip_suffix(".md").expect(".md file");
    match stem {
        "README" => "crate::guide".to_string(),
        "FOR_INSTRUCTORS" => "crate::guide::for_instructors".to_string(),
        _ => format!("crate::guide::c{stem}"),
    }
}
