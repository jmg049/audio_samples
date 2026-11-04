//! The book's gallery chapter lists every example, with the features it needs.
//!
//! `book/build.sh` compiles and runs the examples; this test keeps the chapter
//! and `Cargo.toml` from drifting apart without building the book.

use std::collections::BTreeMap;

/// `(name, required features)` of each `[[example]]` in `Cargo.toml`.
fn declared_examples() -> BTreeMap<String, Vec<String>> {
    let manifest = include_str!("../Cargo.toml");
    let mut examples = BTreeMap::new();
    for block in manifest.split("[[example]]").skip(1) {
        let block = block.split("\n[").next().unwrap_or(block);
        let value = |key: &str| {
            block
                .lines()
                .find(|l| l.trim_start().starts_with(key))
                .and_then(|l| l.split_once('='))
                .map(|(_, v)| v.trim().to_owned())
        };
        let name = value("name").expect("an example has a name");
        let features = value("required-features").unwrap_or_default();
        let features = features
            .trim_matches(|c| c == '[' || c == ']')
            .split(',')
            .map(|f| f.trim().trim_matches('"').to_owned())
            .filter(|f| !f.is_empty())
            .collect();
        examples.insert(name.trim_matches('"').to_owned(), features);
    }
    examples
}

#[test]
fn gallery_lists_every_example_with_its_features() {
    let chapter = include_str!("../book/src/gallery.md");
    let examples = declared_examples();
    assert!(examples.len() >= 45, "found {} examples", examples.len());
    for (name, features) in &examples {
        let mut command = format!("cargo run --example {name}");
        if !features.is_empty() {
            command.push_str(&format!(" --features {}", features.join(",")));
        }
        assert!(
            chapter.lines().any(|l| l == command),
            "book/src/gallery.md must show `{command}`"
        );
    }
}

#[test]
fn every_example_file_is_declared() {
    let examples = declared_examples();
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/examples");
    for entry in std::fs::read_dir(dir).expect("examples directory") {
        let path = entry.expect("directory entry").path();
        if path.extension().is_some_and(|e| e == "rs") {
            let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
            assert!(
                examples.contains_key(&stem),
                "{stem}.rs has no [[example]] section"
            );
        }
    }
}
