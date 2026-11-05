//! Keeps the HTML of an example's plot when `AUDIO_SAMPLES_EXAMPLE_PLOTS` names
//! a directory; the book's gallery embeds these files. Without the variable the
//! examples only render the plots to memory and check them.

/// Writes `html` to `<dir>/<name>.html` if the variable is set.
pub fn keep(name: &str, html: &str) {
    if let Some(dir) = std::env::var_os("AUDIO_SAMPLES_EXAMPLE_PLOTS") {
        std::fs::create_dir_all(&dir).expect("create the plot directory");
        let path = std::path::Path::new(&dir).join(format!("{name}.html"));
        std::fs::write(path, html).expect("write the plot");
    }
}
