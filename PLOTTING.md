# Static Plots Setup

The `static-plots` feature enables PNG and SVG export of plots. It
requires a browser and WebDriver to be installed and configured via
environment variables **before building**.

## Quick Setup

Choose your platform and browser, then run one of the commands below
before building with `--features static-plots`:

### Linux

**Chrome/Chromium (recommended):**
```bash
export BROWSER_PATH=$(command -v chromium || command -v google-chrome || command -v chromium-browser || echo "/usr/bin/chromium")
export WEBDRIVER_PATH=$(command -v chromedriver || echo "auto")
```

**Firefox:**
```bash
export BROWSER_PATH=$(command -v firefox || echo "/usr/bin/firefox")
export WEBDRIVER_PATH=$(command -v geckodriver || echo "/usr/local/bin/geckodriver")
```

### macOS

**Chrome:**
```bash
export BROWSER_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
export WEBDRIVER_PATH=$(command -v chromedriver || echo "/usr/local/bin/chromedriver")
```

**Firefox:**
```bash
export BROWSER_PATH="/Applications/Firefox.app/Contents/MacOS/firefox"
export WEBDRIVER_PATH=$(command -v geckodriver || echo "/usr/local/bin/geckodriver")
```

### Windows (PowerShell)

**Chrome:**
```powershell
$env:BROWSER_PATH="C:\Program Files\Google\Chrome\Application\chrome.exe"
$env:WEBDRIVER_PATH="C:\Program Files\chromedriver.exe"
```

**Firefox:**
```powershell
$env:BROWSER_PATH="C:\Program Files\Mozilla Firefox\firefox.exe"
$env:WEBDRIVER_PATH="C:\Program Files\geckodriver.exe"
```

## Building with Static Plots

After setting the environment variables:

```bash
cargo build --features "plotting static-plots transforms"
# or run an example
cargo run --example plotting_basic --features "plotting static-plots transforms"
```

## Permanent Setup

Add to your shell configuration to avoid setting these every session:

**Linux/macOS** (`~/.bashrc` or `~/.zshrc`):
```bash
# Chrome/Chromium
export BROWSER_PATH=/usr/bin/chromium
export WEBDRIVER_PATH=/usr/local/bin/chromedriver
```

**Windows** (PowerShell profile):
```powershell
[Environment]::SetEnvironmentVariable("BROWSER_PATH", "C:\Program Files\Google\Chrome\Application\chrome.exe", "User")
[Environment]::SetEnvironmentVariable("WEBDRIVER_PATH", "C:\Program Files\chromedriver.exe", "User")
```

## Installing a Browser and WebDriver

**Ubuntu/Debian:**
```bash
# Chrome/Chromium
sudo apt install chromium-browser chromium-chromedriver
# Firefox
sudo apt install firefox && cargo install geckodriver
```

**macOS:**
```bash
# Chrome
brew install --cask google-chrome && brew install chromedriver
# Firefox
brew install --cask firefox && brew install geckodriver
```

**Arch/Manjaro:**
```bash
# Chrome/Chromium
sudo pacman -S chromium chromedriver
# Firefox
sudo pacman -S firefox geckodriver
```

**Windows:**
- Download [Chrome](https://www.google.com/chrome/) or [Firefox](https://www.mozilla.org/firefox/)
- Download [chromedriver](https://chromedriver.chromium.org/downloads) or [geckodriver](https://github.com/mozilla/geckodriver/releases)
- Extract the driver to a location in your PATH or set `WEBDRIVER_PATH` to the full path

## Troubleshooting

**Build fails with `Failed to detect browser path`:**
1. Confirm the browser is installed and executable
2. Set the exact path: `export BROWSER_PATH=/path/to/your/browser`
3. Set the driver path: `export WEBDRIVER_PATH=/path/to/driver`
4. Verify: `echo $BROWSER_PATH && echo $WEBDRIVER_PATH`

**Driver version mismatch:**
- Ensure your WebDriver version matches your browser version
- Update both to the latest available releases
