# Design slides: one source, generated exports

[coco-security-design-3-slides.md](coco-security-design-3-slides.md) is the only
authoritative slide source, including its theme settings and three slide breaks.
Edit it, not HTML, PDF or PowerPoint exports. Generated exports are intentionally
absent from Git and the public deployment-package inventory.

## Generate all formats

Use Node.js 22 or 24 LTS, Marp CLI **4.5.1**, and a supported installed Chrome/Chromium
browser. The source contains no remote images, fonts or scripts. PDF and PPTX
conversion require a browser; the standard PPTX export contains rendered slide
images, not editable text. See the [Marp CLI documentation](https://github.com/marp-team/marp-cli/tree/v4.5.1).

Install the exporter outside this source package (this step requires network):

```bash
SLIDE_TOOLS="$HOME/.local/share/coco-slide-tools"
mkdir -p "$SLIDE_TOOLS"
npm install --prefix "$SLIDE_TOOLS" --save-exact @marp-team/marp-cli@4.5.1
```

From the package root, choose a **new** output directory outside the package:

```bash
MARP_BIN="$SLIDE_TOOLS/node_modules/.bin/marp" \
  BROWSER_BIN=/path/to/chrome-or-chromium \
  bash docs/export-slides.sh /path/to/new-coco-slide-exports
```

This produces HTML, PDF and PPTX from the same Markdown. The script refuses
an existing output directory or another Marp CLI version and records the
source SHA-256 and exporter/Node/browser versions. Set `BROWSER_BIN` to the
actual Chrome/Chromium executable; omit it to use Marp's browser discovery.
It does not modify source or upload files. Render only reviewed Markdown with
this toolchain; keep it separate from credentials and deployment hosts.

Retain the tool installation's `package.json` and `package-lock.json` with the
export artifacts. On subsequent builds, restore those files to a fresh tools
directory and run `npm ci --prefix "$SLIDE_TOOLS"`. Record the source Git commit,
Node/browser versions and fonts with the release; use the same environment to
reproduce rendering. File metadata/timestamps can differ: byte-identical PDF or
PPTX output across toolchains is not promised.

Review all three rendered slides, then publish the exports as documentation
downloads or release attachments through the project's normal publication
process. Do not add exports to `PACKAGE-FILES.txt` or commit them to the source
tree. Deployment hosts do not need Node, Marp or a browser.
