# NVIDIA FLARE GitHub Pages Website

## Installation

### 1. Install Dependencies

```bash
npm install
```

### 2. Start development Server

```bash
npm run dev
```

### 3. Preview & Build

```bash
npm run preview
npm run build
```

## Project Structure

```
/
├── public/
│   └── ...
├── src/
│   ├── components/
│   │   └── ...
│   ├── layouts/
│   │   └── ...
│   └── pages/
│       └── ...
└── package.json
```

## Maintaining the research page

`src/data/research.ts` holds the project catalog and selected paper highlights;
`src/pages/research.astro` renders both from that data. Compare the catalog with
`research/README.md` and each project README when research folders change. The
initial refresh covers upstream `main` at `7636087a0` (2026-09-23), including the
Auto-FL paper linked from its project README and the shared-data/template folders.

Keep paper results separate from each example's implementation scope. Examples
always link to GitHub `main` because newer research folders may not exist on a
release branch. Documentation links follow the site's configured branch.

For figure sources, versions, and attribution, see
[`src/images/research/highlights/README.md`](src/images/research/highlights/README.md).
Keep figures local, preserve their aspect ratios, and supply descriptive alt text,
a caption, a paper source link, and access to the original image.

After edits, build the site and preview `/NVFlare/research/` on desktop and mobile.
Check text search, area filters, the empty state, full-size figure links, and the
complete catalog with JavaScript disabled. Also check a versioned build with
`VITE_PUBLIC_GH_BRANCH` set so local asset paths keep the deployment base prefix.
