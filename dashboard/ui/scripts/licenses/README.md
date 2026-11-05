# License fallbacks

The notice generator first reads license and NOTICE files from installed runtime packages, including nested vendored notices and license sections in READMEs. These reviewed fallbacks cover published packages that omit separate license files.

- `formik-2.4.9-LICENSE.txt`: upstream Formik Apache-2.0 license, including its Formik, Inc. attribution, retrieved from `https://github.com/jaredpalmer/formik/blob/main/LICENSE` on 2026-11-05 (Git blob `3a60e31d9b5f50b3bed08bc2ced87bb93c18d928`).
- `type-fest-2.19.0-LICENSE.txt`: the MIT option of the package's `(MIT OR CC0-1.0)` license, retrieved from `https://github.com/sindresorhus/type-fest/blob/main/license-mit` on 2026-11-05 (Git blob `fa7ceba3eb4a9657a9db7f3ffca4e4e97a9019de`).
- `tiny-case-1.0.3-LICENSE.txt`: the published package declares MIT and names Jason Quense as its author, but includes no separate copyright notice. The fallback preserves those facts and the standard MIT text without inventing a copyright statement.

The same-version Next.js package supplies the MIT license for its `@next/env` and platform-specific SWC packages. React's installed MIT license supplies the notice for the React `client-only@0.0.1` marker package. These packages identify their parent project in their metadata.

Fallbacks are version-specific. A dependency update that has no supplied license text fails notice generation until its provenance is reviewed. Do not substitute an SPDX identifier alone for the actual license text.
