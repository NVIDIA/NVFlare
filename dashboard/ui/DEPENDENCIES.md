# Frontend dependency baseline

The initial supported baseline was selected on 2026-11-04. Exact dependency and transitive versions are recorded in `package-lock.json`.

| Component | Baseline |
| --- | --- |
| Node LTS used for validation | 24.22.2 |
| npm | 11.19.1 |
| Next.js | 16.4.5 |
| React / React DOM | 19.3.1 |
| MUI Material / System / icons | 9.5.0 |
| MUI X Data Grid / Date Pickers | 9.17.0 |
| TypeScript | 6.0.3 |
| ESLint | 10.14.0 |
| Jest / jsdom environment | 30.5.2 |
| Playwright | 1.64.1 |
| Axios | 1.21.0 |

MUI Core and MUI X have independent minor versions; both use their compatible v9 major. The community packages used here are MIT-licensed. TypeScript stays on 6.0 because the current `typescript-eslint` peer range excludes 6.1 and 7. Jest uses Next's SWC transform through `next/jest`, with no `ts-jest` or custom Babel transform.

The aggregate `eslint-config-next` package pulls an import plugin whose peer range rejects ESLint 10. The flat configuration instead composes compatible ESLint, TypeScript, Next, and React Hooks plugins directly. Installation uses strict peer checks without `--force` or `--legacy-peer-deps`.

## Update and audit

Keep direct dependencies and their peers as a tested set. After changing versions, refresh the lockfile, run type checking, lint, Jest, the static build, browser tests, and backend integration tests. Re-run both `npm audit` and `npm audit --omit=dev`; audit results change over time and do not prove that code is vulnerability-free.

Public integration validation on 2026-11-05 found zero production advisories. The full audit reported 28 affected development packages (25 moderate, 3 low), including propagated parent-package findings. Development tools still reported findings from these advisory sources, including propagated findings for parent packages:

- `glob`: [GHSA-3g3m-2fjr-6pw4](https://github.com/advisories/GHSA-3g3m-2fjr-6pw4), cancellation handling during filesystem traversal. Tool parents require older majors than the patched 14.0.2.
- `minimatch`: [GHSA-g3hv-75wg-m249](https://github.com/advisories/GHSA-g3hv-75wg-m249), glob-pattern amplification. Tool parents require older majors than the patched 11.0.1.
- `deepmerge`: [GHSA-gcqx-w48m-w2h4](https://github.com/advisories/GHSA-gcqx-w48m-w2h4), prototype poisoning. The affected development-tool release has no published patch.

There were no high or critical findings after the transitive patch refresh. These remaining dependencies process trusted repository/configuration inputs during validation. Incompatible major overrides or unsupported tool downgrades have not been used merely to suppress audit output.

## Distribution and notices

The production build generates browser source maps for license inventory. The notice generator reads the source maps for every exported JavaScript/CSS asset and selects the actual installed packages and used vendored components. Missing maps or unknown source formats fail the build; generated Next.js page-loader and manifest data are handled explicitly. Native/server/build-only packages are not selected merely because npm installed them.

The generator preserves complete license and NOTICE content, sharing identical text and the standard MIT and Apache-2.0 bodies between explicitly mapped packages. Only whitespace may change during text matching; distinct copyrights, exceptions, appendices, and additional conditions remain intact. Reconstruction checks verify that each original notice can be recovered without losing words or punctuation. Version-specific fallbacks are documented in `scripts/licenses/README.md`.

The final export contains `THIRD_PARTY_NOTICES.txt`, with no browser source maps or source-map URL comments. Maps stay under the ignored `.next/` build directory. The publishing helper and wheel smoke check reject accidentally exported maps. Node dependencies and native image-processing binaries are not included in the Python distribution.

## License scope of the distributed Dashboard

The 2026-11-05 production browser source maps identify 51 installed package roots
(50 distinct package names, because two `react-is` versions are used) and two
Next.js vendored components. Their declared licenses and supplied license texts
are MIT, Apache-2.0, or BSD-3-Clause. The exported inventory contains no identified
GPL, AGPL, LGPL, MPL, noncommercial, or proprietary component. MUI Material,
System, Data Grid, and Date Pickers are the MIT-licensed community packages; no
MUI Pro/Premium package or commercial license key is used.

| Distributed components | License |
| --- | --- |
| MUI Community, React, Next.js, Emotion, Axios, Day.js, Yup, and the other MIT-listed browser dependencies | MIT |
| Formik and `@swc/helpers` | Apache-2.0 |
| `hoist-non-react-statics` and `react-transition-group` | BSD-3-Clause |
| Next.js vendored `path-to-regexp` and `process` | MIT |

The larger npm installation contains packages with additional obligations that
are absent from this exported browser inventory:

| Build or server dependency | License and scope |
| --- | --- |
| Optional native libvips/Sharp platform bundles | LGPL-3.0-or-later, sometimes combined with Apache-2.0/MIT; native image-processing binaries are not included in the static UI or Python distributions. |
| Next.js vendored `@vercel/og` 1.0.1 and its Satori license files | MPL-2.0; the unused Open Graph renderer is not exported. Distributing it later would require reviewing the MPL source-availability obligations. |
| `caniuse-lite` browser-support data | CC-BY-4.0; used by build tooling and absent from the exported browser inventory. |
| BlueOak-1.0.0 tools | Permissive development/build tools; absent from the exported browser inventory. |

`type-fest` is dual-licensed MIT OR CC0-1.0; the MIT option is available. It is a
typing/build dependency rather than part of this browser inventory.

The generator accepts only reviewed permissive license expressions for exported
components and fails on unreviewed expressions, including vendored package
metadata. This check is specific to the locked dependencies and the actual build;
a dependency or feature update must repeat the inventory and license review.
Required copyright, license, and NOTICE text remains in the generated notices,
with identical text shared between components. These findings describe the
observed distribution and do not constitute blanket approval of future packages
or of redistributing the complete npm installation.
