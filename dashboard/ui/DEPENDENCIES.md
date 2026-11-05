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

`node scripts/third-party-notices.mjs` reads package metadata and license/NOTICE files from installed production dependencies without importing dependency code. It also retains notices for vendored code inside those packages. The static build writes the result to `out/THIRD_PARTY_NOTICES.txt`; the repository build helper publishes it with the UI assets.

The inventory is conservative: it includes some build-time members of the production dependency tree that are not present in browser JavaScript. Native `@next/swc-*` compiler and `@img/*` image-processing binaries are excluded because they are not exported or shipped with NVFlare's static Dashboard. In particular, the optional libvips binaries in the npm installation have their own LGPL notices; `node_modules` and those binaries must not be copied into the Python package.

Some published packages omit separate license files. Version-specific, reviewed fallbacks and their provenance are documented in `scripts/licenses/README.md`. Notice generation fails if a new required package lacks supplied or reviewed license text. Dependency updates must review any such failure rather than silently dropping attribution.
