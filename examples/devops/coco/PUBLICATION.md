# Public package boundary

This example contains reviewed scripts, public documentation, configuration
**templates**, policy source and immutable public software pins. It
contains no live platform configuration, credentials, certificates, approved
measurements, generated Pod handoffs, private reports, VM images or recovery
archives. Historical operational records are not deployment inputs.

`example.com` endpoints and example accounts are illustrative. Configure real
endpoints and explicit target hostnames as described in [CONFIGURATION.md](CONFIGURATION.md).
Blank reference fields must come from trusted review, never from the adversarial
cluster. Template validation does not establish the trustworthiness of an input.

## Publish only a clean package

`role_kits.py` selects only Git-tracked files beneath this example from a clean
NVFlare checkout. It excludes untracked and ignored files and rejects tracked
changes, missing files, symlinks, submodules and merge conflicts. Commit reviewed
changes before assembly. There is no separate file manifest or checksum manifest.
Authenticate the source commit independently; Git tracking alone is not a safety
review or proof that a file contains no secrets.

`validate-package.py` checks tracked source files, shared dependencies, known
lab-specific identifiers, script syntax and local document links. It does not
run regression tests or evaluate policy behavior. With `--assembled`, it checks
the files present in a newly assembled package without needing Git. It cannot
prove completeness or authenticity of an arbitrary received directory.

The `.gitignore` excludes commonly generated configuration, credentials,
certificates, release/evidence directories and archives. It is defense in depth,
not a secret scanner or authorization to publish ignored material. Review the
tracked changes before committing, and the output before sharing. Never add
live configuration or generated deployment artifacts to Git merely to include
them in a kit. Keep the source checkout separate from deployment working directories.

To export the committed source package from the NVFlare repository root, use
Git rather than archiving a working directory:

```bash
git archive --format=tar.gz --output=/path/to/coco-public.tar.gz \
  HEAD examples/devops/coco
```

Choose a new output file outside the source package. This command exports only
committed content, not local edits, ignored files or untracked files.

That archive is the complete **source** package, including `shared/`; it is not
an assembled kit. To distribute independently runnable role directories, run
from `examples/devops/coco` in a clean Git checkout (not an unpacked archive):

```bash
python3 role_kits.py /path/to/new-coco-role-kits
python3 /path/to/new-coco-role-kits/validate-package.py --assembled
```

The output includes the materialized templates and implementation copies, and
assembled validation compares them with the shared sources. Archive
only the intended role directory from this output for its operator. Never copy
source wrappers without their shared tree or edit generated implementation copies.
Regeneration refuses to overwrite existing output. Runtime configuration remains
private and is supplied separately after transfer.

The reviewed public documentation in `docs/` is limited to these three files:

- [Design slides (Markdown)](docs/coco-security-design-3-slides.md).
- [Four-party sequence diagram (Mermaid source)](docs/coco-four-party-sequence.mmd).
- [Four-party sequence diagram (offline interactive HTML)](docs/coco-four-party-sequence.html).

The HTML is the approved rendered sequence diagram, including script-name
tooltips; it is not a slide export. PDF/PPTX exports, slide exporters and other
files in `docs/` remain excluded. The validator rejects other `docs/` files and
duplicated role `CURRENT-STATE.md` summaries.

## Offline static checks

Use Python 3.11+ and Bash; source validation also requires Git. From this
example's directory in the checkout, run:

```bash
python3 validate-package.py
```

For a full assembled package, add `--assembled`. These checks require no
PyYAML, OPA or network access and do not deploy services or perform attestation.
They do not validate authorization behavior; perform the documented live
verification and negative checks before deployment.

Generate secrets on their owning machines and transfer them only to the
necessary party. Admin receives its publisher credential, not KBS administration
or service TLS private keys. Secure services receives the workload decryption
key, not its signing private key. CoCo receives only public runtime/trust
inputs and the final Pod YAML. Public certificates become deployment-specific
inputs when generated; they are not committed here.

The lab-only teardown scripts and obsolete signed-bundle unpacker are excluded.
See [maintenance boundaries](trusted_system/TEARDOWN.md).
