# Read the Docs redirects

[readthedocs_redirects.json](readthedocs_redirects.json) records exact redirects for
the `nvflare` project on Read the Docs. These rules recover missing introduction,
quickstart, and XGBoost pages, plus the old 2.4 documentation homepage.
They move readers from retired-release URLs to the corresponding current
`main` documentation.

**Merging this file does not activate redirects.** Read the Docs stores these
settings in its project dashboard or API. A build of `main`, Sphinx redirect
pages, or a `redirects` key in `.readthedocs.yaml` cannot configure redirects for
missing older versions.

## Apply the reviewed rules

1. Sign in as a maintainer of the
   [NVFlare Read the Docs project](https://app.readthedocs.org/projects/nvflare/)
   and open **Admin > Redirects**.
2. Record the existing settings before changing them. Check whether a rule
   already handles each `from_url`, including broader matching rules. Reuse an
   equivalent rule instead of creating a duplicate; review any conflicting rule
   before replacing it.
3. Confirm that each source is still missing and its destination returns HTTP
   200 with the intended content. If historical documentation must remain
   available at a source URL, restore that version instead of redirecting it.
4. For each JSON object, select **Exact redirect** and copy the full `from_url`
   and `to_url`, including `/en/<version>/`. Use HTTP **301**, enable the rule,
   and leave **Apply even if the page exists** disabled (`force: false`).
   Do not replace these mappings with a wildcard or a redirect of every old
   page to the homepage.
5. Save the rules and verify the public URLs below. Read the Docs activates
   saved rules immediately; no documentation rebuild is required.

Each object also uses the fields accepted by the Read the Docs
[redirect-create API](https://docs.readthedocs.com/platform/stable/api/v3.html#redirect-create).
The API accepts **one object per request**, not the complete JSON array.
Specify `http_status: 301` explicitly: the API default is a temporary 302.
Keep authentication tokens out of the repository.

## Verify deployment

Check the initial response without following redirects, for example:

```bash
curl --silent --show-error --max-time 30 --output /dev/null --dump-header - \
  https://nvflare.readthedocs.io/en/2.4/fl_introduction.html
```

Expect HTTP **301** and a `Location` resolving to
`https://nvflare.readthedocs.io/en/main/fl_introduction.html`. Then check the
complete path:

```bash
curl --fail --silent --show-error --location --max-redirs 5 --max-time 30 \
  --output /dev/null --write-out '%{http_code} %{url_effective}\n' \
  https://nvflare.readthedocs.io/en/2.4/fl_introduction.html
```

Expect HTTP **200** at the intended destination. Repeat both checks for every
`from_url` in the JSON file, and confirm that the final page contains the
corresponding guide rather than an error page. Read the Docs does not apply
project redirects to pull-request preview domains; verify the production host.

Record the activation date and results in the tracking issue. Use Search
Console URL Inspection to check the old and replacement URLs after deployment.
Keep the issue open until these live checks pass. To roll back a rule, disable
the newly added rule or restore its recorded previous settings; permanent
redirects may remain cached by clients.

## Extending the map

Add an exact mapping only after confirming the missing URL and a suitable
replacement. Preserve working version-specific documentation. Review and apply
changes in the dashboard separately from the repository PR.

See Read the Docs documentation on
[exact redirects](https://docs.readthedocs.com/platform/stable/user-defined-redirects.html#exact-redirects)
and [configuring redirects](https://docs.readthedocs.com/platform/stable/guides/redirects.html).
