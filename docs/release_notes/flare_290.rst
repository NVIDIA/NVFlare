:orphan:

**************************
What's New in FLARE v2.9.0
**************************

Compatibility and Migration Notes
=================================

- CellPipe cell names now keep the runtime token and pipe mode in one
  explicitly marked, ``~``-delimited FQCN leaf segment
  (``site-1.cellpipe~plain~<job-id>~active``, or
  ``<relay>.cellpipe~alias~<site>~<job-id>~active`` behind a relay) so a
  pipe cell's FQCN parent matches the cell it actually connects to and pipe
  names can never be confused with other cell names. As part of this change,
  CellPipe validates tokens at construction: tokens must be non-empty, may
  not contain the reserved ``~`` separator, and may not contain ``.`` when
  the pipe connects to the site's own CP or a relay. Custom
  ``FlareAgentWithCellPipe`` agent ids that violate these rules now fail fast
  with a ``ValueError`` instead of producing unroutable cell names.
- Both ends of a CellPipe pair derive each other's cell names independently,
  so a Client Job process and an external training process must run the same
  NVFlare naming scheme. A training environment pinned to an older NVFlare
  fails with "peer FQCN mismatch" when paired with a 2.9 CJ; align the
  training environment's NVFlare version with the site's. Only the flat
  whole-FQCN alias used by NVFlare 2.8 and earlier (a root-connected pipe
  named ``<site>_<token>_<mode>``) is still recognized for backward
  compatibility. The forms used through 2.8 when nested under a CP or relay
  (``<parent>.<site>_<token>_<mode>``) are not, because an unmarked leaf
  inside a longer FQCN is indistinguishable from a real cell of that name.
  When upgrading to 2.9, upgrade a site and its relay together, including
  sites currently running NVFlare 2.8.
