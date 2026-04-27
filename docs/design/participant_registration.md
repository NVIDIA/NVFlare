# Participant Registration Design

## Introduction

This document proposes a long-term server-side participant-registration capability for NVFlare. Its purpose is to give the running server a trusted inventory of provisioned participants, their roles, and their organizations, independent of current connection state.

This is intentionally separate from `docs/design/multistudy_cli.md`. The current study CLI can ship with a simpler Phase 1 model, while participant registration provides the stronger validation path for future iterations.

---

## Goals

- provide the server with a durable participant registry for sites and admin users
- support both centralized provisioning and distributed provisioning workflows
- allow incremental registry updates after initial deployment
- enable offline validation of site existence, user existence, and org membership
- separate identity/configuration truth from runtime connectivity truth

## Non-Goals

- replacing TLS or mTLS as the trust anchor for connected peers
- changing certificate issuance/signing workflows directly
- designing revocation in full detail
- redesigning study CLI itself beyond the validation hooks that would consume the registry

---

## Problem Statement

Today the running server has only partial identity visibility:

- admin connections expose caller identity, role, and org from the presented certificate
- connected FL clients are visible through `ClientManager`, but only while online
- there is no durable, server-owned global inventory of all provisioned participants across centralized and distributed provisioning

That creates product gaps:

- study registration cannot validate offline sites cleanly
- org-scoped study/user mutation cannot reliably validate target-user org
- dynamic site growth requires an operational way to teach the running server about new participants

---

## Design Overview

Introduce a server-side participant registry that records the provisioned identity inventory and can be updated over time.

The registry becomes the source of truth for:

- whether a site or admin user is known to the deployment
- the declared org for a participant
- the declared participant type and allowed role metadata

Connected runtime objects remain useful, but for a different question:

- registry answers: "is this a real provisioned participant?"
- live connection state answers: "is this participant currently connected?"

---

## Data Sources

### Centralized Provisioning

Centralized provisioning can generate a participant inventory file under the server startup area, for example `startup/local/participants.json`, and the server can load it at startup.

This file should include at least:

- participant name / CN
- participant type
- org
- role metadata where applicable

### Distributed Provisioning

Distributed provisioning does not naturally give the running server a complete global inventory. However, the project admin signs participant certificates and therefore has the operational vantage point to maintain that inventory.

To support this model, add a new project-admin CLI workflow that can send participant records to the server:

- initial registry bootstrap
- incremental addition of newly signed participants
- updates when project-admin-issued participant metadata changes

This is not the same as study CLI. It is a separate management surface for server identity inventory.

---

## Dynamic Updates

Participant registration must support changes after server startup.

### New Sites or Users

For both provisioning models, newly added participants require a registry update:

- centralized provisioning: dynamic provision by project admin must also update the server registry
- distributed provisioning: once the project admin signs a new participant certificate, the project admin must send the additional participant metadata to the server

That implies:

- the server registry must support incremental updates
- updates must be authenticated and authorized
- the update path should be automatable through CLI

---

## Proposed Capabilities

### Server Registry Lifecycle

- load participant registry at server startup when present
- keep an in-memory authoritative registry object
- persist registry updates safely to disk
- support hot reload after accepted updates

### Project Admin CLI

Add a new CLI family, separate from `nvflare study`, for participant registration management. The exact command set is future work, but likely needs:

- registry bootstrap or import
- add participant
- update participant
- remove or deactivate participant
- list and show participant records

Only `project_admin` should be allowed to mutate the registry.

### Validation Consumers

Once available, the participant registry can be used by:

- study CLI for offline site/user/org validation
- dynamic provisioning flows
- runtime checks comparing provisioned metadata with presented cert metadata

---

## Relationship To Live Client Metadata

Adding org to connected client properties is still useful even with a participant registry.

Recommended model:

- participant registry: provisioned identity truth
- connected client props: presented cert metadata for the active connection

That enables checks such as:

- provisioned site org vs connected cert org mismatch
- offline known participant vs currently unavailable participant

So the registry does not replace live metadata. It complements it.

### Runtime Enhancement: Persist Connected Client Org

Even before a full participant registry exists, the server can preserve the connected client's presented org from its certificate.

Current code shape:

- admin login already parses certificate subject fields by calling `cert_to_dict(...)` and `get_identity_info(...)`
- `get_identity_info(...)` already extracts `organizationName` into `IdentityKey.ORG`
- FL client authentication in `ClientManager.authenticated_client()` currently verifies CN/signature but does not persist org on the resulting `Client` object

Suggested enhancement:

1. Add `ORG = "org"` to `ClientPropKey` in `nvflare/apis/client.py`
2. In `nvflare/private/fed/server/client_manager.py`, after successful certificate verification:
   - convert the client cert bytes with `cert_to_dict(...)`
   - call `get_identity_info(...)`
   - read `identity.get(IdentityKey.ORG)`
3. Store that value on the connected client object with `client.set_prop(ClientPropKey.ORG, org)`

Likely code touchpoints:

- `nvflare/apis/client.py`
  - add `ClientPropKey.ORG`
- `nvflare/private/fed/server/client_manager.py`
  - import `cert_to_dict`
  - import `get_identity_info` and `IdentityKey`
  - parse the already-verified client cert and set `Client.props[ClientPropKey.ORG]`
- optional follow-on:
  - expose org in any client debug/status views where it is useful operationally

Purpose:

- make connected client org available as trusted runtime metadata
- support future checks comparing provisioned org vs connected-cert org
- support submit-time validation and diagnostics without reparsing cert state elsewhere

Limitation:

- this only describes currently connected clients
- it does not replace a provisioned participant registry for offline validation

---

## Open Design Areas

This deserves its own implementation/design effort because it introduces several independent questions:

- exact on-disk schema for participant records
- persistence location and upgrade strategy
- bootstrap vs merge vs replace semantics
- how distributed-provisioning project admins package and submit updates
- conflict handling and duplicate detection
- removal, deactivation, and revocation semantics
- compatibility with existing centralized provisioning artifacts
- auditability of registry mutations

---

## Recommended Staging

### Phase 1

- keep `nvflare study` on the simpler declarative model
- optionally enrich connected `Client` objects with org metadata from the presented cert
- enforce runtime connectivity at job submission, not study registration

### Phase 2

- implement server-side participant registry
- add project-admin participant-registration CLI
- update study CLI to use the registry for stronger offline validation

This keeps current study CLI work moving while preserving a clear long-term path.
