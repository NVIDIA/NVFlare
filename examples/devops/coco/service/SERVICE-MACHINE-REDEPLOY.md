# Secure-services guide entry

Use [SERVICE-INSTALLATION.md](SERVICE-INSTALLATION.md) for a fresh machine,
starting with private configuration from `platform.env.example`.
Stages 01–11 install and verify the platform; 02/10/11 also support later
reference-only updates. Stages 12/13 authorize and verify individual workloads.
No live deployment state or host-specific maintenance scripts are included.

Stage 05 is not a policy-preserving restart: rerunning it replaces the active
KBS resource policy with default-deny, removing prior workload release approvals.
Before doing so, follow the
[backup and authorization-recovery procedure](README.md#rerunning-stage-05-resets-workload-authorization).
Stages 09–11 do not restore workload approvals, and resetting the policy cannot
revoke keys already released to running guests.

Platform handoff: SNP five-field or TDX v2 complete-profile JSON through the
provisioning node. See [TDX references](TDX-REFERENCE-VALUES.md) for the latter.
Workload handoff: separate confidential resources and exact release policy.
CoCo runtime handoff: public chart and digest pins, no signed runtime bundle.
