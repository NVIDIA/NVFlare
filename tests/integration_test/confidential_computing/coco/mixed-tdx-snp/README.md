# Mixed-runtime provisioning fixtures

This directory supplies the offline mixed SNP/TDX provisioning regression in
`tests/unit_test/lighter/cc_provision/impl/mixed_coco_workflow_test.py`.

The tests copy the fixture and replace endpoint, credential, application-image,
platform-configuration, and build-command paths before provisioning. These are
test inputs, not an additional deployment workflow.

For deployment, use [the CoCo example](../../../../../examples/devops/coco/README.md)
and select each participant's supported CPU/GPU mechanism in its CC configuration.
