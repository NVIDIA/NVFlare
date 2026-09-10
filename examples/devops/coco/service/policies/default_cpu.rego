package policy

import rego.v1

default executables := 33
default hardware := 97
default configuration := 36

executables := 3 if {
    input.snp
    input.snp.measurement in query_reference_value("snp_launch_measurement")
}

hardware := 2 if {
    input.snp

    # These are scalar numeric floors provisioned by the independent service
    # administrator.  Keep the type guards: Trustee returns null for a missing
    # reference value, and a missing/malformed floor must fail closed.
    min_bootloader := query_reference_value("snp_min_reported_tcb_bootloader")
    min_tee := query_reference_value("snp_min_reported_tcb_tee")
    min_snp := query_reference_value("snp_min_reported_tcb_snp")
    min_microcode := query_reference_value("snp_min_reported_tcb_microcode")

    is_number(min_bootloader)
    is_number(min_tee)
    is_number(min_snp)
    is_number(min_microcode)

    input.snp.reported_tcb_bootloader >= min_bootloader
    input.snp.reported_tcb_tee >= min_tee
    input.snp.reported_tcb_snp >= min_snp
    input.snp.reported_tcb_microcode >= min_microcode
}

configuration := 3 if {
    input.snp
    input.snp.policy_debug_allowed == false
    input.snp.policy_migrate_ma == false
}

trust_claims := {
    "executables": executables,
    "hardware": hardware,
    "configuration": configuration,
    "file-system": 0,
    "instance-identity": 0,
    "runtime-opaque": 0,
    "storage-opaque": 0,
    "sourced-data": 0,
}

extensions := [
    {"name": "ear.trustee.identifiers", "key": -18,
     "value": {"validated": validated_identifiers}}
]

validated_identifiers := object.union_n([container_images_id, container_uids_id])

container_images := [img |
    container := input["init_data_claims"]["agent_policy_claims"]["containers"][_]
    img := container["OCI"]["Annotations"]["io.kubernetes.cri.image-name"]
]

container_images_id := {"container_images": container_images} if {
    count(container_images) > 0
} else := {}

container_uids := [uid |
    container := input["init_data_claims"]["agent_policy_claims"]["containers"][_]
    uid := container["OCI"]["Process"]["User"]["UID"]
]

container_uids_id := {"container_uids": container_uids} if {
    count(container_uids) > 0
} else := {}
