package policy

import rego.v1

default executables := 33
default hardware := 97
default configuration := 36

executables := 3 if {
    input.snp
    not input.tdx
    input.snp.measurement in query_reference_value("snp_launch_measurement")
}

hardware := 2 if {
    input.snp
    not input.tdx

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

    is_number(input.snp.reported_tcb_bootloader)
    is_number(input.snp.reported_tcb_tee)
    is_number(input.snp.reported_tcb_snp)
    is_number(input.snp.reported_tcb_microcode)

    input.snp.reported_tcb_bootloader >= min_bootloader
    input.snp.reported_tcb_tee >= min_tee
    input.snp.reported_tcb_snp >= min_snp
    input.snp.reported_tcb_microcode >= min_microcode
}

configuration := 3 if {
    input.snp
    not input.tdx
    input.snp.policy_debug_allowed == false
    input.snp.policy_migrate_ma == false
}

# TDX claim paths and encodings follow Trustee 338610fbfed57b66c61a8a3a60e0e4386bdce793:
# deps/verifier/src/tdx/{mod.rs,claims.rs} and ear_default_policy_cpu.rego.
# The verifier validates the DCAP quote and replays CCEL against all four RTMRs.
# This policy additionally requires a complete, administrator-approved TDVF
# profile. There is deliberately no Grub/no-event-log fallback.
executables := 3 if {
    tdx_approved_profile
}

hardware := 2 if {
    tdx_approved_profile
    input.tdx.quote.header.tee_type == "81000000"
    input.tdx.quote.header.vendor_id == "939a7233f79c4ca9940a0db3957f0607"
    input.tdx.tcb_status == "UpToDate"
    # Newer QVL supplemental data additionally appraises the current platform
    # TCB. Older QVL omits it; if reported, it must also be acceptable.
    object.get(input.tdx, "tcb_status_current", "UpToDate") == "UpToDate"
    input.tdx.collateral_expiration_status == "0"
}

configuration := 2 if {
    tdx_approved_profile
    input.tdx.td_attributes.debug == false
}

tdx_approved_profile if {
    input.tdx
    not input.snp
    # A single fixed RVPS key is replaced as one value. Profile IDs are labels,
    # never reference paths supplied by evidence. All eight fields must match the
    # SAME object, rather than independent allowlists for each measurement.
    profiles := query_reference_value("coco_tdx_profiles_v2")
    is_array(profiles)
    count(profiles) > 0
    count(profiles) <= 64
    some profile in profiles
    is_object(profile)
    object.keys(profile) == {"id", "mr_td", "rtmr_0", "rtmr_1", "rtmr_2", "rtmr_3", "xfam", "tdvfkernel", "tdvfkernelparams"}
    regex.match("^[a-z0-9][a-z0-9_.-]{0,63}$", profile.id)
    every field in ["mr_td", "rtmr_0", "rtmr_1", "rtmr_2", "rtmr_3", "tdvfkernel", "tdvfkernelparams"] {
        regex.match("^[0-9a-f]{96}$", profile[field])
    }
    regex.match("^[0-9a-f]{16}$", profile.xfam)
    input.tdx.quote.body.mr_td == profile.mr_td
    input.tdx.quote.body.rtmr_0 == profile.rtmr_0
    input.tdx.quote.body.rtmr_1 == profile.rtmr_1
    input.tdx.quote.body.rtmr_2 == profile.rtmr_2
    input.tdx.quote.body.rtmr_3 == profile.rtmr_3
    input.tdx.quote.body.xfam == profile.xfam
    tdx_kernel_digest == profile.tdvfkernel
    tdx_params_digest == profile.tdvfkernelparams
}

tdx_kernel_digest := digest if {
    events := [event |
        event := input.tdx.uefi_event_logs[_]
        event.type_name == "EV_EFI_BOOT_SERVICES_APPLICATION"
        "File(kernel)" in event.details.device_paths
    ]
    count(events) == 1
    digests := [item.digest |
        item := events[0].digests[_]
        item.alg == "SHA-384"
    ]
    count(digests) == 1
    digest := digests[0]
}

tdx_params_digest := digest if {
    events := [event |
        event := input.tdx.uefi_event_logs[_]
        event.type_name == "EV_EVENT_TAG"
        event.details.string == "LOADED_IMAGE::LoadOptions"
    ]
    count(events) == 1
    digests := [item.digest |
        item := events[0].digests[_]
        item.alg == "SHA-384"
    ]
    count(digests) == 1
    digest := digests[0]
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
