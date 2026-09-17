# CVM CPU appraisal for the claim contract pinned in DESIGN.md.
# TCB/configuration references are administrator-approved inputs, not values
# learned from an arbitrary guest. Missing references fail closed.
package policy
import rego.v1

# v0.22 RVPS does not filter expired records; enforce operator deadlines here.
reference(name) := value if {
    expiry := query_reference_value("cvm_reference_expiry")[name]
    is_number(expiry)
    time.now_ns() < expiry * 1000000000
    value := query_reference_value(name)
}

default executables := 33
default hardware := 97
default configuration := 36

executables := 3 if {
    input.snp.measurement in reference("snp_launch_measurement")
}
hardware := 2 if {
    input.snp.reported_tcb_bootloader in reference("snp_bootloader")
    input.snp.reported_tcb_microcode in reference("snp_microcode")
    input.snp.reported_tcb_snp in reference("snp_snp_svn")
    input.snp.reported_tcb_tee in reference("snp_tee_svn")
}
configuration := 2 if {
    input.snp.policy_debug_allowed == false
    input.snp.policy_migrate_ma == false
    input.snp.platform_smt_enabled == reference("snp_smt_enabled")
    input.snp.platform_tsme_enabled == reference("snp_tsme_enabled")
    input.snp.policy_abi_major == reference("snp_guest_abi_major")
    input.snp.policy_abi_minor == reference("snp_guest_abi_minor")
    input.snp.policy_single_socket == reference("snp_single_socket")
    input.snp.policy_smt_allowed == reference("snp_smt_allowed")
}

executables := 3 if {
    input.tdx.quote.body.mr_td in reference("mr_td")
    input.tdx.quote.body.rtmr_0 in reference("rtmr_0")
    input.tdx.quote.body.rtmr_1 in reference("rtmr_1")
    input.tdx.quote.body.rtmr_2 in reference("rtmr_2")
    count(input.tdx.uefi_event_logs) > 0
}
hardware := 2 if {
    input.tdx.quote.header.tee_type == "81000000"
    input.tdx.quote.body.mr_seam in reference("mr_seam")
    input.tdx.quote.body.tcb_svn in reference("tcb_svn")
    input.tdx.tcb_status == "OK"
    input.tdx.collateral_expiration_status == "0"
    every advisory in input.tdx.advisory_ids {
        advisory in reference("allowed_advisory_ids")
    }
}
configuration := 2 if {
    input.tdx.td_attributes.debug == false
    input.tdx.quote.body.xfam in reference("xfam")
}

trust_claims := {"executables": executables, "hardware": hardware, "configuration": configuration}
