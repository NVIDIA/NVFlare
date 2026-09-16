# CVM CPU appraisal for the claim contract pinned in DESIGN.md.
# TCB/configuration references are administrator-approved inputs, not values
# learned from an arbitrary guest. Missing references fail closed.
package policy
import rego.v1

default executables := 33
default hardware := 97
default configuration := 36

executables := 3 if {
    input.snp.measurement in data.reference.snp_launch_measurement
}
hardware := 2 if {
    input.snp.reported_tcb_bootloader in data.reference.snp_bootloader
    input.snp.reported_tcb_microcode in data.reference.snp_microcode
    input.snp.reported_tcb_snp in data.reference.snp_snp_svn
    input.snp.reported_tcb_tee in data.reference.snp_tee_svn
}
configuration := 2 if {
    input.snp.policy_debug_allowed == false
    input.snp.policy_migrate_ma == false
    input.snp.platform_smt_enabled == data.reference.snp_smt_enabled
    input.snp.platform_tsme_enabled == data.reference.snp_tsme_enabled
    input.snp.policy_abi_major == data.reference.snp_guest_abi_major
    input.snp.policy_abi_minor == data.reference.snp_guest_abi_minor
    input.snp.policy_single_socket == data.reference.snp_single_socket
    input.snp.policy_smt_allowed == data.reference.snp_smt_allowed
}

executables := 3 if {
    input.tdx.quote.body.mr_td in data.reference.mr_td
    input.tdx.quote.body.rtmr_0 in data.reference.rtmr_0
    input.tdx.quote.body.rtmr_1 in data.reference.rtmr_1
    input.tdx.quote.body.rtmr_2 in data.reference.rtmr_2
    count(input.tdx.uefi_event_logs) > 0
}
hardware := 2 if {
    input.tdx.quote.header.tee_type == "81000000"
    input.tdx.quote.body.mr_seam in data.reference.mr_seam
    input.tdx.quote.body.tcb_svn in data.reference.tcb_svn
    input.tdx.tcb_status == "OK"
    input.tdx.collateral_expiration_status == "0"
    every advisory in input.tdx.advisory_ids {
        advisory in data.reference.allowed_advisory_ids
    }
}
configuration := 2 if {
    input.tdx.td_attributes.debug == false
    input.tdx.quote.body.xfam in data.reference.xfam
}
