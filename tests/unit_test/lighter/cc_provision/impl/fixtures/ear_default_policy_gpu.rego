package policy

import rego.v1

default hardware := 97

default executables := 33

default configuration := 36

default file_system := 0

default instance_identity := 0

default runtime_opaque := 0

default storage_opaque := 0

default sourced_data := 0

hardware := 2 if {
	input.sampledevice.svn in data.reference.device_svn
}

trust_claims := {
	"executables": executables,
	"hardware": hardware,
	"configuration": configuration,
	"file-system": file_system,
	"instance-identity": instance_identity,
	"runtime-opaque": runtime_opaque,
	"storage-opaque": storage_opaque,
	"sourced-data": sourced_data,
}

# GPUs verified by NRAS
hardware := 2 if {
	input.nvidia

	input.nvidia["x-nvidia-gpu-attestation-report-cert-chain"]["x-nvidia-cert-ocsp-status"] == "good"
	input.nvidia["x-nvidia-gpu-attestation-report-cert-chain"]["x-nvidia-cert-status"] == "valid"

	input.nvidia["x-nvidia-gpu-attestation-report-cert-chain-fwid-match"]
	input.nvidia["x-nvidia-gpu-attestation-report-parsed"]
	input.nvidia["x-nvidia-gpu-attestation-report-signature-verified"]

	input.nvidia["x-nvidia-gpu-arch-check"]
}

configuration := 2 if {
	input.nvidia.secboot
	input.nvidia.dbgstat == "disabled"
	input.nvidia["x-nvidia-gpu-vbios-version"] in query_reference_value("allowed_vbios_versions")
	input.nvidia["x-nvidia-gpu-driver-version"] in query_reference_value("allowed_driver_versions")
}

else := 3 if {
	input.nvidia.secboot
	input.nvidia.dbgstat == "disabled"
}

executables := 3 if {
	input.nvidia["x-nvidia-gpu-vbios-rim-cert-chain"]["x-nvidia-cert-ocsp-status"] == "good"
	input.nvidia["x-nvidia-gpu-vbios-rim-cert-chain"]["x-nvidia-cert-status"] == "valid"

	input.nvidia["x-nvidia-gpu-driver-rim-fetched"]
	input.nvidia["x-nvidia-gpu-driver-rim-measurements-available"]
	input.nvidia["x-nvidia-gpu-driver-rim-schema-validated"]
	input.nvidia["x-nvidia-gpu-driver-rim-signature-verified"]
	input.nvidia["x-nvidia-gpu-driver-rim-version-match"]

	input.nvidia["x-nvidia-gpu-vbios-rim-fetched"]
	input.nvidia["x-nvidia-gpu-vbios-rim-measurements-available"]
	input.nvidia["x-nvidia-gpu-vbios-rim-schema-validated"]
	input.nvidia["x-nvidia-gpu-vbios-rim-signature-verified"]
	input.nvidia["x-nvidia-gpu-vbios-rim-version-match"]

	input.nvidia.measres == "success"
}
