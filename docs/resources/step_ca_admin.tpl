{{- /*
Example for a dedicated provisioner with one organization and one FLARE role.
Customize the fixed values below; preserve existing identity/role mapping when
adding study SANs to another template. projectURIPath is a percent-encoded label,
not an authorization boundary. See the published provisioning guide for setup.
*/ -}}
{{- $projectURIPath := "demo" -}}
{{- $organization := "example" -}}
{{- $flareRole := "lead" -}}
{{- $requiredAdminGroup := "nvflare-demo-example-lead" -}}
{{- $studyURIBase := printf "https://nvidia.com/nvflare/v1/project/%s/study/" $projectURIPath -}}
{{- if not (kindIs "map" .Token) -}}
  {{- fail "a signed OIDC token is required" -}}
{{- end -}}
{{- $token := .Token -}}
{{- if .Insecure.User -}}
  {{- fail "user-supplied template data is not accepted" -}}
{{- end -}}
{{- $email := get $token "email" -}}
{{- if or (not (kindIs "string" $email)) (eq $email "") -}}
  {{- fail "the signed OIDC token must contain a non-empty email claim" -}}
{{- end -}}
{{- $emailVerified := get $token "email_verified" -}}
{{- if or (not (kindIs "bool" $emailVerified)) (not $emailVerified) -}}
  {{- fail "the signed OIDC token must contain email_verified=true" -}}
{{- end -}}
{{- $groups := get $token "groups" -}}
{{- if or (not (kindIs "slice" $groups)) (not (has $requiredAdminGroup $groups)) -}}
  {{- fail (printf "the signed OIDC token is missing required admin group %q" $requiredAdminGroup) -}}
{{- end -}}
{{- $hasStudyClaim := hasKey $token "nvflare_studies" -}}
{{- $sans := list (dict "type" "email" "value" $email) -}}
{{- if $hasStudyClaim -}}
  {{- $requestedStudies := get $token "nvflare_studies" -}}
  {{- if not (kindIs "slice" $requestedStudies) -}}
    {{- fail "nvflare_studies must be an array" -}}
  {{- end -}}
  {{- if gt (len $requestedStudies) 64 -}}
    {{- fail "nvflare_studies exceeds the 64-study limit" -}}
  {{- end -}}

  {{- $seen := dict -}}
  {{- range $study := $requestedStudies -}}
    {{- if not (kindIs "string" $study) -}}
      {{- fail "each nvflare_studies entry must be a string" -}}
    {{- end -}}
    {{- if eq $study "default" -}}
      {{- fail "the reserved default study must not appear in nvflare_studies" -}}
    {{- end -}}
    {{- if not (regexMatch "^[a-z0-9]([a-z0-9_-]{0,61}[a-z0-9])?$" $study) -}}
      {{- fail (printf "invalid study name %q in nvflare_studies" $study) -}}
    {{- end -}}
    {{- if hasKey $seen $study -}}
      {{- fail (printf "duplicate study %q in nvflare_studies" $study) -}}
    {{- end -}}
    {{- $_ := set $seen $study true -}}
    {{- $sans = append $sans (dict "type" "uri" "value" (printf "%s%s" $studyURIBase $study)) -}}
  {{- end -}}
{{- end -}}
{
  "subject": {
    "commonName": {{ toJson $email }},
    "organization": {{ toJson $organization }},
    "extraNames": [
      {
        "type": "1.2.840.113549.1.9.2",
        "value": {{ toJson $flareRole }}
      }
    ]
  },
  "sans": {{ toJson $sans }},
{{- if typeIs "*rsa.PublicKey" .Insecure.CR.PublicKey }}
  "keyUsage": ["keyEncipherment", "digitalSignature"],
{{- else }}
  "keyUsage": ["digitalSignature"],
{{- end }}
  "extKeyUsage": ["clientAuth"]
}
