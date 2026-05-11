{{/*
NVFlare client chart helpers
*/}}
{{- define "nvflare-client.name" -}}
{{- default (default .Chart.Name .Values.name) .Values.serviceName | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "nvflare-client.labels" -}}
app.kubernetes.io/name: {{ include "nvflare-client.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "nvflare-client.selectorLabels" -}}
app.kubernetes.io/name: {{ include "nvflare-client.name" . }}
{{- end }}

{{- define "nvflare-client.image" -}}
{{- if .Values.image.tag }}
{{- printf "%s:%s" .Values.image.repository .Values.image.tag }}
{{- else }}
{{- .Values.image.repository }}
{{- end }}
{{- end }}
