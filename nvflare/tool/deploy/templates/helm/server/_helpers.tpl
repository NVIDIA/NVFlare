{{/*
NVFlare server chart helpers
*/}}
{{- define "nvflare-server.name" -}}
{{- default .Chart.Name .Values.name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "nvflare-server.labels" -}}
app.kubernetes.io/name: {{ include "nvflare-server.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "nvflare-server.selectorLabels" -}}
app.kubernetes.io/name: {{ include "nvflare-server.name" . }}
{{- end }}

{{- define "nvflare-server.image" -}}
{{- if .Values.image.tag }}
{{- printf "%s:%s" .Values.image.repository .Values.image.tag }}
{{- else }}
{{- .Values.image.repository }}
{{- end }}
{{- end }}

{{- define "nvflare-server.hostPortEnabled" -}}
{{- if hasKey .Values "hostPortEnabled" }}
{{- .Values.hostPortEnabled | toString }}
{{- else }}
{{- "false" }}
{{- end }}
{{- end }}

{{- define "nvflare-server.tcpConfigMapEnabled" -}}
{{- if hasKey .Values "tcpConfigMapEnabled" }}
{{- .Values.tcpConfigMapEnabled | toString }}
{{- else }}
{{- "false" }}
{{- end }}
{{- end }}
