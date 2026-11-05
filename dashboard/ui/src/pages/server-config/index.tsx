// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { FormEvent, useEffect, useState } from 'react'
import { Alert, Box, Button, Checkbox, FormControlLabel, MenuItem, Paper, Stack, TextField, Typography } from '@mui/material'
import Layout from 'components/Layout'
import { APICall } from 'apis'
import { getErrorMessage, getValidationErrors } from 'api-errors'
import { Project } from 'types'

export default function ServerConfiguration() {
  const [project, setProject] = useState<Project>()
  const [server, setServer] = useState('')
  const [cc, setCc] = useState(false)
  const [environment, setEnvironment] = useState('bare_metal_cvm')
  const [cpu, setCpu] = useState('amd')
  const [error, setError] = useState('')
  const [fieldError, setFieldError] = useState('')
  const [busy, setBusy] = useState(false)
  const [saved, setSaved] = useState(false)

  function load(configuration: Project) {
    setProject(configuration)
    setServer(configuration.server1 || '')
    setCc(Boolean(configuration.server_props?.cc_enabled))
    setEnvironment(configuration.server_props?.deployment_environment || 'bare_metal_cvm')
    setCpu(configuration.server_props?.cpu_family || 'amd')
  }

  useEffect(() => {
    APICall.getProject().then(response => {
      if (response.status !== 'ok' || !response.project) throw new Error(response.message || 'Unable to load server settings.')
      load(response.project)
    }).catch(error => setError(getErrorMessage(error, 'Unable to load server settings.')))
  }, [])

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setBusy(true)
    setError('')
    setFieldError('')
    setSaved(false)
    try {
      const response = await APICall.patchProject({ server1: server, server_props: { cc_enabled: cc, deployment_environment: environment, cpu_family: cpu } })
      if (response.status !== 'ok' || !response.project) throw new Error(response.message || 'Unable to save server settings.')
      load(response.project)
      setSaved(true)
    } catch (error) {
      setFieldError(getValidationErrors(error)?.server1 || '')
      setError(getErrorMessage(error, 'Unable to save server settings.'))
    } finally {
      setBusy(false)
    }
  }

  return <Layout title="Server Configuration">
    <Stack spacing={2} sx={{ maxWidth: 800 }}>
      {error && <Alert severity="error">{error}</Alert>}
      {saved && <Alert severity="success">Server settings saved.</Alert>}
      {project?.frozen && <Alert severity="info">This project has been frozen. Server settings are no longer editable.</Alert>}
      {project && <Paper sx={{ p: 3 }}><Box component="form" onSubmit={submit}>
        <Stack spacing={2}>
          <Typography variant="h6" component="h1">Server Settings</Typography>
          <TextField name="server1" label="Server hostname or IP address" value={server} required disabled={busy || project.frozen} error={Boolean(fieldError)} helperText={fieldError} onChange={event => { setServer(event.target.value); setSaved(false) }} />
          <FormControlLabel label="CC Enabled" control={<Checkbox checked={cc} disabled={busy || project.frozen} onChange={event => { setCc(event.target.checked); setSaved(false) }} />} />
          {cc && <>
            <TextField select label="Target Deployment Environment" value={environment} disabled={busy || project.frozen} onChange={event => { setEnvironment(event.target.value); setSaved(false) }}><MenuItem value="azure_cc">Azure CC</MenuItem><MenuItem value="bare_metal_cvm">Bare metal: CVM</MenuItem></TextField>
            <TextField select label="CPU Family" value={cpu} disabled={busy || project.frozen} onChange={event => { setCpu(event.target.value); setSaved(false) }}><MenuItem value="amd">AMD</MenuItem><MenuItem value="intel">Intel</MenuItem></TextField>
          </>}
          <Button type="submit" variant="contained" disabled={busy || project.frozen}>Save</Button>
        </Stack>
      </Box></Paper>}
    </Stack>
  </Layout>
}
