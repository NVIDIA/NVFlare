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
import Link from 'next/link'
import { useRouter } from 'next/router'
import { Alert, Box, Button, Dialog, DialogActions, DialogContent, DialogTitle, Paper, Stack, TextField, Typography } from '@mui/material'
import { GridColDef } from '@mui/x-data-grid'
import Layout from 'components/Layout'
import Table from 'components/shared/Table'
import { APICall } from 'apis'
import { getErrorMessage, getErrorStatus } from 'api-errors'
import { getSession, setSession } from 'sess/session-manager'
import { ClientFromSchema, Project } from 'types'
import siteparams from '../../siteparams.json'

const columns: GridColDef<ClientFromSchema>[] = [
  { field: 'name', headerName: 'Client Site', flex: 1, minWidth: 160 },
  { field: 'organization', headerName: 'Organization', flex: 1, minWidth: 140 },
  { field: 'capacity', headerName: 'Capacity', flex: 1, minWidth: 240,
    renderCell: ({ row }) => `NUM GPUS: ${row.capacity?.num_of_gpus ?? 0}; MEM/GPU: ${row.capacity?.mem_per_gpu_in_GiB ?? 0} GiB` },
]

export default function ProjectHome() {
  const { push } = useRouter()
  const session = getSession()
  const [project, setProject] = useState<Project>()
  const [clients, setClients] = useState<ClientFromSchema[]>([])
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [confirmFreeze, setConfirmFreeze] = useState(false)
  const canFreeze = Boolean(project?.public && !project.frozen)

  useEffect(() => {
    APICall.getProject().then(response => {
      if (response.status !== 'ok' || !response.project) throw new Error(response.message || 'Unable to load the project.')
      setProject(response.project)
    }).catch(error => setError(getErrorMessage(error, 'Unable to connect to the Dashboard API.')))
    if (session.user.role > 0) {
      APICall.getClients().then(response => setClients(response.client_list.filter(client => client.approval_state >= 100)))
        .catch(error => setError(getErrorMessage(error, 'Unable to load client sites.')))
    }
  }, [session.user.role, session.user.token])

  async function login(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setBusy(true)
    setError('')
    try {
      const response = await APICall.login({ email, password })
      if (response.status !== 'ok' || !response.user || !response.access_token) throw new Error(response.message || 'Unable to log in.')
      const roles: Record<string, number> = { project_admin: 4, org_admin: 3, lead: 2, member: 1, '': -1 }
      const role = roles[response.user.role]
      if (role === undefined) throw new Error('This account has an unsupported Dashboard role. Contact the project administrator.')
      setSession(response.user.email, response.access_token, response.user.id, role, response.user.organization)
      await push(role === -1 ? '/registration-form' : role === 4 ? (project?.public ? '/project-admin-dashboard' : '/project-configuration') : '/user-dashboard')
    } catch (error) {
      setError(getErrorStatus(error) === 401 ? 'Unable to log in. Check your email and password.' : getErrorMessage(error, 'Unable to log in. Check your email and password.'))
    } finally {
      setBusy(false)
    }
  }

  async function freeze() {
    if (!canFreeze || busy) return
    setBusy(true)
    setError('')
    try {
      const response = await APICall.patchProject({ frozen: true })
      if (response.status !== 'ok' || !response.project) throw new Error(response.message || 'Unable to freeze the project.')
      setProject(response.project)
      setConfirmFreeze(false)
    } catch (error) {
      setError(getErrorMessage(error, 'Unable to freeze the project. Please check the server configuration.'))
    } finally {
      setBusy(false)
    }
  }

  return <Layout title="Project Home">
    <Stack spacing={3} sx={{ maxWidth: 1100 }}>
      {error && <Alert severity="error">{error}</Alert>}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h5" component="h1">{project?.title || siteparams.projectname}</Typography>
        <Typography sx={{ whiteSpace: 'pre-wrap', my: 2 }}>{project?.description}</Typography>
        {project && <Typography>Project status: {project.frozen ? 'Frozen' : 'Setup in progress'}</Typography>}
        {project?.app_location && <Button component="a" href={project.app_location} target="_blank" rel="noopener noreferrer">Application download</Button>}
        {session.user.role === 4 && <><Stack direction="row" spacing={2} sx={{ mt: 2 }}>
          <Button component={Link} href="/project-configuration">Project Configuration</Button>
          <Button component={Link} href="/server-config">Server Configuration</Button>
          <Button variant="contained" disabled={!canFreeze || busy} onClick={() => { setError(''); setConfirmFreeze(true) }}>Freeze Project</Button>
        </Stack>
          {project && !project.public && !project.frozen && <Alert severity="info" sx={{ mt: 2 }}>Make the project public in Project Configuration before freezing it.</Alert>}
        </>}
      </Paper>
      {session.user.role === 0 ? <Paper sx={{ p: 3, maxWidth: 480 }}>
        <Typography variant="h6" component="h2">Log in</Typography>
        {session.expired && <Alert severity="warning" sx={{ my: 2 }}>Your session has expired. Please log in again.</Alert>}
        <Box component="form" onSubmit={login}>
          <Stack spacing={2} sx={{ mt: 2 }}>
            <TextField name="email" label="Email" type="email" value={email} required autoComplete="username" onChange={event => setEmail(event.target.value)} />
            <TextField name="password" label="Password" type="password" value={password} required autoComplete="current-password" onChange={event => setPassword(event.target.value)} />
            <Button type="submit" variant="contained" disabled={busy}>Log in</Button>
            {project?.public && <Button component={Link} href="/registration-form">Sign Up</Button>}
          </Stack>
        </Box>
      </Paper> : <Paper sx={{ p: 3 }}>
        <Typography variant="h6" component="h2" sx={{ mb: 2 }}>Approved Client Sites</Typography>
        <Table columns={columns} data={clients} />
      </Paper>}
    </Stack>
    <Dialog open={confirmFreeze} onClose={() => !busy && setConfirmFreeze(false)}>
      <DialogTitle>Freeze Project</DialogTitle>
      <DialogContent>
        <Typography>Freeze the completed project configuration and create its trust material? Project settings cannot be edited afterward.</Typography>
        {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
      </DialogContent>
      <DialogActions><Button disabled={busy} onClick={() => setConfirmFreeze(false)}>Cancel</Button><Button disabled={!canFreeze || busy} onClick={freeze}>Freeze</Button></DialogActions>
    </Dialog>
  </Layout>
}
