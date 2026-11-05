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

import { useEffect, useState } from 'react'
import { Alert, Button, Dialog, DialogActions, DialogContent, DialogTitle, Paper, Stack, Table, TableBody, TableCell, TableHead, TableRow, Typography } from '@mui/material'
import Layout from 'components/Layout'
import { APICall } from 'apis'
import { getDownloadErrorMessage, getErrorMessage } from 'api-errors'
import { getSession } from 'sess/session-manager'
import { ClientFromSchema, Project, User } from 'types'

function makePin(): string {
  const bytes = new Uint8Array(6)
  crypto.getRandomValues(bytes)
  return Array.from(bytes, value => value.toString(16).padStart(2, '0')).join('').toUpperCase()
}

export default function Downloads() {
  const session = getSession()
  const [user, setUser] = useState<User>()
  const [project, setProject] = useState<Project>()
  const [clients, setClients] = useState<ClientFromSchema[]>([])
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [pin, setPin] = useState('')
  const approved = Number(user?.approval_state) >= 100
  const ready = approved && Boolean(project?.frozen) && !busy

  useEffect(() => {
    Promise.all([APICall.getUser(session.user.id), APICall.getProject(), APICall.getClients()])
      .then(([profile, configuration, sites]) => {
        if (profile.status !== 'ok' || !profile.user || configuration.status !== 'ok' || !configuration.project || sites.status !== 'ok') {
          throw new Error(profile.message || configuration.message || sites.message || 'Unable to load the downloads page.')
        }
        setUser(profile.user)
        setProject(configuration.project)
        setClients(sites.client_list)
      }).catch(error => setError(getErrorMessage(error, 'Unable to load downloads. Please log in again or contact the project administrator.')))
  }, [session.user.id, session.user.token])

  async function download(kind: 'user' | 'server' | 'client', client?: ClientFromSchema) {
    setError('')
    setBusy(true)
    const nextPin = makePin()
    try {
      if (kind === 'client' && client) await APICall.getClientStartupKit(client.id, `${client.name}.zip`, nextPin)
      else if (kind === 'server') await APICall.getServerStartupKit('server.zip', nextPin)
      else if (user) await APICall.getUserStartupKit(user.id, 'flare-console.zip', nextPin)
      else throw new Error('Your profile is not available. Please reload the page.')
      setPin(nextPin)
    } catch (error) {
      setError(await getDownloadErrorMessage(error))
    } finally {
      setBusy(false)
    }
  }

  return <Layout title="Downloads">
    <Stack spacing={3} sx={{ maxWidth: 1100 }}>
      {error && <Alert severity="error">{error}</Alert>}
      {user && !approved && <Alert severity="warning">Your requested role and organization are awaiting approval by the Project Admin.</Alert>}
      {project && !project.frozen && <Alert severity="info">The Project Admin must complete and freeze project setup before startup kits can be downloaded.</Alert>}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" component="h1" sx={{ mb: 2 }}>FLARE Console</Typography>
        <Button variant="contained" disabled={!ready} onClick={() => download('user')}>Download FLARE Console</Button>
      </Paper>
      {user?.role === 'project_admin' && <Paper sx={{ p: 3 }}>
        <Typography variant="h6" component="h2" sx={{ mb: 2 }}>Server Startup Kit</Typography>
        <Button variant="contained" disabled={!ready} onClick={() => download('server')}>Download Server Startup Kit</Button>
      </Paper>}
      {(user?.role === 'org_admin' || user?.role === 'project_admin') && <Paper sx={{ p: 3, overflowX: 'auto' }}>
        <Typography variant="h6" component="h2">Client Startup Kits</Typography>
        <Table aria-label="Client startup kits"><TableHead><TableRow><TableCell>Site Name</TableCell><TableCell>Organization</TableCell><TableCell>Approval</TableCell><TableCell>Startup Kit</TableCell></TableRow></TableHead>
          <TableBody>{clients.map(client => <TableRow key={client.id}>
            <TableCell>{client.name}</TableCell><TableCell>{client.organization}</TableCell>
            <TableCell>{client.approval_state >= 100 ? 'Approved' : client.approval_state < 0 ? 'Denied' : 'Pending'}</TableCell>
            <TableCell><Button disabled={!ready || client.approval_state < 100} onClick={() => download('client', client)}>Download Startup Kit</Button></TableCell>
          </TableRow>)}</TableBody>
        </Table>
        {!clients.length && <Typography sx={{ mt: 2 }}>No client sites are available.</Typography>}
      </Paper>}
    </Stack>
    <Dialog open={Boolean(pin)} onClose={() => setPin('')}>
      <DialogTitle>Startup kit downloaded</DialogTitle>
      <DialogContent><Typography>The secure PIN is: <strong>{pin}</strong></Typography><Typography sx={{ mt: 2 }}>Record this PIN to open the encrypted ZIP file.</Typography></DialogContent>
      <DialogActions><Button onClick={() => setPin('')}>Okay</Button></DialogActions>
    </Dialog>
  </Layout>
}
