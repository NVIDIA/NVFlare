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
import { Alert, Button, Dialog, DialogContent, DialogTitle, Paper, Stack, TextField, Typography } from '@mui/material'
import { Form, Formik } from 'formik'
import * as Yup from 'yup'
import { APICall } from 'apis'
import { getErrorMessage } from 'api-errors'
import Layout from 'components/Layout'
import ParticipantSites from 'components/ParticipantSites'
import { siteRequest, toSiteView } from 'participant-sites'
import { getSession } from 'sess/session-manager'
import type { SiteView, User } from '../../types'

const passwordSchema = Yup.object({
  password: Yup.string().required('Enter a new password'),
  confirm_password: Yup.string().oneOf([Yup.ref('password')], 'Passwords do not match').required('Confirm your password'),
})
const roleNames: Record<string, string> = { org_admin: 'Org Admin', member: 'Member', lead: 'Lead', project_admin: 'Project Admin' }

export default function UserDashboard() {
  const [user, setUser] = useState<User | null>(null)
  const [sites, setSites] = useState<SiteView[]>([])
  const [sitesLoading, setSitesLoading] = useState(true)
  const [error, setError] = useState('')
  const [profileOpen, setProfileOpen] = useState(false)

  useEffect(() => {
    let active = true
    async function loadProfile() {
      try {
        const response = await APICall.getUser(getSession().user.id)
        if (response?.status !== 'ok' || !response.user) throw new Error('Unable to load your profile.')
        if (!active) return
        setUser(response.user)
        if (response.user.role === 'org_admin' && Number(response.user.approval_state) >= 0) {
          const clients = await APICall.getClients()
          if (clients.status !== 'ok') throw new Error('Unable to load your client sites.')
          if (active) setSites(clients.client_list.map(toSiteView))
        }
      } catch (failure) {
        if (active) setError(getErrorMessage(failure, 'Unable to load your dashboard. Please reload to try again.'))
      } finally {
        if (active) setSitesLoading(false)
      }
    }
    void loadProfile()
    return () => { active = false }
  }, [])

  async function saveSite(site: SiteView) {
    if (!user) throw new Error('Reload your profile before saving client sites.')
    const payload = siteRequest(site, user.organization)
    const response = site.id === -1 ? await APICall.postClient(payload) : await APICall.patchClient(site.id, payload)
    if (response?.status !== 'ok' || !response.client) throw new Error(response?.message || 'Unable to save the client site.')
    const saved = toSiteView(response.client)
    setSites(previous => site.id === -1 ? [...previous, saved] : previous.map(entry => entry.id === saved.id ? saved : entry))
  }

  async function deleteSite(site: SiteView) {
    const response = await APICall.deleteClient(site.id)
    if (response?.status !== 'ok') throw new Error(response?.message || 'Unable to delete the client site.')
    setSites(previous => previous.filter(entry => entry.id !== site.id))
  }

  const approval = Number(user?.approval_state)
  const pending = approval >= 0 && approval < 100

  return (
    <Layout title="User Dashboard">
      <Stack spacing={3} sx={{ maxWidth: 950 }}>
        {error && <Alert severity="error">{error}</Alert>}
        {!user && !error && <Typography role="status">Loading your profile…</Typography>}
        {user && <>
          {pending && <Alert severity="warning">Your requested role and organization are awaiting approval by the Project Admin. Once approved, you can download your FLARE Console{user.role === 'org_admin' ? ' and Client Site Startup Kits' : ''} from the Downloads page after the project is frozen.</Alert>}
          {approval < 0 && <Alert severity="error">Your registration has been denied by the Project Admin.</Alert>}
          <Paper variant="outlined" sx={{ p: 3 }}>
            <Stack spacing={1}>
              <Stack direction="row" sx={{ justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography component="h2" variant="h6">{user.name}</Typography>
                <Button variant="outlined" onClick={() => setProfileOpen(true)}>Edit My Profile</Button>
              </Stack>
              <Typography>{user.email}</Typography>
              <Typography>Organization: {user.organization}</Typography>
              <Typography>{pending ? 'Requested role: ' : 'Role: '}{roleNames[user.role] ?? user.role}</Typography>
            </Stack>
          </Paper>
          {user.role === 'org_admin' && approval >= 0 && <ParticipantSites sites={sites} organization={user.organization} loading={sitesLoading} onSave={saveSite} onDelete={deleteSite} />}
        </>}
      </Stack>
      <Dialog open={profileOpen} onClose={(_event, reason) => { if (reason !== 'backdropClick') setProfileOpen(false) }} fullWidth maxWidth="sm">
        <DialogTitle>Edit user details</DialogTitle>
        <DialogContent>
          <Formik initialValues={{ password: '', confirm_password: '' }} validationSchema={passwordSchema} onSubmit={async (values, helpers) => {
            helpers.setStatus(undefined)
            try {
              const response = await APICall.patchUser(user!.id, { password: values.password })
              if (response?.status !== 'ok' || !response.user) throw new Error(response?.message || 'Unable to update your password.')
              setUser(response.user)
              setProfileOpen(false)
            } catch (failure) {
              helpers.setStatus(getErrorMessage(failure, 'Unable to update your password. Please try again.'))
            } finally {
              helpers.setSubmitting(false)
            }
          }}>{formik => <Form><Stack spacing={2} sx={{ pt: 1 }}>
            <Typography>Contact the Project Admin to change your name, email, organization, or role.</Typography>
            {formik.status && <Alert severity="error">{formik.status}</Alert>}
            <TextField label="New password" name="password" type="password" value={formik.values.password} onChange={formik.handleChange} onBlur={formik.handleBlur} error={Boolean(formik.touched.password && formik.errors.password)} helperText={formik.touched.password && formik.errors.password} disabled={formik.isSubmitting} />
            <TextField label="Confirm new password" name="confirm_password" type="password" value={formik.values.confirm_password} onChange={formik.handleChange} onBlur={formik.handleBlur} error={Boolean(formik.touched.confirm_password && formik.errors.confirm_password)} helperText={formik.touched.confirm_password && formik.errors.confirm_password} disabled={formik.isSubmitting} />
            <Stack direction="row" spacing={1} sx={{ justifyContent: 'flex-end' }}>
              <Button onClick={() => setProfileOpen(false)} disabled={formik.isSubmitting}>Cancel</Button>
              <Button type="submit" variant="contained" disabled={formik.isSubmitting}>Save</Button>
            </Stack>
          </Stack></Form>}</Formik>
        </DialogContent>
      </Dialog>
    </Layout>
  )
}
