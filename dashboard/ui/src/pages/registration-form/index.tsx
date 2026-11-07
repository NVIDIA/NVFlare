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
import { Alert, Button, MenuItem, Paper, Stack, TextField, Typography } from '@mui/material'
import { Form, Formik } from 'formik'
import * as Yup from 'yup'
import { useRouter } from 'next/router'
import { APICall } from 'apis'
import { getErrorMessage, getErrorStatus, getValidationErrors } from 'api-errors'
import NoMenuLayout from 'components/NoMenuLayout'
import ParticipantSites from 'components/ParticipantSites'
import { siteRequest, toSiteView } from 'participant-sites'
import { getSession, setSession } from 'sess/session-manager'
import type { SiteView } from '../../types'

const accountSchema = Yup.object({
  name: Yup.string().trim().required('Name is required'),
  email: Yup.string().email('Enter a valid email address').required('Email is required'),
  password: Yup.string().required('Password is required'),
  confirm_password: Yup.string().oneOf([Yup.ref('password')], 'Passwords do not match').required('Confirm your password'),
})
const registrationSchema = Yup.object({
  organization: Yup.string().trim().required('Organization is required'),
  role: Yup.string().oneOf(['member', 'lead', 'org_admin']).required('Select a role'),
})
const roleNumbers: Record<string, number> = { member: 1, lead: 2, org_admin: 3, project_admin: 4 }

export default function RegistrationForm() {
  const { push } = useRouter()
  const [session] = useState(getSession)
  const [step, setStep] = useState(() => session.user.token && session.user.role === -1 ? 2 : 1)
  const [userId, setUserId] = useState(session.user.id)
  const [organization, setOrganization] = useState('')
  const [confirmed, setConfirmed] = useState(false)
  const [sites, setSites] = useState<SiteView[]>([])
  const [siteError, setSiteError] = useState('')
  const resuming = Boolean(session.user.token && session.user.role === -1)
  const [loadingProfile, setLoadingProfile] = useState(resuming)
  const [profileError, setProfileError] = useState('')
  const [registrationFields, setRegistrationFields] = useState({ organization: '', role: 'member' })
  const [organizationLocked, setOrganizationLocked] = useState(false)
  const [roleLocked, setRoleLocked] = useState(false)
  const [identityApproved, setIdentityApproved] = useState(false)

  useEffect(() => {
    if (!resuming) return
    let active = true
    async function resumeRegistration() {
      try {
        const response = await APICall.getUser(session.user.id)
        if (response?.status !== 'ok' || !response.user) throw new Error(response?.message || 'Unable to load your current registration.')
        if (!active) return
        const profile = response.user
        if (profile.role && profile.organization) {
          setSession(profile.email, session.user.token, profile.id, roleNumbers[profile.role] ?? 1, profile.organization)
          void push('/user-dashboard')
          return
        }
        setRegistrationFields({ organization: profile.organization, role: profile.role || 'member' })
        setOrganizationLocked(Boolean(profile.organization))
        setRoleLocked(Boolean(profile.role))
        setIdentityApproved(Number(profile.approval_state) >= 100)
      } catch (failure) {
        if (active) setProfileError(getErrorMessage(failure, 'Unable to load your current registration. Please reload and try again.'))
      } finally {
        if (active) setLoadingProfile(false)
      }
    }
    void resumeRegistration()
    return () => { active = false }
  }, [push, resuming, session.user.id, session.user.token])

  async function saveSite(site: SiteView) {
    const payload = siteRequest(site, organization)
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

  return (
    <NoMenuLayout title="User Registration">
      <Paper variant="outlined" sx={{ maxWidth: 850, p: 3 }}>
        {step === 1 && <Formik
          initialValues={{ name: '', email: '', password: '', confirm_password: '' }}
          validationSchema={accountSchema}
          onSubmit={async (values, helpers) => {
            helpers.setStatus(undefined)
            try {
              const response = await APICall.postUser(values)
              if (response?.status !== 'ok' || !response.user) throw new Error(response?.message || 'Unable to create your account.')
              let login
              try {
                login = await APICall.login({ email: values.email, password: values.password })
                if (login?.status !== 'ok' || !login.user || !login.access_token) throw new Error('Login failed')
              } catch {
                throw new Error('Your account was created. Please log in from the home page to continue registration.')
              }
              setSession(login.user.email, login.access_token, login.user.id, -1)
              setUserId(login.user.id)
              setStep(2)
            } catch (failure) {
              const errors = getValidationErrors(failure)
              if (errors) helpers.setErrors(errors)
              helpers.setStatus(getErrorMessage(failure, 'Unable to create your account. Please try again.'))
            } finally {
              helpers.setSubmitting(false)
            }
          }}
        >{formik => <Form><Stack spacing={2}>
          <Typography component="h2" variant="h6">Create your account</Typography>
          {formik.status && <Alert severity="error">{formik.status}</Alert>}
          <TextField label="Name" name="name" value={formik.values.name} onChange={formik.handleChange} onBlur={formik.handleBlur} disabled={formik.isSubmitting} error={Boolean(formik.touched.name && formik.errors.name)} helperText={formik.touched.name && formik.errors.name} />
          <TextField label="Email" name="email" type="email" value={formik.values.email} onChange={formik.handleChange} onBlur={formik.handleBlur} disabled={formik.isSubmitting} error={Boolean(formik.touched.email && formik.errors.email)} helperText={formik.touched.email && formik.errors.email} />
          <TextField label="Password" name="password" type="password" value={formik.values.password} onChange={formik.handleChange} onBlur={formik.handleBlur} disabled={formik.isSubmitting} error={Boolean(formik.touched.password && formik.errors.password)} helperText={formik.touched.password && formik.errors.password} />
          <TextField label="Confirm password" name="confirm_password" type="password" value={formik.values.confirm_password} onChange={formik.handleChange} onBlur={formik.handleBlur} disabled={formik.isSubmitting} error={Boolean(formik.touched.confirm_password && formik.errors.confirm_password)} helperText={formik.touched.confirm_password && formik.errors.confirm_password} />
          <Button variant="contained" type="submit" disabled={formik.isSubmitting}>Register</Button>
        </Stack></Form>}</Formik>}
        {step === 2 && loadingProfile && <Typography role="status">Loading your current registration…</Typography>}
        {step === 2 && profileError && <Alert severity="error">{profileError}</Alert>}
        {step === 2 && !loadingProfile && !profileError && <Formik
          initialValues={registrationFields}
          validationSchema={registrationSchema}
          onSubmit={async (values, helpers) => {
            helpers.setStatus(undefined)
            try {
              const response = await APICall.patchUser(userId, values)
              if (response?.status !== 'ok' || !response.user) throw new Error(response?.message || 'Unable to save your registration.')
              const session = getSession()
              setSession(session.user.email, session.user.token, response.user.id, roleNumbers[response.user.role] ?? -1, response.user.organization)
              setOrganization(response.user.organization)
              if (response.user.role === 'org_admin') {
                try {
                  const clients = await APICall.getClients()
                  if (clients.status !== 'ok') throw new Error('Unable to load client sites.')
                  setSites(clients.client_list.map(toSiteView))
                } catch (failure) {
                  setSiteError(getErrorMessage(failure, 'Unable to load your client sites. Open your dashboard to retry.'))
                }
                setConfirmed(true)
              } else {
                setStep(3)
              }
            } catch (failure) {
              const errors = getValidationErrors(failure)
              if (errors) Object.entries(errors).forEach(([field, message]) => helpers.setFieldError(field, message))
              helpers.setStatus(getErrorStatus(failure) === 401 ? 'Your session has expired. Please log in again.' : getErrorMessage(failure, 'Unable to save your registration. Please try again.'))
            } finally {
              helpers.setSubmitting(false)
            }
          }}
        >{formik => <Form><Stack spacing={2}>
          <Typography component="h2" variant="h6">Configure organization and role</Typography>
          <Typography>Request a role in your organization. The Project Admin must review and approve both before you can download startup kits.</Typography>
          {formik.status && <Alert severity="error">{formik.status}</Alert>}
          {identityApproved && <Alert severity="warning">Your account is already approved. Contact the Project Admin to complete or correct its role and organization.</Alert>}
          <TextField label="Organization" name="organization" value={formik.values.organization} onChange={formik.handleChange} onBlur={formik.handleBlur} disabled={confirmed || identityApproved || organizationLocked || formik.isSubmitting} error={Boolean(formik.touched.organization && formik.errors.organization)} helperText={formik.touched.organization && formik.errors.organization} />
          <TextField select label="Role" name="role" value={formik.values.role} onChange={formik.handleChange} disabled={confirmed || identityApproved || roleLocked || formik.isSubmitting}>
            <MenuItem value="member">Member</MenuItem><MenuItem value="lead">Lead</MenuItem><MenuItem value="org_admin">Org Admin</MenuItem>
          </TextField>
          {!confirmed && <Button variant="contained" type="submit" disabled={identityApproved || formik.isSubmitting}>Save</Button>}
          {confirmed && <>
            <Alert severity="info">Org Admin request pending approval: register your own client sites here.</Alert>
            {siteError && <Alert severity="error">{siteError}</Alert>}
            <ParticipantSites sites={sites} organization={organization} onSave={saveSite} onDelete={deleteSite} />
            <Button variant="contained" onClick={() => setStep(3)}>Next</Button>
          </>}
        </Stack></Form>}</Formik>}
        {step === 3 && <Stack spacing={2}>
          <Typography component="h2" variant="h6">Thank you for completing your registration.</Typography>
          <Typography>Your role and organization, and any client sites, remain pending until the Project Admin approves them. Startup-kit downloads also require a frozen project.</Typography>
          <Button variant="contained" onClick={() => void push('/user-dashboard')}>Go to my dashboard</Button>
        </Stack>}
      </Paper>
    </NoMenuLayout>
  )
}
