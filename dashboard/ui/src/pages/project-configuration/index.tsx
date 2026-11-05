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
import dayjs from 'dayjs'
import { Alert, Box, Button, Paper, Stack, TextField, Typography } from '@mui/material'
import Layout from 'components/Layout'
import { DatePicker } from 'components/shared/DatePicker'
import { APICall } from 'apis'
import { getErrorMessage, getValidationErrors } from 'api-errors'
import { Project } from 'types'

export default function ProjectConfiguration() {
  const [project, setProject] = useState<Project>()
  const [draft, setDraft] = useState<Project>()
  const [error, setError] = useState('')
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState(false)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    APICall.getProject().then(response => {
      if (response.status !== 'ok' || !response.project) throw new Error(response.message || 'Unable to load the project.')
      setProject(response.project)
      setDraft(response.project)
    }).catch(error => setError(getErrorMessage(error, 'Unable to load the project.')))
  }, [])

  function change(field: keyof Project, value: string) {
    setDraft(previous => previous ? { ...previous, [field]: value } : previous)
    setSaved(false)
  }

  async function persist(values: Partial<Project>, preserveDraft = false) {
    setBusy(true)
    setError('')
    setErrors({})
    setSaved(false)
    try {
      const response = await APICall.patchProject(values)
      if (response.status !== 'ok' || !response.project) throw new Error(response.message || 'Unable to save the project.')
      setProject(response.project)
      setDraft(previous => preserveDraft && previous ? { ...previous, public: response.project.public, frozen: response.project.frozen } : response.project)
      setSaved(true)
    } catch (error) {
      setErrors(getValidationErrors(error) || {})
      setError(getErrorMessage(error, 'Unable to save the project.'))
    } finally {
      setBusy(false)
    }
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!draft) return
    if (draft.starting_date && draft.end_date && dayjs(draft.starting_date).isAfter(dayjs(draft.end_date))) {
      setError('The project start date must be before the end date.')
      return
    }
    await persist({ short_name: draft.short_name, title: draft.title, description: draft.description,
      app_location: draft.app_location, starting_date: draft.starting_date, end_date: draft.end_date })
  }

  return <Layout title="Project Configuration">
    <Stack spacing={2} sx={{ maxWidth: 800 }}>
      {error && <Alert severity="error">{error}</Alert>}
      {saved && <Alert severity="success">Project settings saved.</Alert>}
      {project?.frozen && <Alert severity="info">This project has been frozen. Project values are no longer editable.</Alert>}
      {draft && <Paper sx={{ p: 3 }}><Box component="form" onSubmit={submit}>
        <Stack spacing={2}>
          <Typography variant="h6" component="h1">Project Values</Typography>
          <TextField name="short_name" label="Project Short Name" value={draft.short_name} required slotProps={{ htmlInput: { maxLength: 16 } }} disabled={busy || project?.frozen} error={Boolean(errors.short_name)} helperText={errors.short_name || 'Maximum 16 characters; used in certificates.'} onChange={event => change('short_name', event.target.value)} />
          <TextField name="title" label="Title" value={draft.title || ''} disabled={busy || project?.frozen} onChange={event => change('title', event.target.value)} />
          <TextField name="description" label="Project Description" value={draft.description || ''} multiline minRows={3} disabled={busy || project?.frozen} onChange={event => change('description', event.target.value)} />
          <TextField name="app_location" label="Docker Download Link" value={draft.app_location || ''} disabled={busy || project?.frozen} onChange={event => change('app_location', event.target.value)} />
          <DatePicker label="Project Start Date" value={draft.starting_date ? dayjs(draft.starting_date).toDate() : null} disabled={busy || project?.frozen} maxDate={draft.end_date ? dayjs(draft.end_date).toDate() : undefined} onChange={date => change('starting_date', date ? dayjs(date).format('YYYY-MM-DD') : '')} />
          <DatePicker label="Project End Date" value={draft.end_date ? dayjs(draft.end_date).toDate() : null} disabled={busy || project?.frozen} minDate={draft.starting_date ? dayjs(draft.starting_date).toDate() : undefined} onChange={date => change('end_date', date ? dayjs(date).format('YYYY-MM-DD') : '')} />
          <Button type="submit" variant="contained" disabled={busy || project?.frozen || JSON.stringify(draft) === JSON.stringify(project)}>Save</Button>
          <Button disabled={busy || project?.frozen || project?.public} onClick={() => persist({ public: true }, true)}>{project?.public ? 'Project is Public' : 'Make Project Public'}</Button>
          <Typography>If the project is public, participants can sign up. Complete project and server settings before freezing the project on the home page.</Typography>
        </Stack>
      </Box></Paper>}
    </Stack>
  </Layout>
}
