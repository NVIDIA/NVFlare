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

import { ReactNode, useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/router'
import { AppBar, Box, Button, Drawer, IconButton, List, ListItemButton, ListItemText, Toolbar, Typography } from '@mui/material'
import MenuIcon from '@mui/icons-material/Menu'
import ProjectName from '../ProjectName'
import { getSession, setSession } from '../../sess/session-manager'

const width = 240
const adminLinks = [
  ['Project Home', '/'], ['My Info', '/user-dashboard'], ['Users Dashboard', '/project-admin-dashboard'],
  ['Client Sites', '/site-dashboard'], ['Project Configuration', '/project-configuration'],
  ['Server Configuration', '/server-config'], ['Downloads', '/downloads'],
]
const participantLinks = [['Project Home', '/'], ['My Info', '/user-dashboard'], ['Downloads', '/downloads']]

export default function Layout({ children, headerChildren, title }: { children: ReactNode; headerChildren?: ReactNode; title: string }) {
  const { pathname, push } = useRouter()
  const session = getSession()
  const [menuOpen, setMenuOpen] = useState(false)
  const links = session.user.role === 4 ? adminLinks : session.user.role > 0 ? participantLinks : session.user.role === -1 ? [['Continue Registration', '/registration-form']] : [['Login', '/'], ['User Registration Form', '/registration-form']]
  const navigation = <List aria-label="Dashboard navigation">{links.map(([label, href]) =>
    <ListItemButton key={href} component={Link} href={href} selected={pathname === href} onClick={() => setMenuOpen(false)}><ListItemText primary={label} /></ListItemButton>
  )}</List>
  function logout() { setSession('', '', -1, 0); void push('/') }
  return <Box sx={{ display: 'flex', minHeight: '100vh' }}>
    <AppBar position="fixed" sx={{ zIndex: theme => theme.zIndex.drawer + 1 }}><Toolbar>
      <IconButton color="inherit" aria-label="Open navigation" edge="start" onClick={() => setMenuOpen(true)} sx={{ mr: 2, display: { md: 'none' } }}><MenuIcon /></IconButton>
      <Typography variant="h6" sx={{ flexGrow: 1 }}><ProjectName /></Typography>
      {session.user.token && <Button color="inherit" onClick={logout}>Logout</Button>}
    </Toolbar></AppBar>
    <Box component="nav" sx={{ width: { md: width }, flexShrink: { md: 0 } }}>
      <Drawer variant="temporary" open={menuOpen} onClose={() => setMenuOpen(false)} sx={{ display: { xs: 'block', md: 'none' }, '& .MuiDrawer-paper': { width } }}><Toolbar />{navigation}</Drawer>
      <Drawer variant="permanent" sx={{ display: { xs: 'none', md: 'block' }, '& .MuiDrawer-paper': { width, boxSizing: 'border-box' } }}><Toolbar />{navigation}</Drawer>
    </Box>
    <Box component="main" sx={{ flexGrow: 1, minWidth: 0, p: { xs: 2, md: 4 } }}><Toolbar />
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}><Typography variant="h4" component="h1">{title}</Typography>{headerChildren}</Box>
      {children}
    </Box>
  </Box>
}
