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

import { ReactNode } from 'react'
import Link from 'next/link'
import { AppBar, Box, Button, Toolbar, Typography } from '@mui/material'
import ProjectName from '../ProjectName'

export default function NoMenuLayout({ children, title, headerChildren }: { children: ReactNode; title: string; headerChildren?: ReactNode }) {
  return <Box sx={{ minHeight: '100vh' }}>
    <AppBar position="static"><Toolbar><Typography variant="h6" sx={{ flexGrow: 1 }}><ProjectName /></Typography><Button color="inherit" component={Link} href="/">Login</Button></Toolbar></AppBar>
    <Box component="main" sx={{ maxWidth: 1100, mx: 'auto', p: { xs: 2, md: 4 } }}><Typography variant="h4" component="h1" sx={{ mb: 3 }}>{title}</Typography>{headerChildren}{children}</Box>
  </Box>
}
