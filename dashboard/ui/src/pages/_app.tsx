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

import { useSyncExternalStore } from 'react'
import { ThemeProvider } from '@mui/material/styles'
import CssBaseline from '@mui/material/CssBaseline'
import type { AppProps } from 'next/app'
import theme from '../../styles/customtheme'
import '../page-styles/globals.css'

const subscribe = () => () => {}
const clientSnapshot = () => true
const serverSnapshot = () => false

export default function App({ Component, pageProps }: AppProps) {
  const hydrated = useSyncExternalStore(subscribe, clientSnapshot, serverSnapshot)
  return <ThemeProvider theme={theme}><CssBaseline />{hydrated ? <Component {...pageProps} /> : <p>Loading dashboard…</p>}</ThemeProvider>
}
