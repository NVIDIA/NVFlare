/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

'use client'
import "./globals.css";
import { AppBar } from "@kui-react/app-bar";
import { HorizontalNavItem, HorizontalNavLink, HorizontalNavList, HorizontalNavRoot } from "@kui-react/horizontal-nav";
import { ThemeProvider } from '@kui-react/theme'

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
      <ThemeProvider
        theme="dark"
        withFonts
        withReset
      >
        <AppBar
          slotLeft={<h2 style={{color: "white"}}>Holoscan Federated Analytics | Dashboard</h2>}
          slotCenter={
            <HorizontalNavRoot>
              <HorizontalNavList>
                <HorizontalNavItem style={{width:"auto"}}>
                  <HorizontalNavLink to="/apps">
                    Apps
                  </HorizontalNavLink>
                </HorizontalNavItem>
                <HorizontalNavItem>
                  <HorizontalNavLink to="/about">
                    About
                  </HorizontalNavLink>
                </HorizontalNavItem>
              </HorizontalNavList>
            </HorizontalNavRoot>
           }
        />
        {children}
      </ThemeProvider>
     </body>
    </html>
  );
}
