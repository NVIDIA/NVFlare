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

'use client';
import { useState, useEffect } from "react";
import AppCard from "../components/AppCard";
import dynamic from "next/dynamic";

const ThemeProvider = dynamic(
  () => import("@kui-react/theme").then((m) => m.ThemeProvider),
  {
    ssr: false,
  }
);

const AppsPage = () => {
  const [data, setData] = useState<string[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Fetch data from visualization backend API using useEffect
  const fetchData = () => {
    (async () => {
      try {
        const authorizationHeader = `Bearer ${process.env.NEXT_PUBLIC_AUTHORIZATION_HEADER}`;
        const res = await fetch(`${process.env.NEXT_PUBLIC_ROOT_URI}/get_apps/`, {
          method: 'GET',
          headers: {
            'Authorization': `${authorizationHeader}`,
            'Content-Type': 'application/json'
          }
        });
        if (!res.ok) {
          throw new Error(`An error occurred while fetching the list of registered apps: ${res.statusText}`);
        }
        const result = await res.json();
        setData(result);
      } catch (err: unknown) {
        if (err instanceof Error){
          setError((err as Error).message);
        }
      }
    })();
  };

  useEffect(() => {
    fetchData();
  }, []);

  return (
    <ThemeProvider theme="dark" withFonts withReset>
      <div style={{ padding: '20px' }}>
        {error && <p>Error: {error}</p>}
        {!data && !error && <p>Loading...</p>}
        {data && (
          <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
            {data.map(app => (
              <AppCard title={app} key={app} />
            ))}
          </div>
        )}
      </div>
    </ThemeProvider>
  );
};

export default AppsPage;
