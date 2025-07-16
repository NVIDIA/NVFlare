# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Stage 1: Build the Next.js application
FROM node:20.18-alpine AS builder

# Install pnpm
RUN npm install -g pnpm

# Set the working directory inside the container
WORKDIR /app

# Copy package.json, pnpm-lock.yaml, and .npmrc (if available)
COPY package.json pnpm-lock.yaml ./
# COPY .npmrc .npmrc

# Copy local packages
COPY vendor ./vendor

# Install dependencies using pnpm
RUN pnpm install

# Copy the rest of the application code to the container
COPY src ./src
COPY next.config.mjs ./next.config.mjs

# Handle build-time arguments
ARG NEXT_PUBLIC_AUTHORIZATION_HEADER
ENV NEXT_PUBLIC_AUTHORIZATION_HEADER=$NEXT_PUBLIC_AUTHORIZATION_HEADER
ARG NEXT_PUBLIC_ROOT_URI
ENV NEXT_PUBLIC_ROOT_URI=$NEXT_PUBLIC_ROOT_URI

# Build the Next.js app
RUN pnpm run build

# Stage 2: Run the Next.js application in production
FROM node:20.18-alpine AS runner

# Install pnpm
RUN npm install -g pnpm

# Set working directory
WORKDIR /app

# Copy built app and node_modules from the builder stage
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json
# Copy the public folder into the /app directory
COPY public /app/public

# Expose the port Next.js will run on
ENV PORT 8888
EXPOSE 8888

# Start the Next.js application in production mode
CMD ["pnpm", "run", "start"]