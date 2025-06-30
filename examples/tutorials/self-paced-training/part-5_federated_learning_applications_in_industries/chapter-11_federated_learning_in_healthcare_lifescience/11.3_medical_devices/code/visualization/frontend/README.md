This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

This is the project for Holoscan Federated Analytics Dashboard.

## Getting Started

First run the Analytics backend as per the instructions provided in `holoscan-federated-analytics/visualization/backend/README.md`. This provides a necessary rest API service required for the analytics dashboard frontend.

Run the analytics dashboard with the following command:
```bash
cd visualization/frontend/analytics-dashboard

# modify `NEXT_PUBLIC_ROOT_URI` variable in the `analytics-dashboard/.env` file to point to the IP where analytics backend is running.

# Modify `NEXT_PUBLIC_AUTHORIZATION_HEADER` variable in the `analytics-dashboard/.env` file to point to the newly created test JWT token as per the steps mentioned in `holoscan-federated-analytics/visualization/backend/README.md`.

# Build the frontend docker container
docker compose build

# Run the frontend container - it will start webserver on the local host and port 8888.
docker compose run
```

Open [http://ip:8888](http://ip:8888) with your browser to see the result.

Dashboard currently supports:
- A page listing all the analytics enabled applications.
- Statistics page supports:
  - Visualization of global latest statistics.
  - Visualization of datewise statistics.
  - Visualization of accumulated statistics for the range of dates.
  - Visualization of latest and datewise statistics for a particular hirrarchical level.
  - Visualization of histogram for a particular feature.
- An about page with the high level details of Holoscan Federated Analytics.
