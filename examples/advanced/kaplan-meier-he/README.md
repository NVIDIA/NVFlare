# Secure Federated Kaplan-Meier Analysis via Time-Binning and Homomorphic Encryption

This example illustrates two features:
* How to perform Kaplan-Meier survival analysis in federated setting without and with secure features via time-binning and Homomorphic Encryption (HE).
* How to use the Recipe API with Flare ModelController for job configuration and execution in both simulation and production environments.

## Basics of Kaplan-Meier Analysis
Kaplan-Meier survival analysis is a non-parametric statistic used to estimate the survival function from lifetime data. It is used to analyze the time it takes for an event of interest to occur. For example, during a clinical trial, the Kaplan-Meier estimator can be used to estimate the proportion of patients who survive a certain amount of time after treatment. 

The Kaplan-Meier estimator takes into account the time of the event (e.g. "Survival Days") and whether the event was observed or censored. An event is observed if the event of interest (e.g. "death") occurred at the end of the observation process. An event is censored if the event of interest did not occur (i.e. patient is still alive) at the end of the observation process.

One example dataset used here for Kaplan-Meier analysis is the `veterans_lung_cancer` dataset. This dataset contains information about the survival time of veterans with advanced lung cancer. Below we provide some samples of the dataset:

| ID | Age | Celltype   | Karnofsky  | Diagtime | Prior | Treat     | Status | Survival Days |
|----|-----|------------|------------|----------|-------|-----------|--------|---------------|
| 1  | 64  | squamous   | 70         | 5        | yes   | standard  | TRUE   | 411           |
| 20 | 55  | smallcell  | 40         | 3        | no    | standard  | FALSE  | 123           |
| 45 | 61  | adeno      | 20         | 19       | yes   | standard  | TRUE   | 8             |
| 63 | 62  | large      | 90         | 2        | no    | standard  | FALSE  | 182           |

To perform the analysis, in this data, we have:
- Time `Survival Days`: days passed from the beginning of the observation till the end
- Event `Status`: whether event (i.e. death) happened at the end of the observation, or not
Based on the above understanding, we can interpret the data as follows:
- Patient #1 goes through an observation period of 411 days, and passes away at Day 411
- Patient #20 goes through an observation period of 123 days, and is still alive when the observation stops at Day 123 

The purpose of Kaplan-Meier analysis is to estimate the survival function, which is the probability that a patient survives beyond a certain time. Naturally, it will be a monotonic decreasing function, since the probability of surviving will decrease as time goes by.

## Secure Multi-party Kaplan-Meier Analysis
As described above, Kaplan-Meier survival analysis is a one-shot (non-iterative) analysis performed on a list of events (`Status`) and their corresponding time (`Survival Days`). In this example, we use [lifelines](https://zenodo.org/records/10456828) to perform this analysis. 

Essentially, the estimator needs to get access to this event list, and under the setting of federated analysis, the aggregated event list from all participants.

However, this poses a data security concern - the event list is equivalent to the raw data. If it gets exposed to external parties, it essentially breaks the core value of federated analysis.

Therefore, we would like to design a secure mechanism to enable collaborative Kaplan-Meier analysis without the risk of exposing the raw information from a participant, the targeted protection includes:
- Prevent clients from getting RAW data from each other;
- Prevent the aggregation server to access ANY information from participants' submissions.

This is achieved by two techniques:
- Condense the raw event list to two histograms (one for observed events and the other for censored event) using binning at certain interval (e.g. a week)
- Perform the aggregation of the histograms using Homomorphic Encryption (HE)

With time-binning, the above event list will be converted to histograms: if using a week as interval:
- Patient #1 will contribute 1 to the 411/7 = 58th bin of the observed event histogram
- Patient #20 will contribute 1 to the 123/7 = 17th bin of the censored event histogram

In this way, events happened within the same bin from different participants can be aggregated and will not be distinguishable for the final aggregated histograms. Note that coarser binning will lead to higher protection, but also lower resolution of the final Kaplan-Meier curve.

Local histograms will then be encrypted as one single vector before sending to server, and the global aggregation operation at server side will be performed entirely within encryption space with HE. This will not cause any information loss, while the server will not be able to access any plain-text information.

With these two settings, the server will have no access to any knowledge regarding local submissions, and participants will only receive global aggregated histograms that will not contain distinguishable information regarding any individual participants (client number >= 3 - if only two participants, one can infer the other party's info by subtracting its own histograms).

The final Kaplan-Meier survival analysis will be performed locally on the global aggregated event list, recovered from decrypted global histograms.

## Baseline Kaplan-Meier Analysis
We first illustrate the baseline centralized Kaplan-Meier analysis without any secure features. We used veterans_lung_cancer dataset by
`from sksurv.datasets import load_veterans_lung_cancer`, and used `Status` as the event type and `Survival_in_days` as the event time to construct the event list.

To run the baseline script, simply execute:
```commandline
python utils/baseline_kaplan_meier.py
```
By default, this will generate a KM curve image `km_curve_baseline.png` under `/tmp/nvflare/baseline` directory. The resulting KM curve is shown below:
![KM survival baseline](figs/km_curve_baseline.png)
Here, we show the survival curve for both daily (without binning) and weekly binning. The two curves aligns well with each other, while the weekly-binned curve has lower resolution.


## Federated Kaplan-Meier Analysis w/o and w/ HE 
We make use of FLARE ModelController API to implement the federated Kaplan-Meier analysis, both without and with HE.

The Flare ModelController API (`ModelController`) provides the functionality of flexible FLModel payloads for each round of federated analysis. This gives us the flexibility of transmitting various information needed by our scheme at different stages of federated learning.

Our [existing HE examples](../cifar10/cifar10-real-world) uses data filter mechanism for HE, provisioning the HE context information (specs and keys) for both client and server of the federated job under [CKKS](../../../nvflare/app_opt/he/model_encryptor.py) scheme. In this example, we would like to illustrate ModelController's capability in supporting customized needs beyond the existing HE functionalities (designed mainly for encrypting deep learning models):
- Different content at different rounds of federated learning, where only specific payloads need to be encrypted
- Flexibility in choosing what to encrypt (histograms) versus what to send in plain text (metadata)

With the ModelController API, such "proof of concept" experiment becomes easy. In this example, the federated analysis pipeline includes 2 rounds without HE, or 3 rounds with HE.

For the federated analysis without HE, the detailed steps are as follows:
1. Server sends the simple start message without any payload.
2. Clients submit the local event histograms to server. Server aggregates the histograms with varying lengths by adding event counts of the same slot together, and sends the aggregated histograms back to clients.

For the federated analysis with HE, we need to ensure proper HE aggregation using CKKS, and the detailed steps are as follows:
1. Server send the simple start message without any payload. 
2. Clients collect the information of the local maximum bin number (for event time) and send to server, where server aggregates the information by selecting the maximum among all clients. The global maximum number is then distributed back to clients. This step is necessary because we would like to standardize the histograms generated by all clients, such that they will have the exact same length and can be encrypted as vectors of same size, which will be addable.
3. Clients condense their local raw event lists into two histograms with the global length received, encrypt the histrogram value vectors, and send to server. Server aggregated the received histograms by adding the encrypted vectors together, and sends the aggregated histograms back to clients.

After these rounds, the federated work is completed. Then at each client, the aggregated histograms will be decrypted and converted back to an event list, and Kaplan-Meier analysis can be performed on the global information.

### HE Context and Data Management

- **Simulation Mode**: 
  - Uses **CKKS scheme** (approximate arithmetic, compatible with production)
  - HE context files are manually created via `prepare_he_context.py`:
    - Client context: `/tmp/nvflare/he_context/he_context_client.txt`
    - Server context: `/tmp/nvflare/he_context/he_context_server.txt`
  - Data prepared at `/tmp/nvflare/dataset/km_data`
  - Paths can be customized via `--he_context_path` (for client context) and `--data_root`
- **Production Mode**: 
  - Uses **CKKS scheme**
  - HE context is automatically provisioned into startup kits via `nvflare provision`
  - Context files are resolved by NVFlare's SecurityContentService:
    - Clients automatically use: `client_context.tenseal` (from their startup kit)
    - Server automatically uses: `server_context.tenseal` (from its startup kit)
  - The `--he_context_path` parameter is ignored in production mode
  - **Reuses the same data** from simulation mode at `/tmp/nvflare/dataset/km_data` by default

**Note:** CKKS scheme provides strong encryption with approximate arithmetic, which works well for this Kaplan-Meier analysis. The histogram counts are encrypted as floating-point numbers and rounded back to integers after decryption. Both simulation and production modes use the same CKKS scheme for consistency and compatibility. Production mode can reuse the data prepared during simulation mode, eliminating redundant data preparation.

## Run the job

This example supports both **Simulation Mode** (for local testing) and **Production Mode** (for real-world deployment).

| Feature | Simulation Mode | Production Mode |
|---------|----------------|-----------------|
| **Use Case** | Testing & Development | Real-world Deployment / Production Testing |
| **HE Context** | Manual preparation via script | Auto-provisioned via startup kits |
| **Security** | Single machine, no encryption between processes | Secure startup kits with certificates |
| **Setup** | Quick & simple | Requires provisioning & starting all parties |
| **Startup** | Single command | `start_all.sh` (local) or manual (distributed) |
| **Participants** | All run locally in one process | Distributed servers/clients running separately |
| **Data** | Prepared once, shared by all | Same data reused from simulation |

### Simulation Mode

For simulation mode (testing and development), we manually prepare the data and HE context:

**Step 1: Prepare Data**

Split and generate data files for each client with binning interval of 7 days:
```commandline
python utils/prepare_data.py --site_num 5 --bin_days 7 --out_path "/tmp/nvflare/dataset/km_data"
```

**Step 2: Prepare HE Context (Simulation Only)**

For simulation mode, manually prepare the HE context with CKKS scheme:
```commandline
# Remove old HE context if it exists
rm -rf /tmp/nvflare/he_context
# Generate new CKKS HE context
python utils/prepare_he_context.py --out_path "/tmp/nvflare/he_context"
```

This generates the HE context with CKKS scheme (poly_modulus_degree=8192, global_scale=2^40) compatible with production mode.

**Step 3: Run the Job**

Run the job without and with HE:
```commandline
python job.py
python job.py --encryption
```

The script will execute the job in simulation mode and display the job status. Results (KM curves and analysis details) will be saved to each simulated client's workspace directory under `/tmp/nvflare/workspaces/`.

### Production Mode

For production deployments, the HE context is automatically provisioned through secure startup kits.

**Quick Start for Local Testing:**
If you want to quickly test production mode on a single machine:
1. Run provisioning: `nvflare provision -p project.yml -w /tmp/nvflare/prod_workspaces`
2. Start all parties: `./start_all.sh`
3. Start admin console: `cd /tmp/nvflare/prod_workspaces/km_he_project/prod_00/admin@nvidia.com && ./startup/fl_admin.sh` (use username `admin@nvidia.com`)
4. Submit job: `python job.py --encryption --startup_kit_location /tmp/nvflare/prod_workspaces/km_he_project/prod_00/admin@nvidia.com`
5. Monitor job via admin console: `list_jobs`, `check_status client`, `download_job <job_id>`
6. Shutdown: `shutdown all` in admin console

For detailed steps and distributed deployment, continue below:

**Step 1: Install NVFlare with HE Support**

```commandline
pip install nvflare[HE]
```

**Step 2: Provision Startup Kits with HE Context**

The `project.yml` file in this directory is pre-configured with `HEBuilder` using the CKKS scheme. Run provisioning to output to `/tmp/nvflare/prod_workspaces`:

```commandline
nvflare provision -p project.yml -w /tmp/nvflare/prod_workspaces
```

This generates startup kits in `/tmp/nvflare/prod_workspaces/km_he_project/prod_00/`:
- `localhost/` - Server startup kit with `server_context.tenseal`
- `site-1/`, `site-2/`, etc. - Client startup kits, each with `client_context.tenseal`
- `admin@nvidia.com/` - Admin console

The HE context files are automatically included in each startup kit and do not need to be manually distributed.

**Step 3: Distribute Startup Kits**

Securely distribute the startup kits to each participant from `/tmp/nvflare/prod_workspaces/km_he_project/prod_00/`:
- `localhost/` directory is the server (for local testing, no need to send)
- Send `site-1/`, `site-2/`, etc. directories to each client host (for distributed deployment)
- Keep `admin@nvidia.com/` directory for the admin user

**Step 4: Start All Parties**

**Option A: Quick Start (Local Testing)**

For local testing where all parties run on the same machine, use the convenience script:

```commandline
./start_all.sh
```

This will start the server and all 5 clients in the background. Logs are saved to `/tmp/nvflare/logs/`.

Then start the admin console:
```commandline
cd /tmp/nvflare/prod_workspaces/km_he_project/prod_00/admin@nvidia.com
./startup/fl_admin.sh
```

**Important:** When prompted for "User Name:", enter `admin@nvidia.com` (this matches the admin defined in project.yml).

Once connected, check the status of all participants:
```
> check_status server
> check_status client
```

**Option B: Manual Start (Distributed Deployment)**

For distributed deployment where parties run on different machines:

**On the Server Host:**
```commandline
cd /tmp/nvflare/prod_workspaces/km_he_project/prod_00/localhost
./startup/start.sh
```

Wait for the server to be ready (you should see "Server started" in the logs).

**On Each Client Host:**
```commandline
# On site-1
cd /tmp/nvflare/prod_workspaces/km_he_project/prod_00/site-1
./startup/start.sh

# On site-2
cd /tmp/nvflare/prod_workspaces/km_he_project/prod_00/site-2
./startup/start.sh

# Repeat for site-3, site-4, and site-5
```

**On the Admin Machine:**
```commandline
cd /tmp/nvflare/prod_workspaces/km_he_project/prod_00/admin@nvidia.com
./startup/fl_admin.sh
# When prompted, use username: admin@nvidia.com
```

**Step 5: Submit and Run the Job**

With all parties running, submit the job using the Recipe API. The job will automatically use:
- The provisioned HE context from each participant's startup kit
- The data already prepared in simulation mode at `/tmp/nvflare/dataset/km_data`

```commandline
python job.py --encryption --startup_kit_location /tmp/nvflare/prod_workspaces/km_he_project/prod_00/admin@nvidia.com
```

The script will output the job status. Note the job ID from the output.

**Monitoring Job Progress:**

The job runs asynchronously on the FL system. Use the admin console to monitor progress:

```commandline
# In the admin console
> list_jobs                    # View all jobs
> check_status server          # Check server status
> check_status client          # Check all clients status
> download_job <job_id>        # Download results after completion
```

Results will be saved to each client's workspace directory after the job completes:
- `/tmp/nvflare/prod_workspaces/km_he_project/prod_00/site-1/{JOB_ID}/`
- Look for `km_curve_fl_he.png` and `km_global.json` in each client's job directory

**Note:** In production mode with HE, the HE context paths are automatically configured to use the provisioned context files from each participant's startup kit:
- Clients use: `client_context.tenseal` 
- Server uses: `server_context.tenseal`

The `--he_context_path` parameter is only used for simulation mode and is ignored in production mode. No manual HE context distribution is needed in production.

**Step 6: Shutdown All Parties**

After the job completes, shut down all parties gracefully via admin console:

```
> shutdown all
```

## Display Result

By comparing the two curves, we can observe that all curves are identical:
![KM survival fl](figs/km_curve_fl.png)
![KM survival fl_he](figs/km_curve_fl_he.png)
