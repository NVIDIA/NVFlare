# Secure Federated Kaplan-Meier Analysis via Time-Binning and Homomorphic Encryption

This example illustrates two features:
* How to perform Kaplan-Meier survival analysis in federated setting without and with secure features via time-binning and Homomorphic Encryption (HE).
* How to use the Flare ModelController API to contract a workflow to facilitate HE under simulator mode.

## Secure Multi-party Kaplan-Meier Analysis
Kaplan-Meier survival analysis is a one-shot (non-iterative) analysis performed on a list of events and their corresponding time. In this example, we use [lifelines](https://zenodo.org/records/10456828) to perform this analysis. 

Essentially, the estimator needs to get access to the event list, and under the setting of federated analysis, the aggregated event list from all participants.

However, this poses a data security concern - by sharing the event list, the raw data can be exposed to external parties, which break the core value of federated analysis.

Therefore, we would like to design a secure mechanism to enable collaborative Kaplan-Meier analysis without the risk of exposing the raw information from a participant, the targeted protection includes:
- Prevent clients from getting RAW data from each other;
- Prevent the aggregation server to access ANY information from submissions.

This is achieved by two techniques:
- Condense the raw event list to two histograms (one for observed events and the other for censored event) using binning at certain interval (e.g. a week), such that events happened within the same bin from different participants can be aggregated and will not be distinguishable for the final aggregated histograms. Note that coarser binning will lead to higher protection, but also lower resolution of the final Kaplan-Meier curve.
- The local histograms will be encrypted as one single vector before sending to server, and the global aggregation operation at server side will be performed entirely within encryption space with HE. This will not cause any information loss, while the server will perform aggregation within encryption space.

With these two settings, the server will have no access to any knowledge regarding local submissions, and participants will only receive global aggregated histograms that will not contain distinguishable information regarding any individual participants (client number >= 3 - if only two participants, one can infer the other party's info by subtracting its own histograms).

The final Kaplan-Meier survival analysis will be performed locally on the global aggregated event list, recovered from decrypted global histograms.

## Baseline Kaplan-Meier Analysis
We first illustrate the baseline centralized Kaplan-Meier analysis without any secure features. We used veterans_lung_cancer dataset by
`from sksurv.datasets import load_veterans_lung_cancer`, and used `Status` as the event type and `Survival_in_days` as the event time to construct the event list.

To run the baseline script, simply execute:
```commandline
python utils/baseline_kaplan_meier.py
```
By default, this will generate a KM curve image `km_curve_baseline.png` under `/tmp` directory. The resutling KM curve is shown below:
![KM survival baseline](figs/km_curve_baseline.png)
Here, we show the survival curve for both daily (without binning) and weekly binning. The two curves aligns well with each other, while the weekly-binned curve has lower resolution.


## Federated Kaplan-Meier Analysis w/o and w/ HE 
We make use of FLARE ModelController API to implement the federated Kaplan-Meier analysis, both without and with HE.

The Flare ModelController API (`ModelController`) provides the functionality of flexible FLModel payloads for each round of federated analysis. This gives us the flexibility of transmitting various information needed by our scheme at different stages of federated learning.

Our [existing HE examples](https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/cifar10/cifar10-real-world) uses data filter mechanism for HE, provisioning the HE context information (specs and keys) for both client and server of the federated job under [CKKS](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/app_opt/he/model_encryptor.py) scheme. In this example, we would like to illustrate ModelController's capability in supporting customized needs beyond the existing HE functionalities (designed mainly for encrypting deep learning models).
- different HE schemes (BFV) rather than CKKS
- different content at different rounds of federated learning, and only specific payload needs to be encrypted

With the ModelController API, such "proof of concept" experiment becomes easy. In this example, the federated analysis pipeline includes 2 rounds without HE, or 3 rounds with HE.

For the federated analysis without HE, the detailed steps are as follows:
1. Server sends the simple start message without any payload.
2. Clients submit the local event histograms to server. Server aggregates the histograms with varying lengths by adding event counts of the same slot together, and sends the aggregated histograms back to clients.

For the federated analysis with HE, we need to ensure proper HE aggregation using BFV, and the detailed steps are as follows:
1. Server send the simple start message without any payload. 
2. Clients collect the information of the local maximum bin number (for event time) and send to server, where server aggregates the information by selecting the maximum among all clients. The global maximum number is then distributed back to clients. This step is necessary because we would like to standardize the histograms generated by all clients, such that they will have the exact same length and can be encrypted as vectors of same size, which will be addable.
3. Clients condense their local raw event lists into two histograms with the global length received, encrypt the histrogram value vectors, and send to server. Server aggregated the received histograms by adding the encrypted vectors together, and sends the aggregated histograms back to clients.

After these rounds, the federated work is completed. Then at each client, the aggregated histograms will be decrypted and converted back to an event list, and Kaplan-Meier analysis can be performed on the global information.

## Run the job
First, we prepared data for a 5-client federated job. We split and generate the data files for each client with binning interval of 7 days. 
```commandline
python utils/prepare_data.py --site_num 5 --bin_days 7 --out_path "/tmp/flare/dataset/km_data"
```

Then we prepare HE context for clients and server, note that this step is done by secure provisioning for real-life applications, but in this study experimenting with BFV scheme, we use this step to distribute the HE context. 
```commandline
python utils/prepare_he_context.py --out_path "/tmp/flare/he_context"
```

Next, we set the location of the job templates directory.
```commandline
nvflare config -jt ./job_templates
```

Then we can generate the job configurations from the `kaplan_meier` template:

Both for the federated job without HE:
```commandline
N_CLIENTS=5
nvflare job create -force -j "/tmp/flare/jobs/kaplan-meier" -w "kaplan_meier" -sd "./src" \
-f config_fed_client.conf app_script="kaplan_meier_train.py" app_config="--data_root /tmp/flare/dataset/km_data" \
-f config_fed_server.conf min_clients=${N_CLIENTS}
```
and for the federated job with HE:
```commandline
N_CLIENTS=5
nvflare job create -force -j "/tmp/flare/jobs/kaplan-meier-he" -w "kaplan_meier_he" -sd "./src" \
-f config_fed_client.conf app_script="kaplan_meier_train_he.py" app_config="--data_root /tmp/flare/dataset/km_data --he_context_path /tmp/flare/he_context/he_context_client.txt" \
-f config_fed_server.conf min_clients=${N_CLIENTS} he_context_path="/tmp/flare/he_context/he_context_server.txt"
```

And we can run the federated job:
```commandline
nvflare simulator -w /tmp/flare/workspace_km -n 5 -t 5 /tmp/flare/jobs/kaplan-meier
```
```commandline
nvflare simulator -w /tmp/flare/workspace_km_he -n 5 -t 5 /tmp/flare/jobs/kaplan-meier-he
```
By default, this will generate a KM curve image `km_curve_fl.png` and `km_curve_fl_he.png` under each client's directory.

## Display Result

By comparing the two curves, we can observe that all curves are identical:
![KM survival fl](figs/km_curve_fl.png)
![KM survival fl_he](figs/km_curve_fl_he.png)
