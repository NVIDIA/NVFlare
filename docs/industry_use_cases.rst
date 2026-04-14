.. _industry_use_cases:

####################
Industry Use Cases
####################

Federated Learning is being adopted across industries where data cannot be centralized due to
privacy, regulatory, or competitive constraints. NVIDIA FLARE provides the platform infrastructure
to enable these deployments.

Healthcare & Life Sciences
==========================

Federated learning enables multi-hospital collaboration for medical AI model development
without sharing patient data.

**Key applications:**

- Medical image analysis (radiology, pathology, ophthalmology)
- Drug discovery and molecular property prediction
- Electronic health records (EHR) analysis
- Clinical trial optimization
- Genomics and precision medicine

**Why federated?** HIPAA, patient consent, and institutional data governance policies
prevent centralized data aggregation. Federated learning allows hospitals to train
better models together while keeping patient data local.

**Cancer AI Alliance (CAIA):**
The Cancer AI Alliance -- a consortium of leading cancer centers -- uses NVIDIA FLARE
with Rhino Federated Computing Platform to train AI models across multiple institutions
while keeping sensitive patient data behind each center's firewall. Only model weights
are exchanged; cancer centers retain full control over data selection, access policies,
and local execution. The platform supports differential privacy and model encryption for projects requiring enhanced protection.

- `How CAIA operationalizes secure, multi-site research with federated learning (Dec 2025) <https://www.canceralliance.ai/blog/caia-multi-site-research-federated-learning>`_
- `Federating cancer research at scale (FLARE Day 2025) <https://www.nvidia.com/en-us/on-demand/session/nvidiaflareday25-nvfd08?playlistId=playList-eacb3be4-9f4b-48d0-98fc-f7a40f93d759>`_

**Eli Lilly TuneLab -- Federated Drug Discovery:**
In September 2025, Eli Lilly launched TuneLab, an AI/ML platform that gives biotech
companies access to drug discovery models trained on over $1 billion of Lilly's research
data. The platform uses federated learning so that biotech partners can fine-tune Lilly's
models on their own proprietary molecular data without exposing it. In return, partners
contribute training data that continuously improves the shared models for the entire ecosystem.

- `Lilly launches TuneLab platform (Sep 2025) <https://investor.lilly.com/news-releases/news-release-details/lilly-launches-tunelab-platform-give-biotechnology-companies>`_

**Federated AI for Therapeutic Engineering (FAITE) -- AbbVie, Amgen, AstraZeneca, J&J, UCB:**
Launched in 2025, FAITE is a cross-industry biopharmaceutical consortium that uses federated
and active learning to train models for predicting biologics properties. Member companies
contribute training on local proprietary molecular data without sharing it, enabling collaborative
model improvement while maintaining competitive and regulatory data boundaries.

- `FAITE: Federated AI for biologics property prediction (FLARE Day 2025) <https://www.nvidia.com/en-us/on-demand/session/nvidiaflareday25-nvfd13?playlistId=playList-eacb3be4-9f4b-48d0-98fc-f7a40f93d759>`_
- `Training federated AI models to predict protein properties (NVIDIA Blog) <https://developer.nvidia.com/blog/training-federated-ai-models-to-predict-protein-properties/>`_

**Other references:**

- `Federated Learning for Brain Tumor Segmentation (Nature Communications) <https://doi.org/10.1038/s41467-022-33407-5>`_
-  see more use cases in FLARE DAY recordings

Financial Services
==================

Financial institutions use federated learning for fraud detection, credit risk modeling,
and anti-money laundering without exposing sensitive transaction data.

**Key applications:**

- Fraud detection across payment networks
- Credit scoring with broader data representations
- Anti-money laundering (AML) model training
- Market risk analysis

**Why federated?** Banking regulations (SOX, FINRA, GDPR, PSD2) and competitive
sensitivity prevent sharing transaction data between institutions.

**Swift Collaborative Fraud Defence:**
In September 2025, Swift partnered with 13 global banks -- including ANZ, BNY, and Intesa
Sanpaolo -- to test federated learning for cross-border fraud detection. Using privacy-enhancing
technologies, participating institutions trained AI models locally on their own data without
sharing customer information. In trials involving 10 million artificial transactions, the
collaborative federated model was **twice as effective** at detecting known fraudulent transactions
compared to models trained on a single institution's data alone.

**Reference:**

- `Swift-led experiments reveal blueprint for collaborative fraud defence using AI (Sep 2025) <https://www.swift.com/news-events/news/swift-led-experiments-reveal-blueprint-collaborative-fraud-defence-using-ai>`_
- `Federated fraud detection at Swift (FLARE Day 2025) <https://www.nvidia.com/en-us/on-demand/session/nvidiaflareday25-nvfd09?playlistId=playList-eacb3be4-9f4b-48d0-98fc-f7a40f93d759>`_

**JP Morgan, BNY, and RBC -- Federated Financial AI:**
At GTC 2025, JP Morgan, BNY, and Royal Bank of Canada (RBC) presented their experiences applying
federated learning to financial AI models, covering cross-institutional model training for risk
and fraud use cases without sharing sensitive customer transaction data.

- `Federated Learning in Financial Services: JP Morgan, BNY, RBC (GTC 2025) <https://www.nvidia.com/en-us/on-demand/session/gtcdc25-dc51038/?playlistId=playList-fd045586-2409-4d1a-8333-e1d3501d52de>`_

Government & National Security
==============================

Government agencies and national laboratories use federated learning to collaboratively
train AI models on sensitive, geographically distributed datasets that cannot be centralized
due to classification, data sovereignty, or security constraints.

**Trilab Federated AI (Sandia, Los Alamos, Lawrence Livermore):**
In 2025, the three NNSA national security laboratories demonstrated a federated-learning
prototype -- codenamed *Chandler* -- that trains a shared large language model across
three geographically distributed classified systems without exchanging raw data.
Using NVIDIA FLARE to orchestrate training, the labs exchange only model weights (parameters)
between epochs while keeping each laboratory's unique datasets local. The prototype ran
on both NVIDIA and AMD GPU hardware, including Lawrence Livermore's El Capitan, the
world's fastest supercomputer.

*"Federated training is a critical tool to delivering a robust capability in a cost
effective, performant and secure way."* -- Si Hammond, NNSA Office of Advanced Simulation
and Computing

**Reference:**

- `Three national security laboratories, one AI model (Sandia Lab News, Dec 2025) <https://www.sandia.gov/labnews/2025/12/18/three-national-security-laboratories-one-ai-model/>`_
- `Trilab federated LLM training across classified systems (FLARE Day 2025) <https://www.nvidia.com/en-us/on-demand/session/nvidiaflareday25-nvfd18?playlistId=playList-eacb3be4-9f4b-48d0-98fc-f7a40f93d759>`_

**Oak Ridge National Laboratory (ORNL) -- OLCF Scientific Research:**
The Oak Ridge Leadership Computing Facility (OLCF) uses NVIDIA FLARE for federated learning
across distributed scientific datasets, enabling multi-site collaboration on large-scale
scientific computing research without centralizing sensitive experimental data.

- `ORNL OLCF: Federated learning for scientific research (FLARE Day 2024) <https://developer.download.nvidia.com/assets/Clara/flare/NVFLARE_DAY_2024_Part_09_ORNL.mp4>`_

**Taiwan International Federated Learning Center (Ministry of Health and Welfare):**
In January 2026, Taiwan's Ministry of Health and Welfare announced the establishment of an
International High-Computing and Federated Learning Center for training smart medical AI
models while preserving data privacy and sovereignty. The center completed proof of concept
with 16 major hospitals and plans to scale to 100 regional hospitals and ultimately all
Taiwanese hospitals. Medical data remains on local hospital servers while the central AI
model learns from distributed data through federated learning. The center is also facilitating
international collaboration with Thailand's Mahidol University to jointly develop standards
for AI-based medical product verification across ASEAN markets.

**Reference:**

- `Taiwan launches new era of medical AI with global collaboration (Jan 2026) <https://www.rti.org.tw/en/news?uid=3&pid=188971>`_
- `Taiwan International Federated Learning Center (FLARE Day 2025) <https://www.nvidia.com/en-us/on-demand/session/nvidiaflareday25-nvfd23?playlistId=playList-4dd15c3d-2422-425c-b9d9-d21694397574>`_


Transportation
==============

**Federated Learning for Autonomous Vehicles:**
Automotive manufacturers and research teams use federated learning to train perception and
safety models across distributed vehicle fleets and test facilities, improving model quality
without centralizing proprietary driving data.

- `Federated Learning for Autonomous Vehicles (FLARE Day 2024) <https://developer.download.nvidia.com/assets/Clara/flare/NVFLARE_DAY_2024_Part_02_AV.mp4>`_


Edge AI & Scientific Computing
==============================

**NVIDIA Holoscan Federated Analytics at the Edge:**
NVIDIA Holoscan enables federated analytics at the edge for medical devices and industrial
AI applications, allowing inference data to be used for federated model improvement without
transmitting raw sensor streams to a central server.

- `Holoscan Federated Analytics at the Edge (FLARE Day 2025) <https://www.nvidia.com/en-us/on-demand/session/nvidiaflareday25-nvfd26?playlistId=playList-4dd15c3d-2422-425c-b9d9-d21694397574>`_

**NVIDIA Data Federation Mesh -- Federated Data Processing in Scientific Computing:**
The NVIDIA Data Federation Mesh demonstrates federated data processing pipelines for
large-scale scientific computing, enabling distributed analysis across facilities without
moving raw scientific datasets.

- `NVIDIA Data Federation Mesh (FLARE Day 2025) <https://www.nvidia.com/en-us/on-demand/session/nvidiaflareday25-nvfd15?playlistId=playList-eacb3be4-9f4b-48d0-98fc-f7a40f93d759>`_


FLARE Day -- Real-World Deployments
====================================

FLARE Day is an annual event showcasing real-world federated learning deployments across
healthcare, finance, autonomous driving, and more. These talks feature practitioners sharing
production experiences and lessons learned.

- **FLARE Day 2026** -- *Coming September 2026*
- `FLARE Day 2025 <https://developer.nvidia.com/flare-day-2025>`_ -- Real-world FL applications in healthcare, finance, autonomous driving, and more
- `FLARE Day 2024 <https://nvidia.github.io/NVFlare/flareDay>`_ -- Talks and demos featuring real-world FL deployments at NVIDIA, healthcare institutions, and industry partners

Getting Started with Your Industry
====================================

Regardless of your industry, the path to federated learning follows a similar pattern:

1. **Identify the use case** -- What ML model do you want to improve with federated data?
2. **Start with simulation** -- Use the :ref:`FL Simulator <fl_simulator>` to prototype with synthetic data
3. **Prove value with POC** -- Run a :ref:`POC deployment <poc_command>` with 2-3 participating sites
4. **Scale to production** -- Follow the :doc:`Deployment Guide <user_guide/admin_guide/deployment/overview>` for provisioning and infrastructure

For questions about industry-specific deployments, see the :doc:`publications_and_talks` page
for talks and papers relevant to your domain.
