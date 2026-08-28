# Retinal Cohort Data Download Guide

This guide lists the public download pages used to assemble the 39-site
retinal/fundus cohort for the AgenticFL research example. It is a convenience
index, not a redistribution of any dataset. Review and accept each source's
current license, terms, citation requirements, and access restrictions before
downloading or using its contents.

The cohort uses 14 Kaggle datasets directly, 24 Kaggle counterpart or mirror
datasets, and one locally prepared RIGA source. A counterpart is not guaranteed
to be byte-for-byte identical to the original release. Verify its provenance,
record counts, labels, annotations, and task suitability before assigning it to
a client.

## Downloading from Kaggle

Install and authenticate the official [Kaggle CLI](https://github.com/Kaggle/kaggle-cli):

```bash
python -m pip install kaggle
kaggle auth login
```

For any Kaggle reference in the table, inspect its files and download it from
the NVFlare repository root as follows:

```bash
kaggle datasets files OWNER/DATASET
kaggle datasets download OWNER/DATASET \
  --path data/SITE_ID \
  --unzip
```

Kaggle also supports token-based authentication. See the official
[authentication reference](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/auth.md)
and [dataset command reference](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md).
Do not commit downloaded data, Kaggle credentials, or accepted-license records
to the NVFlare repository.

## Cohort Sources

`Kaggle source` identifies the Kaggle listing used directly by the cohort.
`Kaggle counterpart` identifies a third-party mirror or task-equivalent copy
used when the original host was unavailable or required a separate access
workflow.

| Site ID | Access route | Download page or Kaggle reference |
|---|---|---|
| `DRISHTI_GS` | Kaggle counterpart | [lokeshsaipureddi/drishtigs-retina-dataset-for-onh-segmentation](https://www.kaggle.com/datasets/lokeshsaipureddi/drishtigs-retina-dataset-for-onh-segmentation) |
| `STARE` | Kaggle source | [vidheeshnacode/stare-dataset](https://www.kaggle.com/datasets/vidheeshnacode/stare-dataset) |
| `IDRID` | Kaggle source | [aaryapatel98/indian-diabetic-retinopathy-image-dataset](https://www.kaggle.com/datasets/aaryapatel98/indian-diabetic-retinopathy-image-dataset) |
| `RIMONE` | Kaggle source | [orvile/rim-one-retinal-dataset-for-assessing-glaucoma](https://www.kaggle.com/datasets/orvile/rim-one-retinal-dataset-for-assessing-glaucoma) |
| `REFUGE` | Kaggle source | [victorlemosml/refuge2](https://www.kaggle.com/datasets/victorlemosml/refuge2) |
| `CHASE_DB1` | Kaggle counterpart | [buffyhridoy/chase-db1](https://www.kaggle.com/datasets/buffyhridoy/chase-db1) |
| `E_OPHTHA` | Kaggle counterpart | [samriddhibagchi/e-ophtha-diabetic-retinopathy-datasets-ex-ma](https://www.kaggle.com/datasets/samriddhibagchi/e-ophtha-diabetic-retinopathy-datasets-ex-ma) |
| `LES_AV` | Kaggle counterpart | [alfikiafan/retina-av-dataset](https://www.kaggle.com/datasets/alfikiafan/retina-av-dataset) |
| `RIGA` | [Official source](https://deepblue.lib.umich.edu/data/concern/data_sets/3b591905z) | [RIGA dataset on Deep Blue Data](https://deepblue.lib.umich.edu/data/concern/data_sets/3b591905z) |
| `MESSIDOR` | Kaggle counterpart | [parikshakaur/messidor](https://www.kaggle.com/datasets/parikshakaur/messidor) |
| `DRIVE` | Kaggle counterpart | [andrewmvd/drive-digital-retinal-images-for-vessel-extraction](https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction) |
| `ORIGA` | Kaggle source | [arnavjain1/glaucoma-datasets](https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets) |
| `ODIR_5K` | Kaggle source | [andrewmvd/ocular-disease-recognition-odir5k](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k) |
| `RFMID` | Kaggle source | [andrewmvd/retinal-disease-classification](https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification) |
| `MESSIDOR_2_DF` | Kaggle counterpart | [mariaherrerot/messidor2preprocess](https://www.kaggle.com/datasets/mariaherrerot/messidor2preprocess) |
| `RETINA_BLOOD_VESSEL_SEGMENTATION_DATASET` | Kaggle source | [abdallahwagih/retina-blood-vessel](https://www.kaggle.com/datasets/abdallahwagih/retina-blood-vessel) |
| `DDR_DATASET` | Kaggle counterpart | [samriddhibagchi/ddr-dataset-credits-to-authors](https://www.kaggle.com/datasets/samriddhibagchi/ddr-dataset-credits-to-authors) |
| `HYPERTENSIVE_RETINOPATHY` | Kaggle counterpart | [harshwardhanfartale/hypertension-and-hypertensive-retinopathy-dataset](https://www.kaggle.com/datasets/harshwardhanfartale/hypertension-and-hypertensive-retinopathy-dataset) |
| `SUSTECH_PLUS_SYSU_DATASET` | Kaggle counterpart | [mariaherrerot/the-sustechsysu-dataset](https://www.kaggle.com/datasets/mariaherrerot/the-sustechsysu-dataset) |
| `RITE` | Kaggle counterpart | [priyanagda/ritedataset](https://www.kaggle.com/datasets/priyanagda/ritedataset) |
| `CLAHE_PLUS_ESRGAN_SPLIT_FD` | Kaggle counterpart | [ahmetselukkren/clahe-esrgan-split-fundus-dataset](https://www.kaggle.com/datasets/ahmetselukkren/clahe-esrgan-split-fundus-dataset) |
| `RETINA_FUNDUS_DATASET_CHASE_DB1_DRIVE` | Kaggle counterpart | [ipythonx/retinal-vessel-segmentation](https://www.kaggle.com/datasets/ipythonx/retinal-vessel-segmentation) |
| `CATARACT_CLASSIFICATION_DATASET` | Kaggle source | [gunavenkatdoddi/eye-diseases-classification](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification) |
| `MURED` | Kaggle counterpart | [abhirampolisetti/multi-label-retinal-disease-mured-dataset](https://www.kaggle.com/datasets/abhirampolisetti/multi-label-retinal-disease-mured-dataset) |
| `ROFT` | Kaggle source | [sureshrasappan/retinal-and-ocular-fundus-images-for-diagnosis](https://www.kaggle.com/datasets/sureshrasappan/retinal-and-ocular-fundus-images-for-diagnosis) |
| `EYE_DISEASE_IMAGE_DATASET` | Kaggle counterpart | [mahin661/eye-disease-classification-fundus-image-dataset](https://www.kaggle.com/datasets/mahin661/eye-disease-classification-fundus-image-dataset) |
| `FIVES` | Kaggle counterpart | [amusaabdulahitomisin/fives-new-data](https://www.kaggle.com/datasets/amusaabdulahitomisin/fives-new-data) |
| `AMDP_DATASET` | Kaggle counterpart | [datasetengineer/amdp-dataset](https://www.kaggle.com/datasets/datasetengineer/amdp-dataset) |
| `SMDG` | Kaggle source | [deathtrooper/multichannel-glaucoma-benchmark-dataset](https://www.kaggle.com/datasets/deathtrooper/multichannel-glaucoma-benchmark-dataset) |
| `OCULAR_TOXOPLASMOSIS_DATASET` | Kaggle counterpart | [nafin59/ocular-toxoplasmosis-fundus-images-dataset](https://www.kaggle.com/datasets/nafin59/ocular-toxoplasmosis-fundus-images-dataset) |
| `ONH_SEGMENTATION_DATASET` | Kaggle counterpart | [lucascunhadecarvalho/drionsdb-cunha](https://www.kaggle.com/datasets/lucascunhadecarvalho/drionsdb-cunha) |
| `DRHAGIS_DATASET` | Kaggle counterpart | [swincs/seg-hagis](https://www.kaggle.com/datasets/swincs/seg-hagis) |
| `CATTLE_RETINAL_FUNDUS_IMAGES` | Kaggle source | [animalbiometry/cattle-retinal-fundus-images](https://www.kaggle.com/datasets/animalbiometry/cattle-retinal-fundus-images) |
| `PREPROCESSED_EYE_DISEASES_FUNDUS_IMAGES` | Kaggle source | [gunavenkatdoddi/preprocessed-eye-diseases-fundus-images](https://www.kaggle.com/datasets/gunavenkatdoddi/preprocessed-eye-diseases-fundus-images) |
| `RETINA_FUNDUS_IMAGE_REGISTRATION_DATASET_FIRE` | Kaggle counterpart | [andrewmvd/fundus-image-registration](https://www.kaggle.com/datasets/andrewmvd/fundus-image-registration) |
| `1000_FUNDUS_IMAGES_WITH_39_CATEGORIES` | Kaggle source | [linchundan/fundusimage1000](https://www.kaggle.com/datasets/linchundan/fundusimage1000) |
| `PAPILA_RETINAL_FUNDUS_IMAGES_DATASET` | Kaggle counterpart | [orvile/papila-retinal-fundus-images](https://www.kaggle.com/datasets/orvile/papila-retinal-fundus-images) |
| `DIARETDB1` | Kaggle counterpart | [nguyenhung1903/diaretdb1-standard-diabetic-retinopathy-database](https://www.kaggle.com/datasets/nguyenhung1903/diaretdb1-standard-diabetic-retinopathy-database) |
| `AIROGS` | Kaggle counterpart | [deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2](https://www.kaggle.com/datasets/deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2) |

## Preparation Notes

- Keep one client-local directory per site ID and point its `data_path` in
  `meta/site-meta.json` at that directory. Paths can be absolute or relative to
  the project root supplied to `job_data.py`.
- Treat the table as a source index, not a claim that every archive is already
  in AgenticFL's canonical training form. AgenticFL's client-local adapter owns
  inspection and preparation of each site's records.
- For RIGA, the experiment used a locally prepared `Image/`, `Disc/`, and
  `Cup/` layout derived from the official source. Do not silently substitute a
  differently paired mirror.
- The recorded DRHAGIS counterpart may contain rendered or blended annotation
  overlays rather than clean raw fundus inputs. It is not training-ready unless
  clean source images can be established locally; the raw-image guardrail must
  fail closed otherwise.
- Some Kaggle counterparts are preprocessed or partial variants. In particular,
  verify label completeness and image/annotation pairing before using them as a
  replacement for an original challenge or institutional release.

After preparing the site directories, create `meta/site-meta.json` from
`meta/site-meta.example.json`, then use existing prepared records to build the
local reference bundle as described in the main README.

## Cohort Identification

This table corresponds to the 39-site retinal/fundus cohort used for the
AgenticFL snapshot identified as `data_retinal`, derived from the
[FedAgentBench dataset catalog](https://arxiv.org/abs/2509.23803).
