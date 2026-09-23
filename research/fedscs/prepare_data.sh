#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"

CIFAR_DIR="${DATA_DIR}/cifar-10-batches-py"
ARCHIVE="${DATA_DIR}/cifar-10-python.tar.gz"
URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

CLIENT_DIR="${DATA_DIR}/clients"
TEST_FILE="${DATA_DIR}/test.pt"

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

NUM_CLIENTS=5

# Only Client 5 is corrupted.
NOISY_CLIENT=5

# 80% of Client 5's samples receive salt-and-pepper noise.
NOISY_FRACTION=0.80

# Exact salt-and-pepper parameters from the reference implementation.
SALT_PROB=0.30
PEPPER_PROB=0.30

SEED=42

# MD5 checksum published by the CIFAR-10 dataset authors.
EXPECTED_MD5="c58f30108f718f92721af3b95e74349a"

REQUIRED_FILES=(
    "data_batch_1"
    "data_batch_2"
    "data_batch_3"
    "data_batch_4"
    "data_batch_5"
    "test_batch"
    "batches.meta"
)

TEMP_ARCHIVE="${DATA_DIR}/.cifar-10-python.tar.gz.tmp"
TEMP_DIR="${DATA_DIR}/.cifar-10-extract.tmp"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

cleanup() {
    rm -f "${TEMP_ARCHIVE}"
    rm -rf "${TEMP_DIR}"
}

trap cleanup EXIT

# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------

validate_dataset() {
    local dataset_dir="$1"

    if [ ! -d "${dataset_dir}" ]; then
        echo "Dataset directory does not exist: ${dataset_dir}"
        return 1
    fi

    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "${dataset_dir}/${file}" ]; then
            echo "Missing required CIFAR-10 file: ${file}"
            return 1
        fi

        if [ ! -s "${dataset_dir}/${file}" ]; then
            echo "Required CIFAR-10 file is empty: ${file}"
            return 1
        fi
    done

    return 0
}

verify_archive() {
    local archive="$1"
    local checksum

    if [ ! -f "${archive}" ]; then
        echo "Archive does not exist: ${archive}"
        return 1
    fi

    if [ ! -s "${archive}" ]; then
        echo "Archive is empty: ${archive}"
        return 1
    fi

    checksum="$(md5sum "${archive}" | awk '{print $1}')"

    if [ "${checksum}" != "${EXPECTED_MD5}" ]; then
        echo "CIFAR-10 archive checksum verification failed."
        echo "Expected MD5: ${EXPECTED_MD5}"
        echo "Actual MD5:   ${checksum}"
        return 1
    fi

    return 0
}

# ---------------------------------------------------------------------------
# Download and validate CIFAR-10
# ---------------------------------------------------------------------------

mkdir -p "${DATA_DIR}"

if [ -f "${ARCHIVE}" ] && validate_dataset "${CIFAR_DIR}"; then
    echo "Verifying existing CIFAR-10 archive..."

    if verify_archive "${ARCHIVE}"; then
        echo "CIFAR-10 dataset already prepared."
    else
        echo "Existing CIFAR-10 archive is invalid. Re-downloading..."

        rm -f "${TEMP_ARCHIVE}"

        curl \
            --fail \
            --location \
            --retry 3 \
            --retry-delay 2 \
            --output "${TEMP_ARCHIVE}" \
            "${URL}"

        verify_archive "${TEMP_ARCHIVE}"

        mv "${TEMP_ARCHIVE}" "${ARCHIVE}"

        echo "CIFAR-10 archive replaced successfully."
    fi
else
    echo "Downloading CIFAR-10 dataset..."

    rm -f "${TEMP_ARCHIVE}"

    curl \
        --fail \
        --location \
        --retry 3 \
        --retry-delay 2 \
        --output "${TEMP_ARCHIVE}" \
        "${URL}"

    echo "Verifying CIFAR-10 archive..."

    if ! verify_archive "${TEMP_ARCHIVE}"; then
        echo "Downloaded CIFAR-10 archive is invalid."
        exit 1
    fi

    echo "CIFAR-10 archive checksum verified."

    rm -rf "${TEMP_DIR}"
    mkdir -p "${TEMP_DIR}"

    echo "Extracting CIFAR-10 dataset..."

    tar -xzf "${TEMP_ARCHIVE}" -C "${TEMP_DIR}"

    EXTRACTED_DATASET_DIR="${TEMP_DIR}/cifar-10-batches-py"

    echo "Validating extracted CIFAR-10 dataset..."

    if ! validate_dataset "${EXTRACTED_DATASET_DIR}"; then
        echo "CIFAR-10 dataset validation failed."
        exit 1
    fi

    echo "CIFAR-10 dataset validation successful."

    # Replace the final dataset only after successful validation.
    rm -rf "${CIFAR_DIR}"
    mv "${EXTRACTED_DATASET_DIR}" "${CIFAR_DIR}"

    # Replace the final archive only after checksum verification.
    mv "${TEMP_ARCHIVE}" "${ARCHIVE}"

    echo "CIFAR-10 dataset prepared successfully."
fi

# ---------------------------------------------------------------------------
# Create IID federated client datasets
# ---------------------------------------------------------------------------

echo ""
echo "Creating FedSCS federated dataset..."
echo "  Number of clients:       ${NUM_CLIENTS}"
echo "  Partition:               IID"
echo "  Noisy client:            Client ${NOISY_CLIENT}"
echo "  Noisy sample fraction:   ${NOISY_FRACTION}"
echo "  Salt probability:        ${SALT_PROB}"
echo "  Pepper probability:      ${PEPPER_PROB}"
echo "  Random seed:             ${SEED}"
echo ""

rm -rf "${CLIENT_DIR}"
mkdir -p "${CLIENT_DIR}"

python - \
    "${CIFAR_DIR}" \
    "${CLIENT_DIR}" \
    "${TEST_FILE}" \
    "${NUM_CLIENTS}" \
    "${NOISY_CLIENT}" \
    "${NOISY_FRACTION}" \
    "${SALT_PROB}" \
    "${PEPPER_PROB}" \
    "${SEED}" <<'PY'
import os
import pickle
import sys

import numpy as np
import torch


dataset_dir = sys.argv[1]
client_dir = sys.argv[2]
test_file = sys.argv[3]

num_clients = int(sys.argv[4])
noisy_client = int(sys.argv[5])
noisy_fraction = float(sys.argv[6])
salt_prob = float(sys.argv[7])
pepper_prob = float(sys.argv[8])
seed = int(sys.argv[9])


def load_cifar_batch(path):
    """Load one CIFAR-10 Python batch."""
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")

    data = batch[b"data"]
    labels = np.asarray(batch[b"labels"], dtype=np.int64)

    # CIFAR-10 stores images as:
    #   (N, 3072) = 3 x 32 x 32
    data = data.reshape(-1, 3, 32, 32)

    return data, labels


def add_salt_pepper_noise(
    img_tensor,
    salt_prob,
    pepper_prob,
    rng,
):
    """
    Add salt-and-pepper noise using the reference implementation logic.

    The image is expected to be a CHW tensor normalized to [0, 1].

    A spatial pixel is selected once for either salt or pepper:
      - salt   -> all channels become 1.0
      - pepper -> all channels become 0.0

    Salt and pepper coordinates are non-overlapping.
    """
    img = img_tensor.clone().clamp(0.0, 1.0)

    channels, height, width = img.shape
    total_pixels = height * width

    num_salt = int(total_pixels * salt_prob)
    num_pepper = int(total_pixels * pepper_prob)

    if num_salt + num_pepper > total_pixels:
        raise ValueError(
            "salt_prob + pepper_prob cannot exceed 1.0."
        )

    all_indices = np.arange(total_pixels)
    rng.shuffle(all_indices)

    salt_indices = all_indices[:num_salt]

    pepper_indices = all_indices[
        num_salt:num_salt + num_pepper
    ]

    salt_coords = np.unravel_index(
        salt_indices,
        (height, width),
    )

    pepper_coords = np.unravel_index(
        pepper_indices,
        (height, width),
    )

    # Apply salt to all channels at selected spatial coordinates.
    img[:, salt_coords[0], salt_coords[1]] = 1.0

    # Apply pepper to all channels at selected spatial coordinates.
    img[:, pepper_coords[0], pepper_coords[1]] = 0.0

    return img


# ---------------------------------------------------------------------------
# Load the complete CIFAR-10 training set.
# ---------------------------------------------------------------------------

train_images = []
train_labels = []

for batch_id in range(1, 6):
    images, labels = load_cifar_batch(
        os.path.join(
            dataset_dir,
            f"data_batch_{batch_id}",
        )
    )

    train_images.append(images)
    train_labels.append(labels)

train_images = np.concatenate(
    train_images,
    axis=0,
)

train_labels = np.concatenate(
    train_labels,
    axis=0,
)

if len(train_images) != 50000:
    raise RuntimeError(
        f"Expected 50000 CIFAR-10 training samples, "
        f"found {len(train_images)}."
    )


# ---------------------------------------------------------------------------
# Load the standard clean CIFAR-10 test set.
# ---------------------------------------------------------------------------

test_images, test_labels = load_cifar_batch(
    os.path.join(
        dataset_dir,
        "test_batch",
    )
)

if len(test_images) != 10000:
    raise RuntimeError(
        f"Expected 10000 CIFAR-10 test samples, "
        f"found {len(test_images)}."
    )


# ---------------------------------------------------------------------------
# IID partition
# ---------------------------------------------------------------------------

rng = np.random.default_rng(seed)

indices = rng.permutation(
    len(train_images)
)

client_indices = np.array_split(
    indices,
    num_clients,
)


# ---------------------------------------------------------------------------
# Create client datasets
# ---------------------------------------------------------------------------

for client_id, indices_for_client in enumerate(
    client_indices,
    start=1,
):
    client_images = train_images[
        indices_for_client
    ].copy()

    client_targets = train_labels[
        indices_for_client
    ].copy()

    num_samples = len(client_images)

    if client_id == noisy_client:
        num_noisy = int(
            num_samples * noisy_fraction
        )

        # Randomly select exactly 80% of Client 5's samples.
        noisy_indices = rng.choice(
            num_samples,
            size=num_noisy,
            replace=False,
        )

        noisy_index_set = set(
            noisy_indices.tolist()
        )

        processed_images = []

        for sample_index in range(num_samples):
            # Convert uint8 [0,255] -> float [0,1].
            image = (
                torch.from_numpy(
                    client_images[sample_index]
                ).float()
                / 255.0
            )

            if sample_index in noisy_index_set:
                image = add_salt_pepper_noise(
                    image,
                    salt_prob=salt_prob,
                    pepper_prob=pepper_prob,
                    rng=rng,
                )

            processed_images.append(image)

        client_images_tensor = torch.stack(
            processed_images
        )

        client_labels_tensor = torch.from_numpy(
            client_targets
        )

        print(
            f"Client {client_id}: "
            f"{num_samples} samples "
            f"({num_noisy} noisy, "
            f"{num_samples - num_noisy} clean)"
        )

    else:
        # Clean clients.
        client_images_tensor = (
            torch.from_numpy(client_images)
            .float()
            / 255.0
        )

        client_labels_tensor = torch.from_numpy(
            client_targets
        )

        print(
            f"Client {client_id}: "
            f"{num_samples} clean samples"
        )

    output_file = os.path.join(
        client_dir,
        f"client_{client_id}.pt",
    )

    torch.save(
        {
            "images": client_images_tensor,
            "labels": client_labels_tensor,
        },
        output_file,
    )


# ---------------------------------------------------------------------------
# Save one common clean test set.
# ---------------------------------------------------------------------------

test_images_tensor = (
    torch.from_numpy(test_images)
    .float()
    / 255.0
)

test_labels_tensor = torch.from_numpy(
    test_labels
)

torch.save(
    {
        "images": test_images_tensor,
        "labels": test_labels_tensor,
    },
    test_file,
)


# ---------------------------------------------------------------------------
# Final validation
# ---------------------------------------------------------------------------

print("")
print("Validating generated client datasets...")

for client_id in range(1, num_clients + 1):
    path = os.path.join(
        client_dir,
        f"client_{client_id}.pt",
    )

    if not os.path.isfile(path):
        raise RuntimeError(
            f"Missing client dataset: {path}"
        )

    data = torch.load(
        path,
        weights_only=True,
    )

    if len(data["images"]) != 10000:
        raise RuntimeError(
            f"Client {client_id} does not contain "
            f"10000 samples."
        )

print("All client datasets validated successfully.")

print("")
print("FedSCS federated dataset created:")
print(f"  Clients: {client_dir}")
print(f"  Test:    {test_file}")
PY

echo ""
echo "============================================================"
echo "FedSCS CIFAR-10 dataset preparation completed successfully."
echo "============================================================"
echo "Client datasets: ${CLIENT_DIR}"
echo "Test dataset:    ${TEST_FILE}"
