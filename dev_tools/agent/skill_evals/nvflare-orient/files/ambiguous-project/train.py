# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from project_runtime import build_training_session


def main():
    session = build_training_session()
    session.fit()


if __name__ == "__main__":
    main()
