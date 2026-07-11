# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Template for an image-intensity federated statistics client.

Adapt this template when generating the client:
- swap ``_load_image`` for the format the site actually holds: PIL covers
  PNG/JPEG/BMP/TIFF; DICOM needs ``pydicom`` (or MONAI's ITK reader), NIfTI
  needs ``nibabel`` — preflight the import for the chosen loader and fail
  closed on a missing dependency;
- ``count`` is the number of discovered image files (unreadable files are
  only detected during the histogram pass and surface via ``failure_count``,
  which must be configured explicitly to appear in output; failed paths
  are deduplicated so retried passes cannot inflate the count);
- grayscale conversion is applied before histogramming; state that policy in
  the report (multi-channel per-channel statistics are not supported here);
- parameterize the data location by site identity; do not hardcode one
  site's absolute path.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.statistics_spec import Bin, DataType, Feature, Histogram, HistogramType, Statistics

# ADAPTATION POINT: extend with the site's actual formats (e.g. ".dcm",
# ".nii", ".nii.gz") when swapping _load_image for the matching loader; the
# dataset block's per-site image_formats says which formats are present.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class ImageIntensityStatistics(Statistics):
    def __init__(self, data_root_dir: str, dataset_name: str = "data"):
        """Local pixel-intensity statistics generator for image folders.

        Args:
            data_root_dir: root directory holding one subdirectory per site.
            dataset_name: dataset label used in the output.
        """
        super().__init__()
        self.data_root_dir = data_root_dir
        self.dataset_name = dataset_name
        self.image_paths: Optional[List[Path]] = None
        # failed paths are deduplicated so a retried histogram() call cannot
        # count the same unreadable file twice
        self.failed_paths: set = set()

    def initialize(self, fl_ctx: FLContext):
        site_name = fl_ctx.get_identity_name()
        site_dir = Path(self.data_root_dir) / site_name
        # ADAPTATION POINT: adjust discovery to the site's layout (recursive
        # globs, a datalist JSON, or format-specific extensions).
        self.image_paths = sorted(p for p in site_dir.iterdir() if p.name.lower().endswith(IMAGE_EXTENSIONS))
        if not self.image_paths:
            raise ValueError(f"no image files found under {site_dir}")
        self.log_info(fl_ctx, f"site {site_name}: {len(self.image_paths)} image files")

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        # ADAPTATION POINT: replace with the loader for the site's format
        # (pydicom for DICOM, nibabel for NIfTI); keep the returned array as
        # raw intensities.
        from PIL import Image

        return np.asarray(Image.open(path).convert("L"))

    def features(self) -> Dict[str, List[Feature]]:
        return {self.dataset_name: [Feature("intensity", DataType.FLOAT)]}

    def count(self, dataset_name: str, feature_name: str) -> int:
        return len(self.image_paths)

    def failure_count(self, dataset_name: str, feature_name: str) -> int:
        return len(self.failed_paths)

    def histogram(
        self, dataset_name: str, feature_name: str, num_of_bins: int, global_min_value: float, global_max_value: float
    ) -> Histogram:
        totals = np.zeros((num_of_bins,), dtype=np.int64)
        edges = None
        for path in self.image_paths:
            try:
                arr = self._load_image(path)
                counts, edges = np.histogram(arr, bins=num_of_bins, range=(global_min_value, global_max_value))
                totals += counts
            except Exception:
                self.failed_paths.add(path)
        if edges is None:
            raise ValueError(f"no readable images for {dataset_name}/{feature_name}")
        bins = [
            Bin(low_value=float(edges[j]), high_value=float(edges[j + 1]), sample_count=int(totals[j]))
            for j in range(num_of_bins)
        ]
        return Histogram(HistogramType.STANDARD, bins)
