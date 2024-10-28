# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import skimage.metrics
import torch
from scipy import spatial

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext


class SimMetric(FLComponent):
    def __init__(self):
        """Abstract class used to compute a similarity metric between the source data and reconstruction.

        Returns:
            similarity metric
        """
        super().__init__()

    def __call__(self, source, reference):
        """Subclass must implement this method to filter the provided DXO

        Args:
            source: source image
            reference: reference image the source image should be compared to

        Returns:
            similarity metric

        """
        raise NotImplementedError(f"Subclass of {self.__class__.__name__} must implement this method.")


class ImageSimMetric(SimMetric):
    def __init__(self, metrics=None):
        """Implementation of `SimMetric` for imaging applications.
        Args:
            metrics: String or list of similarity metrics. Support "ssim", "cosine", and "norm".

        Returns:
            similarity metric
        """
        super().__init__()
        if not metrics:
            self.metrics = ["ssim"]
        else:
            if isinstance(metrics, str):
                self.metrics = [metrics]
            elif isinstance(metrics, list):
                self.metrics = metrics
            else:
                raise ValueError("Expected `metrics` to be string or list of strings.")

    @staticmethod
    def check_shape(img):
        if len(np.shape(img)) > 2:
            if np.shape(img)[2] > 3:
                img = img[:, :, 0:3]

        # sim metrics assume images to scaled 0...255
        if np.max(img) <= 1.0:
            img = img * 255.0

        assert np.shape(img)[2] == 3, "Assuming RGB image here"

        return img

    def __call__(self, source, reference, fl_ctx: FLContext, is_channel_first=False):

        # check type
        if isinstance(source, torch.Tensor):
            source = source.detach().cpu().numpy()
        if isinstance(reference, torch.Tensor):
            reference = reference.detach().cpu().numpy()

        source = source.astype(np.float32)
        reference = reference.astype(np.float32)

        if is_channel_first:
            source = np.swapaxes(source, -1, 0)
            reference = np.swapaxes(reference, -1, 0)

        source = self.check_shape(source)
        reference = self.check_shape(reference)

        if not source.shape == reference.shape:
            raise ValueError(
                f"`source` and `reference` must have the same dimensions but they were {source.shape} vs. {reference.shape}."
            )

        # TODO: convert these to warnings
        assert np.min(source) >= 0, f"img min is {np.min(source)}"
        assert np.min(reference) >= 0, f"img_recon min is {np.min(reference)}"
        assert 1 < np.max(source) <= 255
        assert 1 < np.max(reference) <= 255
        # assert np.mean(img) > 1
        # assert np.mean(img_recon) > 1
        if np.mean(source) < 1:
            self.log_warning(
                fl_ctx,
                f"[WARNING] image mean is very low {np.mean(source)} (min={np.min(source)}, max={np.max(source)})",
            )
        if np.mean(reference) < 1:
            self.log_warning(
                fl_ctx,
                f"[WARNING] image mean is very low {np.mean(reference)} (min={np.min(reference)}, max={np.max(reference)})",
            )

        # compute metrics
        outputs = {}
        for metric in self.metrics:
            if not isinstance(metric, str):
                raise ValueError(f"Expect metric to be of type string but got type {type(metric)}")
            if metric == "ssim":
                out_value = skimage.metrics.structural_similarity(
                    reference, source, channel_axis=-1, data_range=256.0
                )  # ,
                # gaussian_weights=True, sigma=1.5,  # Wang et al.
                # use_sample_covariance=False)
            elif metric == "cosine":
                out_value = 1 - spatial.distance.cosine(reference, source)
                assert 0.0 <= out_value <= 1.0, f"cosine similarity is out of range {out_value}"
            elif metric == "norm":
                out_value = np.sqrt(np.sum((reference - source) ** 2))
            else:
                raise ValueError(f"Metric {metric} not supported! Choose from `ssim`, `cosine` or `norm`")
            outputs[metric] = out_value

        return outputs
