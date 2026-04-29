# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from threading import Lock
from typing import List, Tuple, Union

import numpy as np

from nvflare.apis.dxo import DXO, DataKind, MetaKey
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable

# Keep temporary SVT arrays bounded for large models.
_SVT_CHUNK_SIZE = 1_000_000
_SVT_PRIVACY_BUDGET = "svt_privacy_budget"
_SVT_PRIVACY_ACCOUNTANT = "svt_privacy_accountant"


def _sample_partition_counts(group_counts: List[int], total_to_sample: int, replace: bool) -> List[int]:
    """Partition ``total_to_sample`` across groups while preserving the total.

    The last group absorbs any remaining samples after the sequential draws so the
    returned counts always sum to ``total_to_sample``. When ``replace`` is ``True``,
    a group can receive more selected samples than accepted entries because the
    downstream selection step samples from that group's accepted entries with
    replacement.
    """
    sampled_counts = []
    remaining_groups = int(sum(group_counts))
    remaining_to_sample = int(total_to_sample)

    for idx, group_count in enumerate(group_counts):
        group_count = int(group_count)

        if idx == len(group_counts) - 1:
            sampled = remaining_to_sample
        elif remaining_to_sample == 0 or group_count == 0:
            sampled = 0
        elif replace:
            sampled = int(np.random.binomial(remaining_to_sample, float(group_count) / float(remaining_groups)))
        else:
            sampled = int(
                np.random.hypergeometric(
                    ngood=group_count,
                    nbad=remaining_groups - group_count,
                    nsample=remaining_to_sample,
                )
            )

        sampled_counts.append(sampled)
        remaining_groups -= group_count
        remaining_to_sample -= sampled

    return sampled_counts


def _compute_epsilon_split(
    epsilon: float, n_upload: int, epsilon_threshold: float = None, epsilon_query: float = None
) -> Tuple[float, float]:
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0.")
    if n_upload <= 0:
        raise ValueError("n_upload must be > 0.")

    if epsilon_threshold is not None:
        epsilon_threshold = float(epsilon_threshold)
        if epsilon_threshold <= 0:
            raise ValueError("epsilon_threshold must be > 0.")

    if epsilon_query is not None:
        epsilon_query = float(epsilon_query)
        if epsilon_query <= 0:
            raise ValueError("epsilon_query must be > 0.")

    if epsilon_threshold is None and epsilon_query is None:
        # Standard SVT budget allocation for non-monotonic queries:
        # epsilon_threshold : epsilon_query = 1 : (2c)^(2/3).
        query_ratio = float((2.0 * n_upload) ** (2.0 / 3.0))
        epsilon_threshold = float(epsilon) / (1.0 + query_ratio)
        epsilon_query = float(epsilon) - epsilon_threshold
    elif epsilon_threshold is None:
        epsilon_threshold = float(epsilon) - epsilon_query
    elif epsilon_query is None:
        epsilon_query = float(epsilon) - epsilon_threshold

    if epsilon_threshold <= 0 or epsilon_query <= 0:
        raise ValueError("epsilon_threshold and epsilon_query must both be > 0.")

    if not np.isclose(epsilon_threshold + epsilon_query, float(epsilon)):
        raise ValueError("epsilon_threshold + epsilon_query must equal epsilon.")

    return epsilon_threshold, epsilon_query


def _compute_release_epsilon(n_upload: int, noise_var: float, epsilon_release: float = None) -> float:
    if n_upload <= 0:
        raise ValueError("n_upload must be > 0.")

    if epsilon_release is not None:
        epsilon_release = float(epsilon_release)
        if epsilon_release <= 0:
            raise ValueError("epsilon_release must be > 0.")
        return epsilon_release

    noise_var = float(noise_var)
    if noise_var <= 0:
        raise ValueError("noise_var must be > 0 when epsilon_release is not specified.")

    # Preserve the previous release noise scale:
    # scale = 2 * gamma / noise_var = (2 * gamma * c) / (c * noise_var)
    # which corresponds to a total release budget epsilon_release = c * noise_var.
    return float(n_upload) * noise_var


class SVTPrivacy(DXOFilter):
    def __init__(
        self,
        fraction=0.1,
        epsilon=0.1,
        noise_var=0.1,
        gamma=1e-5,
        tau=1e-6,
        data_kinds: [str] = None,
        replace=True,
        epsilon_threshold: float = None,
        epsilon_query: float = None,
        epsilon_release: float = None,
    ):
        """Implementation of the standard Sparse Vector Technique (SVT) differential privacy algorithm.

        lambda_rho = gamma / epsilon_threshold
        threshold = tau + np.random.laplace(scale=lambda_rho)

        Args:
            fraction (float, optional): used to determine dataset threshold. Defaults to 0.1.
            epsilon (float, optional): total privacy budget used for the threshold/query selection phase.
                Defaults to 0.1.
            noise_var (float, optional): legacy release-noise control retained for backward compatibility.
                When epsilon_release is not specified, this implies a total release budget of
                ``n_upload * noise_var`` and preserves the previous release noise scale. Defaults to 0.1.
            gamma (float, optional): Defaults to 1e-5.
            tau (float, optional): Defaults to 1e-6.
            data_kinds (str, optional): Defaults to None.
            replace (bool): whether to sample with replacement. Defaults to True.
            epsilon_threshold (float, optional): privacy budget used for threshold noise. When omitted,
                the standard SVT non-monotonic split is used.
            epsilon_query (float, optional): privacy budget used for query noise. When omitted,
                the remainder of epsilon is assigned after applying the selected split.
            epsilon_release (float, optional): total privacy budget used for the release-noise phase.
                When omitted, the budget is derived from noise_var to preserve existing behavior.
        """
        if not data_kinds:
            data_kinds = [DataKind.WEIGHT_DIFF, DataKind.WEIGHTS]

        super().__init__(supported_data_kinds=[DataKind.WEIGHTS, DataKind.WEIGHT_DIFF], data_kinds_to_filter=data_kinds)

        self.fraction = fraction  # fraction of the model to upload
        self.epsilon = epsilon
        self.noise_var = noise_var
        self.gamma = gamma
        self.tau = tau
        self.replace = replace
        self.epsilon_threshold = epsilon_threshold
        self.epsilon_query = epsilon_query
        self.epsilon_release = epsilon_release
        self._accountant_lock = Lock()
        self._reset_accountant()

    def _reset_accountant(self):
        with self._accountant_lock:
            self._privacy_spent = 0.0
            self._privacy_ledger = []

    def get_privacy_spent(self) -> float:
        with self._accountant_lock:
            return self._privacy_spent

    def get_privacy_ledger(self) -> List[dict]:
        with self._accountant_lock:
            return list(self._privacy_ledger)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        super().handle_event(event_type, fl_ctx)
        if event_type == EventType.START_RUN:
            self._reset_accountant()

    def _record_privacy_accounting(
        self,
        dxo: DXO,
        shareable: Shareable,
        has_scalar_passthrough: bool,
        n_upload: int,
        acceptance_passes: int,
        epsilon_threshold: float,
        epsilon_query: float,
        epsilon_release: float,
        fl_ctx: FLContext,
    ):
        current_round = shareable.get_header(MetaKey.CURRENT_ROUND, dxo.get_meta_prop(MetaKey.CURRENT_ROUND, None))
        epsilon_total = epsilon_threshold + epsilon_query + epsilon_release

        per_call = {
            "accountant": "pure_dp_sum",
            "scope": "filter-level",
            "delta": 0.0,
            "round": current_round,
            "epsilon_total": epsilon_total,
            "epsilon_threshold": epsilon_threshold,
            "epsilon_query": epsilon_query,
            "epsilon_release": epsilon_release,
            "n_upload": n_upload,
            "acceptance_passes": acceptance_passes,
            "has_scalar_passthrough": has_scalar_passthrough,
        }
        if has_scalar_passthrough:
            per_call["note"] = "scalar passthrough is not noise-protected by the SVT accountant."

        with self._accountant_lock:
            self._privacy_spent += epsilon_total
            self._privacy_ledger.append(dict(per_call))
            cumulative_epsilon = self._privacy_spent
            cumulative_calls = len(self._privacy_ledger)

        cumulative = {
            "accountant": "pure_dp_sum",
            "scope": "filter-level",
            "delta": 0.0,
            "epsilon_total": cumulative_epsilon,
            "calls": cumulative_calls,
        }
        if current_round is not None:
            cumulative["latest_round"] = current_round
        if has_scalar_passthrough:
            cumulative["note"] = "scalar passthrough is not included in the SVT accountant guarantee."

        dxo.set_meta_prop(_SVT_PRIVACY_BUDGET, per_call)
        dxo.set_meta_prop(_SVT_PRIVACY_ACCOUNTANT, cumulative)

        self.log_info(
            fl_ctx,
            "SVT privacy accountant: call epsilon %.6g, cumulative epsilon %.6g (delta=0, pure composition)."
            % (epsilon_total, cumulative_epsilon),
        )

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> Union[None, DXO]:
        """Compute the differentially private SVT.

        Args:
            dxo: information from client
            shareable: that the dxo belongs to
            fl_ctx: context provided by workflow

        Returns: filtered result.
        """
        self.log_debug(fl_ctx, "inside filter")
        model_diff = dxo.data
        total_steps = np.float64(dxo.get_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, 1))

        param_items: List[Tuple[str, np.ndarray, bool]] = []
        total_params = 0
        max_abs = 0.0
        min_abs = None
        has_scalar_passthrough = False

        for name in sorted(model_diff):
            value = np.asarray(model_diff[name])
            is_scalar = np.ndim(value) == 0
            has_scalar_passthrough = has_scalar_passthrough or is_scalar
            flat_value = value.reshape(-1)
            total_params += flat_value.size
            param_items.append((name, value, is_scalar))

            for start in range(0, flat_value.size, _SVT_CHUNK_SIZE):
                end = min(start + _SVT_CHUNK_SIZE, flat_value.size)
                abs_chunk = np.abs(flat_value[start:end].astype(np.float64, copy=False) / total_steps)
                if abs_chunk.size == 0:
                    continue
                max_abs = max(max_abs, float(np.max(abs_chunk)))
                chunk_min = float(np.min(abs_chunk))
                min_abs = chunk_min if min_abs is None else min(min_abs, chunk_min)

        if total_params == 0:
            return dxo

        self.log_info(fl_ctx, f"Delta_w: Max abs: {max_abs}, Min abs: {min_abs}, total params: {total_params}.")

        n_upload = int(min(np.ceil(float(total_params) * self.fraction), float(total_params)))
        if n_upload <= 0:
            dp_w = {}
            for name, value, is_scalar in param_items:
                dp_w[name] = value if is_scalar else np.zeros(value.shape, dtype=value.dtype)
            dxo.data = dp_w
            return dxo

        epsilon_threshold, epsilon_query = _compute_epsilon_split(
            self.epsilon,
            n_upload,
            epsilon_threshold=self.epsilon_threshold,
            epsilon_query=self.epsilon_query,
        )
        epsilon_release = _compute_release_epsilon(
            n_upload,
            self.noise_var,
            epsilon_release=self.epsilon_release,
        )

        # eps_1: threshold with noise
        lambda_rho = self.gamma / epsilon_threshold
        threshold = self.tau + np.random.laplace(scale=lambda_rho)
        # eps_2: query with noise
        lambda_nu = self.gamma * 2.0 * n_upload / epsilon_query
        noise_scale = self.gamma * 2.0 * n_upload / epsilon_release
        self.logger.info(
            "total params: %s, epsilon: %s, "
            "threshold epsilon: %s, query epsilon: %s, release epsilon: %s, "
            "threshold tau: %s + f(eps_1) = %s, "
            "clip gamma: %s",
            total_params,
            self.epsilon,
            epsilon_threshold,
            epsilon_query,
            epsilon_release,
            self.tau,
            threshold,
            self.gamma,
        )

        accepted_masks = [np.zeros(value.size, dtype=np.bool_) for _, value, _ in param_items]
        accepted_counts = [0] * len(param_items)
        total_accepted = 0
        acceptance_passes = 0

        while total_accepted < n_upload:
            acceptance_passes += 1
            accepted_before_pass = total_accepted

            for idx, (_, value, _) in enumerate(param_items):
                flat_value = value.reshape(-1)
                accepted_mask = accepted_masks[idx]

                for start in range(0, flat_value.size, _SVT_CHUNK_SIZE):
                    end = min(start + _SVT_CHUNK_SIZE, flat_value.size)
                    mask_chunk = accepted_mask[start:end]
                    remaining_chunk = ~mask_chunk
                    if not np.any(remaining_chunk):
                        continue

                    candidate_values = (
                        flat_value[start:end][remaining_chunk].astype(np.float64, copy=False) / total_steps
                    )
                    noisy_response = np.abs(np.clip(candidate_values, a_min=-self.gamma, a_max=self.gamma))
                    noisy_response += np.random.laplace(scale=lambda_nu, size=candidate_values.size)
                    above_threshold = noisy_response >= threshold
                    if not np.any(above_threshold):
                        continue

                    accepted_idx = np.flatnonzero(remaining_chunk)[above_threshold]
                    mask_chunk[accepted_idx] = True
                    n_new = int(accepted_idx.size)
                    accepted_counts[idx] += n_new
                    total_accepted += n_new

            self.log_debug(
                fl_ctx,
                "SVT acceptance pass {}: +{} responses (total {}/{})".format(
                    acceptance_passes,
                    total_accepted - accepted_before_pass,
                    total_accepted,
                    n_upload,
                ),
            )

        self.log_info(
            fl_ctx,
            "SVT acceptance completed in {} passes to reach {}/{} accepted responses.".format(
                acceptance_passes, total_accepted, n_upload
            ),
        )

        selected_counts = _sample_partition_counts(accepted_counts, n_upload, self.replace)

        noise_max = 0.0
        noise_medians = []
        dp_w = {}

        for idx, (name, value, is_scalar) in enumerate(param_items):
            flat_value = value.reshape(-1)
            accepted_mask = accepted_masks[idx]
            remaining_accepted = accepted_counts[idx]
            remaining_selected = selected_counts[idx]

            if is_scalar:
                dp_w[name] = value
                output_flat = None
            else:
                dp_w[name] = np.zeros(value.shape, dtype=value.dtype)
                output_flat = dp_w[name].reshape(-1)

            for start in range(0, flat_value.size, _SVT_CHUNK_SIZE):
                end = min(start + _SVT_CHUNK_SIZE, flat_value.size)
                mask_chunk = accepted_mask[start:end]
                chunk_accepted = int(np.sum(mask_chunk))

                if end == flat_value.size:
                    chunk_selected = remaining_selected
                elif remaining_selected == 0 or chunk_accepted == 0:
                    chunk_selected = 0
                elif self.replace:
                    chunk_selected = int(
                        np.random.binomial(
                            remaining_selected,
                            float(chunk_accepted) / float(remaining_accepted),
                        )
                    )
                else:
                    chunk_selected = int(
                        np.random.hypergeometric(
                            ngood=chunk_accepted,
                            nbad=remaining_accepted - chunk_accepted,
                            nsample=remaining_selected,
                        )
                    )

                if is_scalar:
                    remaining_accepted -= chunk_accepted
                    remaining_selected -= chunk_selected
                    continue
                elif chunk_selected > 0:
                    accepted_idx = np.flatnonzero(mask_chunk)
                    selected_idx = np.random.choice(accepted_idx, size=chunk_selected, replace=self.replace)
                    noise = np.random.laplace(scale=noise_scale, size=chunk_selected)
                    noise_max = max(noise_max, float(np.max(np.abs(noise))))
                    noise_medians.append(np.median(np.abs(noise)))

                    chunk = flat_value[start:end]
                    selected_values = chunk[selected_idx].astype(np.float64, copy=False) / total_steps
                    selected_values = np.clip(selected_values, a_min=-self.gamma, a_max=self.gamma)
                    selected_values = (selected_values + noise) * total_steps
                    output_flat[start:end][selected_idx] = selected_values

                remaining_accepted -= chunk_accepted
                remaining_selected -= chunk_selected

        median_noise = float(np.median(noise_medians)) if noise_medians else 0.0
        self.log_info(fl_ctx, "noise max: {}, median approx {}".format(noise_max, median_noise))

        dxo.data = dp_w
        self._record_privacy_accounting(
            dxo,
            shareable,
            has_scalar_passthrough,
            n_upload,
            acceptance_passes,
            epsilon_threshold,
            epsilon_query,
            epsilon_release,
            fl_ctx,
        )
        return dxo
