# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Per-client episode partitioning for the GR00T (Isaac-GR00T N1.7) adapter.

Isaac-GR00T's data stack has **no episode whitelist / subset hook** — the
only knob is ``episode_sampling_rate`` (a *fraction of timesteps*, not a
selection of episodes). For federated learning we need each client to train
on a disjoint set of *episodes*. This module implements that subset by:

1. ``_FilteredEpisodeLoader`` — a subclass of GR00T's
   ``LeRobotEpisodeLoader`` that, immediately after the upstream
   ``_load_metadata()`` runs, drops every entry of ``self.episodes_metadata``
   whose **original episode index** is not in the client's keep-set. Because
   ``__getitem__`` reads ``episode_meta["episode_index"]`` to locate the
   parquet/video files, the surviving metadata still points at the correct
   on-disk episodes (we filter the *list*, the per-episode dicts are intact).

2. ``build_partitioned_dataset`` — constructs a
   ``ShardedSingleStepDataset`` whose internal ``LeRobotEpisodeLoader`` is the
   filtered subclass. Upstream hard-codes that class name inside the dataset's
   ``__init__``, so we inject the subclass via a **localized monkeypatch** of
   ``gr00t.data.dataset.sharded_single_step_dataset.LeRobotEpisodeLoader``,
   restored in a ``finally`` block. The patch window is exactly the dataset
   constructor — no global state leaks.

Everything here imports ``gr00t`` lazily (inside functions) so the adapter
module imports cleanly in environments without Isaac-GR00T installed, per the
decent-vla registry pattern (a plugin whose import fails lands in
``registry.errors`` rather than crashing the CLI).

NOTE (parent verify on GPU): the *original episode index* used for filtering
is ``ep_meta["episode_index"]`` from ``meta/episodes.jsonl``. Upstream's
``__getitem__`` indexes ``self.episodes_metadata`` positionally (``idx`` is a
*list position*, not an episode id) and then reads
``episode_meta["episode_index"]`` to build file paths — so filtering the list
keeps both the positional access and the file resolution correct. Confirm the
``episode_index`` field is present and equals the LeRobot episode id on the
real IPEC LIBERO datasets.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # heavy imports are function-local; this is type-only
    from collections.abc import Iterator, Sequence


def _make_filtered_loader_cls(keep_episode_indices: set[int] | None):
    """Build a ``LeRobotEpisodeLoader`` subclass restricted to ``keep_episode_indices``.

    Defined as a factory (rather than a module-level class) so that the
    ``gr00t`` import stays function-local: ``LeRobotEpisodeLoader`` is only
    imported when a real dataset is being built, never at module import time.

    ``keep_episode_indices`` is a set of **original** LeRobot episode indices
    (the ``"episode_index"`` field in ``meta/episodes.jsonl``). ``None`` means
    "keep all episodes" — the subclass then behaves identically to the base.
    """
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

    class _FilteredEpisodeLoader(LeRobotEpisodeLoader):  # type: ignore[valid-type,misc]
        """LeRobotEpisodeLoader that keeps only a whitelist of episode indices."""

        # The whitelist is a class attribute on the *dynamically created*
        # subclass (closure over keep_episode_indices), so each client's
        # partition gets its own loader class — no shared mutable state.
        _keep_episode_indices: set[int] | None = keep_episode_indices

        def __reduce__(self):
            # This subclass is created inside a factory, so its __qualname__
            # contains "<locals>" and pickle cannot reference it by name. With
            # num_workers>0 and multiprocessing_context="spawn", torch pickles
            # the dataset (and this loader instance) to each worker; without a
            # custom reducer that fails with "Can't pickle local object".
            # Reconstruct through a module-level function instead. The already
            # filtered episodes_metadata travels in __dict__, so the worker
            # never re-runs _load_metadata.
            return (
                _reconstruct_filtered_loader,
                (self._keep_episode_indices, dict(self.__dict__)),
            )

        def _load_metadata(self) -> None:
            # Run upstream metadata parsing first (populates
            # self.episodes_metadata, self.tasks_map, self.stats, etc.), then
            # filter the episode list down to the client's partition. Episode
            # *lengths* are recomputed from the filtered list by the base
            # __init__ via get_episode_lengths() (called after _load_metadata).
            super()._load_metadata()
            keep = self._keep_episode_indices
            if keep is None:
                return
            kept = [ep_meta for ep_meta in self.episodes_metadata if int(ep_meta["episode_index"]) in keep]
            if not kept:
                raise ValueError(
                    "GR00T episode partition resolved to 0 episodes — the "
                    f"client's keep-set {sorted(keep)!r} matched none of the "
                    f"{len(self.episodes_metadata)} episodes in "
                    f"{self.dataset_path}. Empty/invalid partition (the "
                    "per-client shard would be void)."
                )
            self.episodes_metadata = kept

    return _FilteredEpisodeLoader


def _reconstruct_filtered_loader(keep_episode_indices, state):
    """Rebuild a filtered-loader instance in a (possibly spawned) worker.

    Module-level so it is picklable by reference. Re-creates the dynamic
    subclass (lazy gr00t import — available in the worker process) and
    restores the already-loaded instance state via __dict__ WITHOUT re-running
    __init__/_load_metadata (the filtered episodes_metadata is part of `state`).
    """
    cls = _make_filtered_loader_cls(keep_episode_indices)
    obj = cls.__new__(cls)
    obj.__dict__.update(state)
    return obj


@contextmanager
def _patched_episode_loader(loader_cls) -> Iterator[None]:
    """Temporarily point ShardedSingleStepDataset at ``loader_cls``.

    ``ShardedSingleStepDataset.__init__`` constructs
    ``LeRobotEpisodeLoader(...)`` by the name it imported at module load. We
    rebind that name to our filtered subclass for the duration of the
    constructor, then restore it. Scope is one dataset build; nothing leaks.
    """
    import gr00t.data.dataset.sharded_single_step_dataset as sss

    original = sss.LeRobotEpisodeLoader
    sss.LeRobotEpisodeLoader = loader_cls
    try:
        yield
    finally:
        sss.LeRobotEpisodeLoader = original


def build_partitioned_dataset(
    dataset_path: str,
    embodiment_tag_value: str,
    modality_configs: dict[str, Any],
    keep_episode_indices: Sequence[int] | None,
    *,
    video_backend: str = "torchcodec",
    video_backend_kwargs: dict[str, Any] | None = None,
    shard_size: int = 2**10,
    episode_sampling_rate: float = 1.0,
    seed: int = 42,
    allow_padding: bool = False,
):
    """Construct a ``ShardedSingleStepDataset`` restricted to a client's episodes.

    Args:
        dataset_path: Path to one LeRobot v2.x dataset dir (a single LIBERO
            suite, e.g. ``.../libero_spatial_no_noops_1.0.0_lerobot``). Must
            contain ``meta/`` with ``info.json``, ``episodes.jsonl``,
            ``tasks.jsonl``, ``modality.json`` and ``stats.json``.
        embodiment_tag_value: The embodiment tag *value* string, e.g.
            ``"libero_sim"`` (``EmbodimentTag.LIBERO_PANDA.value``).
        modality_configs: The per-embodiment modality config dict for this
            embodiment, i.e. ``MODALITY_CONFIGS["libero_sim"]`` (video / state /
            action / language ``ModalityConfig`` entries).
        keep_episode_indices: Original LeRobot episode indices to keep for this
            client. ``None`` -> use every episode.
        episode_sampling_rate: Fraction of *timesteps* per episode to shard
            (GR00T's own knob). Defaults to 1.0 (use all timesteps) — the
            upstream finetune default is 0.1, but for FL we want determinism
            and full coverage of the small per-client partitions.

    Returns:
        A ``ShardedSingleStepDataset``. NOTE: this is **shard-based**, not a
        torch map-style dataset (``__len__`` is the shard count and
        ``get_shard(i)`` returns a list of processed timesteps). It must have
        its ``processor`` set (``dataset.set_processor(processor)``) before
        ``get_shard``/``get_datapoint`` will work — ``build_dataloader`` in the
        backend handles that wiring.
    """
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
    from gr00t.data.embodiment_tags import EmbodimentTag

    keep_set = None if keep_episode_indices is None else {int(i) for i in keep_episode_indices}
    loader_cls = _make_filtered_loader_cls(keep_set)

    with _patched_episode_loader(loader_cls):
        dataset = ShardedSingleStepDataset(
            dataset_path=dataset_path,
            embodiment_tag=EmbodimentTag(embodiment_tag_value),
            modality_configs=modality_configs,
            video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
            shard_size=shard_size,
            episode_sampling_rate=episode_sampling_rate,
            seed=seed,
            allow_padding=allow_padding,
        )
    return dataset
