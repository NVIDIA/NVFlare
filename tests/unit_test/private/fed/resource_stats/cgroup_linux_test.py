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

"""Controlled discovery and limit selection without host cgroup assumptions."""

import ctypes
from pathlib import Path

import pytest

from nvflare.private.fed.resource_stats import collector
from nvflare.private.fed.resource_stats.probes import cgroup_linux as probe

_GIB = 2**30


@pytest.fixture(autouse=True)
def linux_host(monkeypatch):
    monkeypatch.setattr(probe.platform, "system", lambda: "Linux")
    monkeypatch.setattr(probe.os, "sched_getaffinity", lambda _pid: set(range(8)), raising=False)
    monkeypatch.setattr(
        probe.os,
        "sysconf",
        lambda key: {"SC_NPROCESSORS_ONLN": 8, "SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 2_097_152}[key],
    )
    monkeypatch.setattr(probe.platform, "machine", lambda: "")


def _hierarchy(monkeypatch, membership, mounts, files):
    # Exercise the real membership/mount parsers and ancestor traversal, not
    # only a mocked _cgroup_location which could hide discovery regressions.
    mountinfo = "\n".join(
        f"{index} 0 0:{index} {root} {mount} rw - {kind} source {options}"
        for index, (kind, root, mount, options) in enumerate(mounts, 1)
    )
    text = {Path("/proc/self/cgroup"): membership, Path("/proc/self/mountinfo"): mountinfo}
    monkeypatch.setattr(probe, "_read_text", lambda path: text.get(path))
    values = {Path(path): value for path, value in files.items()}
    monkeypatch.setattr(probe, "_read_cgroup_file", lambda path: values.get(path, probe._CGROUP_FILE_MISSING))
    return text


@pytest.mark.parametrize(
    "version,quota,expected,partial",
    [
        ("v1", "2", "0.666666666", False),
        ("v2", "2 3", "0.666666666", False),
        ("v1", "bad", "8", True),
        ("v2", "garbage", "8", True),
        ("v2", probe._CGROUP_FILE_UNREADABLE, "8", True),
        ("v2", "max 100000", "8", False),
    ],
)
def test_cpu_quota_rounding_fallback_and_unlimited(monkeypatch, version, quota, expected, partial):
    if version == "v1":
        membership, mount = "3:cpu:/job", ("cgroup", "/", "/cg", "rw,cpu")
        files = {
            "/cg/job/cpu.cfs_quota_us": quota,
            "/cg/job/cpu.cfs_period_us": "3",
            "/cg/cpu.cfs_quota_us": "-1",
            "/cg/cpu.cfs_period_us": "100000",
        }
    else:
        membership, mount = "0::/job", ("cgroup2", "/", "/cg", "rw")
        files = {"/cg/job/cpu.max": quota, "/cg/cpu.max": "max 100000"}
    _hierarchy(monkeypatch, membership, [mount], files)

    issues = set()
    assert probe.probe_cpu(issues=issues) == {"groups": [{"units": expected}]}
    assert issues == ({"observation_incomplete"} if partial else set())


@pytest.mark.parametrize(
    "version,limit,partial",
    [
        ("v1", "garbage", True),
        ("v1", probe._CGROUP_FILE_UNREADABLE, True),
        ("v2", "garbage", True),
        ("v2", probe._CGROUP_FILE_UNREADABLE, True),
        ("v2", "max", False),
    ],
)
def test_memory_limit_fallback_and_unlimited(monkeypatch, version, limit, partial):
    if version == "v1":
        membership, mount = "4:memory:/job", ("cgroup", "/", "/cg", "rw,memory")
        files = {"/cg/job/memory.limit_in_bytes": limit}
    else:
        membership, mount = "0::/job", ("cgroup2", "/", "/cg", "rw")
        files = {"/cg/job/memory.max": limit, "/cg/memory.max": "max"}
    _hierarchy(monkeypatch, membership, [mount], files)

    issues = set()
    assert probe.probe_memory(issues=issues) == {"bytes": str(8 * _GIB)}
    assert issues == ({"observation_incomplete"} if partial else set())


def test_v2_capacity_uses_cpu_and_memory_ancestor_limits(monkeypatch):
    _hierarchy(
        monkeypatch,
        "0::/group/job",
        [("cgroup2", "/", "/cg", "rw")],
        {
            "/cg/group/job/cpu.max": "400000 100000",
            "/cg/group/cpu.max": "300000 100000",
            "/cg/group/job/cpuset.cpus.effective": "0-7",
            "/cg/group/job/memory.max": str(4 * _GIB),
            "/cg/group/memory.max": str(2 * _GIB),
        },
    )

    assert probe.probe_cpu() == {"groups": [{"units": "3"}]}
    assert probe.probe_memory() == {"bytes": str(2 * _GIB)}


def test_v1_capacity_uses_each_controller_and_ancestor(monkeypatch):
    _hierarchy(
        monkeypatch,
        "2:cpuset:/job\n3:cpu,cpuacct:/job\n4:memory:/job",
        [
            ("cgroup", "/", "/cpuset", "rw,cpuset"),
            ("cgroup", "/", "/cpu", "rw,cpu,cpuacct"),
            ("cgroup", "/", "/memory", "rw,memory"),
        ],
        {
            "/cpuset/job/cpuset.cpus": "0-7",
            "/cpuset/cpuset.cpus": "0-15",
            "/cpu/job/cpu.cfs_quota_us": "400000",
            "/cpu/job/cpu.cfs_period_us": "100000",
            "/cpu/cpu.cfs_quota_us": "200000",
            "/cpu/cpu.cfs_period_us": "100000",
            "/memory/job/memory.limit_in_bytes": str(4 * _GIB),
            "/memory/memory.limit_in_bytes": str(_GIB),
        },
    )

    assert probe.probe_cpu() == {"groups": [{"units": "2"}]}
    assert probe.probe_memory() == {"bytes": str(_GIB)}


def test_hybrid_v2_mount_does_not_hide_v1_cpu_or_memory(monkeypatch):
    _hierarchy(
        monkeypatch,
        "0::/job\n3:cpu,cpuacct:/job\n4:memory:/job",
        [
            ("cgroup2", "/", "/unified", "rw"),
            ("cgroup", "/", "/cpu", "rw,cpu,cpuacct"),
            ("cgroup", "/", "/memory", "rw,memory"),
        ],
        {
            "/cpu/job/cpu.cfs_quota_us": "200000",
            "/cpu/job/cpu.cfs_period_us": "100000",
            "/cpu/cpu.cfs_quota_us": "-1",
            "/cpu/cpu.cfs_period_us": "100000",
            "/memory/job/memory.limit_in_bytes": str(_GIB),
            "/memory/memory.limit_in_bytes": str(8 * _GIB),
        },
    )

    assert probe.probe_cpu() == {"groups": [{"units": "2"}]}
    assert probe.probe_memory() == {"bytes": str(_GIB)}


def test_hybrid_controller_assignments_are_independent(monkeypatch):
    monkeypatch.setattr(probe.os, "sched_getaffinity", lambda _pid: set(range(4)))
    _hierarchy(
        monkeypatch,
        "0::/job\n2:cpuset:/job\n4:memory:/job",
        [
            ("cgroup2", "/", "/unified", "rw"),
            ("cgroup", "/", "/cpuset", "rw,cpuset"),
            ("cgroup", "/", "/memory", "rw,memory"),
        ],
        {
            "/unified/job/cpu.max": "150000 100000",
            "/cpuset/job/cpuset.effective_cpus": "0-3",
            "/cpuset/cpuset.cpus": "0-7",
            "/memory/job/memory.limit_in_bytes": str(_GIB),
            "/memory/memory.limit_in_bytes": str(8 * _GIB),
        },
    )

    assert probe.probe_cpu() == {"groups": [{"units": "1.5"}]}
    assert probe.probe_memory() == {"bytes": str(_GIB)}


@pytest.mark.parametrize("visible_v2", [True, False])
def test_v1_assigned_controllers_do_not_depend_on_unrelated_v2_files(monkeypatch, visible_v2):
    monkeypatch.setattr(probe.os, "sched_getaffinity", lambda _pid: set(range(4)))
    mounts = [("cgroup", "/", "/v1", "rw,cpuset,cpu,memory")]
    if visible_v2:
        mounts.insert(0, ("cgroup2", "/", "/unified", "rw"))
    _hierarchy(
        monkeypatch,
        "0::/job\n3:cpuset,cpu,memory:/",
        mounts,
        {
            "/v1/cpuset.cpus": "0-3",
            "/v1/cpu.cfs_quota_us": "200000",
            "/v1/cpu.cfs_period_us": "100000",
            "/v1/memory.limit_in_bytes": str(_GIB),
            "/unified/job/cpu.max": probe._CGROUP_FILE_UNREADABLE,
            "/unified/job/cpuset.cpus.effective": probe._CGROUP_FILE_UNREADABLE,
            "/unified/job/memory.max": probe._CGROUP_FILE_UNREADABLE,
        },
    )

    assert probe.probe_cpu() == {"groups": [{"units": "2"}]}
    assert probe.probe_memory() == {"bytes": str(_GIB)}


@pytest.mark.parametrize("bad", ["not-a-limit", probe._CGROUP_FILE_UNREADABLE, probe._CGROUP_FILE_MISSING])
@pytest.mark.parametrize("controller", ["cpu", "memory", "cpuset"])
def test_applicable_v1_errors_retain_fallback_even_with_a_v2_mount(monkeypatch, controller, bad):
    if controller == "cpuset":
        monkeypatch.setattr(probe.os, "sched_getaffinity", lambda _pid: set())
    files = {
        "/v1/cpu.cfs_quota_us": "200000",
        "/v1/cpu.cfs_period_us": "100000",
        "/v1/memory.limit_in_bytes": str(_GIB),
        "/v1/cpuset.cpus": "0-3",
    }
    field = {"cpu": "cpu.cfs_quota_us", "memory": "memory.limit_in_bytes", "cpuset": "cpuset.cpus"}[controller]
    files[f"/v1/{field}"] = bad
    _hierarchy(
        monkeypatch,
        f"0::/job\n3:{controller}:/",
        [("cgroup2", "/", "/unified", "rw"), ("cgroup", "/", "/v1", f"rw,{controller}")],
        files,
    )

    issues = set()
    result = probe.probe_memory(issues=issues) if controller == "memory" else probe.probe_cpu(issues=issues)
    assert result == ({"bytes": str(8 * _GIB)} if controller == "memory" else {"groups": [{"units": "8"}]})
    assert issues == {"observation_incomplete"}


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("bad_cpuset", ["not-a-cpu-list", probe._CGROUP_FILE_UNREADABLE])
def test_successful_affinity_skips_cpuset_discovery_and_reads(monkeypatch, version, bad_cpuset):
    membership = "3:cpu,cpuset:/job" if version == "v1" else "0::/job"
    kind = "cgroup" if version == "v1" else "cgroup2"
    files = (
        {
            "/cg/job/cpu.cfs_quota_us": "-1",
            "/cg/cpu.cfs_quota_us": "-1",
            "/cg/job/cpuset.effective_cpus": bad_cpuset,
            "/cg/cpuset.effective_cpus": bad_cpuset,
        }
        if version == "v1"
        else {
            "/cg/job/cpu.max": "max 100000",
            "/cg/cpu.max": "max 100000",
            "/cg/job/cpuset.cpus.effective": bad_cpuset,
            "/cg/cpuset.cpus.effective": bad_cpuset,
        }
    )
    _hierarchy(monkeypatch, membership, [(kind, "/", "/cg", "rw,cpu,cpuset")], files)
    monkeypatch.setattr(probe.os, "sched_getaffinity", lambda _pid: {2, 6})
    controllers = []
    controller_paths = probe._controller_paths
    read_cgroup_file = probe._read_cgroup_file

    def capture_controller(controller, discovery, issues):
        controllers.append(controller)
        return controller_paths(controller, discovery, issues)

    def read_without_cpuset(path):
        assert not path.name.startswith("cpuset.")
        return read_cgroup_file(path)

    monkeypatch.setattr(probe, "_controller_paths", capture_controller)
    monkeypatch.setattr(probe, "_read_cgroup_file", read_without_cpuset)

    issues = set()
    assert probe.probe_cpu(issues=issues) == {"groups": [{"units": "2"}]}
    assert controllers == ["cpu"]
    assert not issues


def test_affinity_selected_cpus_are_still_capped_by_fractional_quota(monkeypatch):
    monkeypatch.setattr(probe.os, "sched_getaffinity", lambda _pid: {2, 6})
    _hierarchy(
        monkeypatch,
        "0::/job",
        [("cgroup2", "/", "/cg", "rw")],
        {"/cg/job/cpu.max": "50000 100000", "/cg/cpu.max": "max 100000"},
    )

    issues = set()
    assert probe.probe_cpu(issues=issues) == {"groups": [{"units": "0.5"}]}
    assert not issues


@pytest.mark.parametrize("quota,cpuset,expected", [(16, None, "8"), (2, None, "2"), (16, "0-3", "4")])
def test_affinity_failure_still_bounds_quota_by_known_online_cpus(monkeypatch, quota, cpuset, expected):
    def denied(_pid):
        raise OSError("affinity cannot be observed")

    monkeypatch.setattr(probe.os, "sched_getaffinity", denied)
    files = {"/cg/job/cpu.max": f"{quota * 100000} 100000"}
    if cpuset is not None:
        files["/cg/job/cpuset.cpus.effective"] = cpuset
    _hierarchy(monkeypatch, "0::/job", [("cgroup2", "/", "/cg", "rw")], files)

    issues = set()
    assert probe.probe_cpu(issues=issues) == {"groups": [{"units": expected}]}
    assert issues == {"observation_incomplete"}


@pytest.mark.parametrize("filename", ["cpuset.effective_cpus", "cpuset.cpus"])
def test_affinity_failure_uses_readable_v1_cpuset_fallback(monkeypatch, filename):
    monkeypatch.setattr(probe.os, "sched_getaffinity", lambda _pid: set())
    _hierarchy(
        monkeypatch,
        "3:cpu,cpuset:/job",
        [("cgroup", "/", "/cg", "rw,cpu,cpuset")],
        {
            f"/cg/job/{filename}": "0-3",
            "/cg/cpuset.cpus": "0-7",
            "/cg/job/cpu.cfs_quota_us": "-1",
            "/cg/cpu.cfs_quota_us": "-1",
        },
    )

    issues = set()
    assert probe.probe_cpu(issues=issues) == {"groups": [{"units": "4"}]}
    assert issues == {"observation_incomplete"}


def test_affinity_and_sysconf_failure_uses_known_cpu_count(monkeypatch):
    def unavailable(_value):
        raise OSError("counter cannot be observed")

    monkeypatch.setattr(probe.os, "sched_getaffinity", unavailable)
    monkeypatch.setattr(probe.os, "sysconf", unavailable)
    monkeypatch.setattr(probe.os, "cpu_count", lambda: 6)
    _hierarchy(monkeypatch, "0::/job", [("cgroup2", "/", "/cg", "rw")], {"/cg/job/cpu.max": "1600000 100000"})

    issues = set()
    assert probe.probe_cpu(issues=issues) == {"groups": [{"units": "6"}]}
    assert issues == {"observation_incomplete"}


@pytest.mark.parametrize("unknown", [None, -1, 0, "8", True])
def test_unusable_sysconf_cpu_count_tries_cpu_count(monkeypatch, unknown):
    monkeypatch.setattr(probe.os, "sched_getaffinity", lambda _pid: set())
    monkeypatch.setattr(probe.os, "sysconf", lambda _key: unknown)
    monkeypatch.setattr(probe.os, "cpu_count", lambda: 6)
    _hierarchy(monkeypatch, "", [], {})

    issues = set()
    assert probe.probe_cpu(issues=issues) == {"groups": [{"units": "6"}]}
    assert issues == {"observation_incomplete"}


@pytest.mark.parametrize("unknown", [None, -1, 0])
def test_no_known_cpu_count_or_limits_is_unavailable(monkeypatch, unknown):
    monkeypatch.setattr(probe.os, "sched_getaffinity", lambda _pid: set())
    monkeypatch.setattr(probe.os, "sysconf", lambda _key: unknown)
    monkeypatch.setattr(probe.os, "cpu_count", lambda: None)
    _hierarchy(monkeypatch, "", [], {})

    assert probe.probe_cpu() is None


@pytest.mark.parametrize("path", ["/proc/self/cgroup", "/proc/self/mountinfo"])
@pytest.mark.parametrize("contents", [None, "malformed data"])
def test_unreadable_or_malformed_discovery_retains_machine_capacity(monkeypatch, path, contents):
    text = _hierarchy(monkeypatch, "0::/job", [("cgroup2", "/", "/cg", "rw")], {})
    text[Path(path)] = contents

    cpu_issues, memory_issues = set(), set()
    assert probe.probe_cpu(issues=cpu_issues) == {"groups": [{"units": "8"}]}
    assert probe.probe_memory(issues=memory_issues) == {"bytes": str(8 * _GIB)}
    assert cpu_issues == memory_issues == {"observation_incomplete"}


def test_recorded_controller_without_a_visible_mount_retains_capacity(monkeypatch):
    _hierarchy(monkeypatch, "3:cpu:/job", [], {})

    cpu_issues, memory_issues = set(), set()
    assert probe.probe_cpu(issues=cpu_issues) == {"groups": [{"units": "8"}]}
    assert probe.probe_memory(issues=memory_issues) == {"bytes": str(8 * _GIB)}
    assert cpu_issues == {"observation_incomplete"}
    assert not memory_issues


def test_unified_membership_without_a_visible_mount_retains_capacity(monkeypatch):
    _hierarchy(monkeypatch, "0::/job", [], {})

    issues = set()
    assert probe.probe_cpu(issues=issues) == {"groups": [{"units": "8"}]}
    assert probe.probe_memory(issues=issues) == {"bytes": str(8 * _GIB)}
    assert issues == {"observation_incomplete"}


def test_missing_v2_controller_files_are_not_unknown_v1_constraints(monkeypatch):
    _hierarchy(monkeypatch, "0::/job", [("cgroup2", "/", "/cg", "rw")], {})

    issues = set()
    assert probe.probe_cpu(issues=issues) == {"groups": [{"units": "8"}]}
    assert probe.probe_memory(issues=issues) == {"bytes": str(8 * _GIB)}
    assert not issues


@pytest.mark.parametrize("kind,controller", [("cgroup2", None), ("cgroup", "cpu")])
def test_discovery_skips_noncovering_mount_and_prefers_broadest_root(monkeypatch, kind, controller):
    _hierarchy(
        monkeypatch,
        "0::/group/job" if controller is None else "3:cpu:/group/job",
        [
            (kind, "/other", "/wrong", "rw,cpu"),
            (kind, "/group/job", "/restricted", "rw,cpu"),
            (kind, "/", "/full", "rw,cpu"),
        ],
        {},
    )

    assert probe._cgroup_location(controller) == (
        Path("/full"),
        Path("/full/group/job"),
        "v2" if controller is None else "v1",
    )


def test_restricted_mount_root_limits_ancestor_walk(monkeypatch):
    _hierarchy(
        monkeypatch,
        "0::/group/job",
        [("cgroup2", "/group", "/cg", "rw")],
        {"/cg/job/cpu.max": "400000 100000", "/cg/cpu.max": "200000 100000", "/cpu.max": "100000 100000"},
    )

    assert probe._cgroup_location(None) == (Path("/cg"), Path("/cg/job"), "v2")
    assert probe._ancestors(Path("/cg/job"), Path("/cg")) == [Path("/cg/job"), Path("/cg")]
    assert probe.probe_cpu() == {"groups": [{"units": "2"}]}


@pytest.mark.parametrize("membership", ["/outside/job", "/group/../../outside"])
def test_uncovered_or_parent_traversing_path_keeps_machine_fallback(monkeypatch, membership):
    _hierarchy(monkeypatch, f"0::{membership}", [("cgroup2", "/group", "/cg", "rw")], {})

    issues = set()
    assert probe.probe_cpu(issues=issues) == {"groups": [{"units": "8"}]}
    assert probe.probe_memory(issues=issues) == {"bytes": str(8 * _GIB)}
    assert issues == {"observation_incomplete"}


def test_mountinfo_escapes_are_decoded(monkeypatch):
    _hierarchy(monkeypatch, "0::/job", [("cgroup2", "/", "/cg\\040space", "rw")], {})

    assert probe._cgroup_location(None) == (Path("/cg space"), Path("/cg space/job"), "v2")


def test_v1_explicit_unlimited_memory_does_not_add_a_false_limit(monkeypatch):
    long_max = (1 << (ctypes.sizeof(ctypes.c_long) * 8 - 1)) - 1
    sentinel = (long_max // 4096) * 4096
    _hierarchy(
        monkeypatch,
        "4:memory:/",
        [("cgroup", "/", "/memory", "rw,memory")],
        {"/memory/memory.limit_in_bytes": str(sentinel)},
    )

    issues = set()
    assert probe.probe_memory(issues=issues) == {"bytes": str(8 * _GIB)}
    assert not issues


def test_non_linux_capacity_remains_unavailable(monkeypatch):
    monkeypatch.setattr(probe.platform, "system", lambda: "Darwin")

    assert probe.probe_cpu() is None
    assert probe.probe_memory() is None


@pytest.mark.parametrize(
    "membership,mounts,files",
    [
        (
            "0::/job",
            [("cgroup2", "/", "/cg", "rw")],
            {"/cg/job/cpu.max": "200000 100000", "/cg/job/memory.max": str(_GIB)},
        ),
        (
            "3:cpu,memory:/job",
            [("cgroup", "/", "/cg", "rw,cpu,memory")],
            {
                "/cg/job/cpu.cfs_quota_us": "200000",
                "/cg/job/cpu.cfs_period_us": "100000",
                "/cg/cpu.cfs_quota_us": "-1",
                "/cg/cpu.cfs_period_us": "100000",
                "/cg/job/memory.limit_in_bytes": str(_GIB),
                "/cg/memory.limit_in_bytes": str(8 * _GIB),
            },
        ),
        (
            "0::/job\n3:memory:/job",
            [("cgroup2", "/", "/cg", "rw"), ("cgroup", "/", "/memory", "rw,memory")],
            {
                "/cg/job/cpu.max": "200000 100000",
                "/memory/job/memory.limit_in_bytes": str(_GIB),
                "/memory/memory.limit_in_bytes": str(8 * _GIB),
            },
        ),
    ],
)
def test_combined_and_standalone_capacity_match_and_share_discovery(monkeypatch, membership, mounts, files):
    _hierarchy(monkeypatch, membership, mounts, files)
    expected = {"cpu": probe.probe_cpu(), "memory": probe.probe_memory()}
    reads = []
    read = probe._read_text

    def counted_read(path):
        reads.append(path)
        return read(path)

    monkeypatch.setattr(probe, "_read_text", counted_read)
    monkeypatch.setattr(collector.gpu_probe_process, "probe_gpu_isolated", lambda: None)

    assert collector.probe_capacity() == expected
    assert reads.count(Path("/proc/self/cgroup")) == 1
    assert reads.count(Path("/proc/self/mountinfo")) == 1
    assert reads.count(Path("/proc/cpuinfo")) == 1


def test_later_capacity_probe_rediscovers_membership(monkeypatch):
    text = _hierarchy(
        monkeypatch,
        "0::/first",
        [("cgroup2", "/", "/cg", "rw")],
        {
            "/cg/first/cpu.max": "100000 100000",
            "/cg/second/cpu.max": "200000 100000",
            "/cg/first/memory.max": str(_GIB),
            "/cg/second/memory.max": str(2 * _GIB),
        },
    )
    monkeypatch.setattr(collector.gpu_probe_process, "probe_gpu_isolated", lambda: None)

    assert collector.probe_capacity() == {"cpu": {"groups": [{"units": "1"}]}, "memory": {"bytes": str(_GIB)}}
    text[Path("/proc/self/cgroup")] = "0::/second"
    assert collector.probe_capacity() == {
        "cpu": {"groups": [{"units": "2"}]},
        "memory": {"bytes": str(2 * _GIB)},
    }


def test_memory_fallback_keeps_cpu_in_shared_snapshot(monkeypatch):
    _hierarchy(
        monkeypatch,
        "0::/job",
        [("cgroup2", "/", "/cg", "rw")],
        {"/cg/job/cpu.max": "200000 100000", "/cg/job/memory.max": "bad"},
    )
    monkeypatch.setattr(collector.gpu_probe_process, "probe_gpu_isolated", lambda: None)

    assert collector.probe_capacity() == {
        "cpu": {"groups": [{"units": "2"}]},
        "memory": {"bytes": str(8 * _GIB)},
        "status": "partial",
        "issues": ["observation_incomplete"],
    }


def test_bad_leaf_and_parent_limits_do_not_hide_other_readable_ancestors(monkeypatch):
    _hierarchy(
        monkeypatch,
        "0::/outer/inner/job",
        [("cgroup2", "/", "/cg", "rw")],
        {
            "/cg/outer/inner/job/cpu.max": probe._CGROUP_FILE_UNREADABLE,
            "/cg/outer/inner/cpu.max": "150000 100000",
            "/cg/outer/cpu.max": "bad",
            "/cg/cpu.max": "max 100000",
            "/cg/outer/inner/job/memory.max": "bad",
            "/cg/outer/inner/memory.max": str(2 * _GIB),
            "/cg/outer/memory.max": probe._CGROUP_FILE_UNREADABLE,
            "/cg/memory.max": str(4 * _GIB),
        },
    )

    issues = set()
    assert probe.probe_cpu(issues=issues) == {"groups": [{"units": "1.5"}]}
    assert probe.probe_memory(issues=issues) == {"bytes": str(2 * _GIB)}
    assert issues == {"observation_incomplete"}


@pytest.mark.parametrize(
    "files,expected",
    [
        ({"cpuset.cpus.effective": "bad", "cpu.max": "200000 100000"}, "2"),
        ({"cpuset.cpus.effective": "0-2", "cpu.max": "bad"}, "3"),
    ],
)
def test_failed_cpu_controller_does_not_hide_other_controller(monkeypatch, files, expected):
    monkeypatch.setattr(probe.os, "sched_getaffinity", lambda _pid: set())
    _hierarchy(
        monkeypatch,
        "0::/job",
        [("cgroup2", "/", "/cg", "rw")],
        {f"/cg/job/{name}": value for name, value in files.items()},
    )

    issues = set()
    assert probe.probe_cpu(issues=issues) == {"groups": [{"units": expected}]}
    assert issues == {"observation_incomplete"}


@pytest.mark.parametrize("period", ["bad", probe._CGROUP_FILE_UNREADABLE, probe._CGROUP_FILE_MISSING])
def test_bad_v1_quota_period_keeps_affinity(monkeypatch, period):
    _hierarchy(
        monkeypatch,
        "3:cpu:/",
        [("cgroup", "/", "/cg", "rw,cpu")],
        {"/cg/cpu.cfs_quota_us": "200000", "/cg/cpu.cfs_period_us": period},
    )

    issues = set()
    assert probe.probe_cpu(issues=issues) == {"groups": [{"units": "8"}]}
    assert issues == {"observation_incomplete"}


def test_v1_unlimited_quota_needs_no_period(monkeypatch):
    _hierarchy(
        monkeypatch,
        "3:cpu:/",
        [("cgroup", "/", "/cg", "rw,cpu")],
        {"/cg/cpu.cfs_quota_us": "-1"},
    )

    issues = set()
    assert probe.probe_cpu(issues=issues) == {"groups": [{"units": "8"}]}
    assert not issues


@pytest.mark.parametrize("unknown", [None, 0, -1, "4096", True])
def test_unknown_physical_memory_keeps_a_readable_cgroup_bound(monkeypatch, unknown):
    monkeypatch.setattr(probe.os, "sysconf", lambda _key: unknown)
    _hierarchy(
        monkeypatch,
        "0::/job",
        [("cgroup2", "/", "/cg", "rw")],
        {"/cg/job/memory.max": str(_GIB)},
    )

    issues = set()
    assert probe.probe_memory(issues=issues) == {"bytes": str(_GIB)}
    assert issues == {"observation_incomplete"}


def test_memory_without_any_usable_bound_is_unavailable(monkeypatch):
    monkeypatch.setattr(probe.os, "sysconf", lambda _key: None)
    _hierarchy(monkeypatch, "0::/job", [("cgroup2", "/", "/cg", "rw")], {"/cg/job/memory.max": "bad"})

    issues = set()
    assert probe.probe_memory(issues=issues) is None
    assert issues == {"observation_incomplete"}


@pytest.mark.parametrize("helper", ["_cpu_models", "_cpu_identity", "_heterogeneous_cpu_groups"])
def test_model_lookup_exceptions_never_erase_capacity(monkeypatch, helper):
    _hierarchy(monkeypatch, "0::/job", [("cgroup2", "/", "/cg", "rw")], {})

    def broken_lookup(*_args):
        raise OSError("optional model description cannot be read")

    monkeypatch.setattr(probe, helper, broken_lookup)
    issues = set()
    assert probe.probe_cpu(issues=issues) == {"groups": [{"units": "8"}]}
    assert not issues


@pytest.mark.parametrize("quota", ["max", "400000"])
def test_cpu_model_parser_is_shared_by_grouping_and_identity(monkeypatch, quota):
    text = _hierarchy(
        monkeypatch,
        "0::/job",
        [("cgroup2", "/", "/cg", "rw")],
        {"/cg/job/cpu.max": f"{quota} 100000"},
    )
    text[Path("/proc/cpuinfo")] = "\n\n".join(
        f"processor : {index}\nmodel name : {'CPU A' if index < 4 else 'CPU B'}" for index in range(8)
    )
    monkeypatch.setattr(probe.platform, "machine", lambda: "x86_64")
    read = probe._read_text
    cpuinfo_reads = []

    def counted_read(path):
        if path == Path("/proc/cpuinfo"):
            cpuinfo_reads.append(path)
        return read(path)

    monkeypatch.setattr(probe, "_read_text", counted_read)
    result = probe.probe_cpu()

    if quota == "max":
        assert result == {
            "groups": [
                {"units": "4", "model": "CPU A", "architecture": "x86_64"},
                {"units": "4", "model": "CPU B", "architecture": "x86_64"},
            ]
        }
    else:
        # A fractional/binding quota cannot be assigned to specific models.
        assert result == {"groups": [{"units": "4", "architecture": "x86_64"}]}
    assert len(cpuinfo_reads) == 1


@pytest.mark.parametrize(
    "cpuinfo",
    [
        None,
        "processor : 0\nmodel name : CPU A",
        "\n\n".join(f"processor : {index}\nmodel name : CPU A" for index in range(8))
        + "\n\nprocessor : 0\nmodel name : CPU B",
    ],
)
def test_missing_or_ambiguous_model_never_erases_cpu_count(monkeypatch, cpuinfo):
    text = _hierarchy(monkeypatch, "0::/job", [("cgroup2", "/", "/cg", "rw")], {})
    text[Path("/proc/cpuinfo")] = cpuinfo

    assert probe.probe_cpu() == {"groups": [{"units": "8"}]}


@pytest.mark.parametrize("quota,expected", [("max", "8"), ("250000", "2.5")])
def test_homogeneous_model_is_preserved_under_a_quota(monkeypatch, quota, expected):
    text = _hierarchy(
        monkeypatch,
        "0::/job",
        [("cgroup2", "/", "/cg", "rw")],
        {"/cg/job/cpu.max": f"{quota} 100000"},
    )
    text[Path("/proc/cpuinfo")] = "\n\n".join(f"processor : {index}\nmodel name : CPU A" for index in range(8))
    monkeypatch.setattr(probe.platform, "machine", lambda: "x86_64")

    assert probe.probe_cpu() == {"groups": [{"units": expected, "model": "CPU A", "architecture": "x86_64"}]}
