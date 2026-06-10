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

"""Host-side Docker image build orchestration."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..agents.base import AgentAdapter
from ..agents.config import ConfigurableAgentAdapter
from ..agents.registry import DEFAULT_BENCHMARK_AGENT, load_agent_adapter
from ..sdks.base import SdkAdapter, SdkSkillsSetup, SdkSource, SdkWheelBuild, SdkWheelVariant
from ..sdks.config import ConfigurableSdkAdapter
from ..sdks.registry import DEFAULT_BENCHMARK_SDK, load_sdk_adapter
from .common import SCRIPT_DIR, emit

REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_UV_IMAGE = "ghcr.io/astral-sh/uv:0.11.19"
DEFAULT_NODE_IMAGE = "node:22.16.0-bookworm-slim"


@dataclass(frozen=True)
class PreparedSdkWheel:
    wheel: Path
    source_type: str
    source_path: Path


def looks_like_profile_path(value: str) -> bool:
    candidate = Path(value).expanduser()
    return candidate.is_absolute() or len(candidate.parts) > 1 or candidate.suffix in {".yaml", ".yml"}


def load_sdk_profile(profile: str) -> SdkAdapter:
    candidate = Path(profile).expanduser()
    if candidate.is_file():
        return ConfigurableSdkAdapter(candidate.resolve())
    if looks_like_profile_path(profile):
        raise SystemExit(f"SDK profile file does not exist: {candidate}")
    try:
        return load_sdk_adapter(profile)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def load_agent_profile(profile: str) -> AgentAdapter:
    candidate = Path(profile).expanduser()
    if candidate.is_file():
        return ConfigurableAgentAdapter(candidate.resolve())
    if looks_like_profile_path(profile):
        raise SystemExit(f"Agent profile file does not exist: {candidate}")
    try:
        return load_agent_adapter(profile)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def canonical_dir(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if not expanded.is_dir():
        raise SystemExit(f"{label} directory does not exist: {expanded}")
    return expanded.resolve()


def assert_sdk_repo_not_in_harness_source(path: Path) -> None:
    try:
        path.relative_to(SCRIPT_DIR)
    except ValueError:
        return
    raise SystemExit(
        "SDK profile source.path must not be inside assist_tools/skills_benchmark. "
        "Only built wheel artifacts are staged into the Docker build context."
    )


def repo_has_markers(path: Path, markers: tuple[str, ...]) -> bool:
    for marker in markers:
        marker_path = path / marker.rstrip("/")
        if marker.endswith("/"):
            if not marker_path.is_dir():
                return False
        elif not marker_path.exists():
            return False
    return True


def resolve_sdk_source(sdk: SdkAdapter) -> SdkSource:
    source = sdk.source(repo_root=REPO_ROOT, home=Path.home())
    if source.source_type == "repo":
        if source.repo_path is None:
            raise SystemExit(f"{sdk.display_name} SDK profile source.type=repo requires source.path")
        repo = canonical_dir(source.repo_path, "SDK profile source.path")
        if not repo_has_markers(repo, source.repo_markers):
            markers = ", ".join(source.repo_markers)
            raise SystemExit(
                f"SDK profile source.path does not look like a {sdk.display_name} checkout: {repo}. "
                f"Expected marker(s): {markers}."
            )
        assert_sdk_repo_not_in_harness_source(repo)
        return SdkSource(source_type="repo", repo_path=repo, repo_markers=source.repo_markers)

    if source.source_type == "wheels":
        raw_wheels = source.wheel_paths or {}
        wheel_paths = {}
        for variant_name in ("skills", "baseline"):
            wheel = raw_wheels.get(variant_name)
            if wheel is None:
                raise SystemExit(f"{sdk.display_name} SDK profile source.wheels.{variant_name} is required")
            expanded = wheel.expanduser()
            if not expanded.is_file():
                raise SystemExit(f"SDK profile source.wheels.{variant_name} file does not exist: {expanded}")
            if expanded.suffix != ".whl":
                raise SystemExit(f"SDK profile source.wheels.{variant_name} must be a .whl file: {expanded}")
            wheel_paths[variant_name] = expanded.resolve()
        return SdkSource(source_type="wheels", wheel_paths=wheel_paths)

    raise SystemExit(f"Unsupported SDK profile source.type={source.source_type!r}")


def latest_sdk_wheel(search_dir: Path, include_globs: tuple[str, ...], exclude_globs: tuple[str, ...]) -> Path | None:
    wheels: dict[Path, None] = {}
    for pattern in include_globs:
        for wheel in search_dir.glob(pattern):
            if wheel.suffix == ".whl":
                wheels[wheel] = None
    matches: list[tuple[Path, float]] = []
    for wheel in wheels:
        if any(fnmatch.fnmatch(wheel.name, pattern) for pattern in exclude_globs):
            continue
        try:
            matches.append((wheel, wheel.stat().st_mtime))
        except OSError:
            continue
    if not matches:
        return None
    return max(matches, key=lambda item: item[1])[0]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def clean_wheels(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for wheel in out_dir.glob("*.whl"):
        wheel.unlink()


def clean_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def copy_directory_contents(src: Path, dst: Path) -> None:
    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {name for name in names if name == "__pycache__" or name.endswith(".pyc")}

    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target, ignore=ignore)
        else:
            shutil.copy2(child, target)


def resolve_sdk_skills_setup(sdk: SdkAdapter) -> SdkSkillsSetup:
    setup = sdk.skills_setup(repo_root=REPO_ROOT, home=Path.home())
    if setup.setup_type == "copy":
        if setup.source_path is None:
            raise SystemExit(f"{sdk.display_name} SDK profile skills.setup.source_path is required")
        source_path = canonical_dir(setup.source_path, "SDK profile skills.setup.source_path")
        return SdkSkillsSetup(
            setup_type=setup.setup_type,
            source_path=source_path,
            install_command=setup.install_command,
            list_command=setup.list_command,
            install_output=setup.install_output,
            list_output=setup.list_output,
            expected_source=setup.expected_source,
        )
    return setup


def stage_sdk_skills_setup(context: Path, setup: SdkSkillsSetup) -> None:
    target = context / "sdk_skills"
    clean_directory(target)
    if setup.setup_type != "copy":
        return
    if setup.source_path is None:
        raise SystemExit("SDK profile skills.setup.source_path is required for copy setup")
    copy_directory_contents(setup.source_path, target)
    emit(f"Using SDK skills folder: {setup.source_path}")


def stage_configured_wheel(
    *,
    wheel: Path,
    sdk: SdkAdapter,
    variant: SdkWheelVariant,
    out_dir: Path,
) -> PreparedSdkWheel:
    if not any(fnmatch.fnmatch(wheel.name, pattern) for pattern in variant.wheel_globs):
        raise SystemExit(
            f"Configured {sdk.package_name} {variant.label} wheel {wheel.name!r} does not match "
            f"expected pattern(s): {variant.wheel_globs}."
        )
    if any(fnmatch.fnmatch(wheel.name, pattern) for pattern in variant.wheel_exclude_globs):
        raise SystemExit(
            f"Configured {sdk.package_name} {variant.label} wheel {wheel.name!r} matches excluded "
            f"pattern(s): {variant.wheel_exclude_globs}."
        )
    clean_wheels(out_dir)
    target = out_dir / wheel.name
    shutil.copy2(wheel, target)
    emit(f"Using configured {variant.label} wheel: {wheel}")
    return PreparedSdkWheel(wheel=target, source_type="wheels", source_path=wheel)


def build_sdk_wheel_from_repo(
    *,
    repo: Path,
    sdk: SdkAdapter,
    variant: SdkWheelVariant,
    out_dir: Path,
) -> PreparedSdkWheel:
    clean_wheels(out_dir)
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit(
            "Host uv is required to build SDK wheels. "
            "To use existing wheels, set source.type: wheels and build.type: provided_wheels in the SDK profile."
        )

    emit(f"=== Building {sdk.package_name} {variant.label} wheel ===")
    env = {**os.environ}
    if sdk.build_env_name:
        env[sdk.build_env_name] = variant.build_env_value
    status = subprocess.call([uv, "build", "--wheel", "--out-dir", str(out_dir)], cwd=repo, env=env)
    if status != 0:
        raise SystemExit(status)

    wheel = latest_sdk_wheel(out_dir, variant.wheel_globs, variant.wheel_exclude_globs)
    if wheel is None:
        raise SystemExit(
            f"No {sdk.package_name} {variant.label} wheel found under {out_dir}. "
            f"Expected a wheel matching {variant.wheel_globs} excluding {variant.wheel_exclude_globs}."
        )
    return PreparedSdkWheel(wheel=wheel, source_type="repo", source_path=repo)


def prepare_sdk_wheel(
    *,
    source: SdkSource,
    wheel_build: SdkWheelBuild,
    sdk: SdkAdapter,
    variant: SdkWheelVariant,
    out_dir: Path,
) -> PreparedSdkWheel:
    if wheel_build.build_type == "uv_wheel":
        if source.repo_path is None:
            raise SystemExit(f"{sdk.display_name} SDK profile build.type=uv_wheel requires source.type=repo")
        return build_sdk_wheel_from_repo(repo=source.repo_path, sdk=sdk, variant=variant, out_dir=out_dir)
    if wheel_build.build_type == "provided_wheels":
        wheel_paths = source.wheel_paths or {}
        wheel = wheel_paths.get(variant.name)
        if wheel is None:
            raise SystemExit(f"{sdk.display_name} SDK profile build.type=provided_wheels requires source.wheels")
        return stage_configured_wheel(wheel=wheel, sdk=sdk, variant=variant, out_dir=out_dir)
    raise SystemExit(f"Unsupported SDK profile build.type={wheel_build.build_type!r}")


def write_wheel_metadata(
    *,
    sdk: SdkAdapter,
    variant: SdkWheelVariant,
    wheel_build: SdkWheelBuild,
    prepared: PreparedSdkWheel,
    out_dir: Path,
) -> None:
    payload = {
        "build_env": {"name": sdk.build_env_name, "value": variant.build_env_value} if sdk.build_env_name else None,
        "build_type": wheel_build.build_type,
        "filename": prepared.wheel.name,
        "git_commit": git_commit(prepared.source_path) if prepared.source_type == "repo" else None,
        "import_name": sdk.import_name,
        "package_name": sdk.package_name,
        "sdk": sdk.metadata(),
        "sdk_name": sdk.name,
        "sha256": file_sha256(prepared.wheel),
        "source_path": str(prepared.source_path),
        "source_type": prepared.source_type,
        "variant": variant.name,
    }
    metadata = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    (out_dir / "sdk_wheel_metadata.json").write_text(metadata, encoding="utf-8")


def copy_harness(src: Path, dst: Path) -> None:
    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {name for name in names if name == "__pycache__" or name.endswith(".pyc")}

    shutil.copytree(src, dst, ignore=ignore)


def copy_harness_package(context: Path) -> None:
    package_root = context / "assist_tools" / "skills_benchmark" / "skills"
    package_root.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "assist_tools" / "__init__.py", context / "assist_tools" / "__init__.py")
    shutil.copy2(SCRIPT_DIR / "__init__.py", context / "assist_tools" / "skills_benchmark" / "__init__.py")
    copy_harness(SCRIPT_DIR / "config", context / "assist_tools" / "skills_benchmark" / "config")
    shutil.copy2(SCRIPT_DIR / "skills" / "__init__.py", package_root / "__init__.py")
    copy_harness(SCRIPT_DIR / "skills" / "harness", package_root / "harness")


def prepare_build_context() -> Path:
    context = Path(tempfile.mkdtemp(prefix="skills-benchmark-build-context.", dir=os.environ.get("TMPDIR") or None))
    try:
        (context / "dist" / "skills").mkdir(parents=True)
        (context / "dist" / "baseline").mkdir(parents=True)
        (context / "sdk_skills").mkdir(parents=True)
        shutil.copy2(SCRIPT_DIR / "docker" / "Dockerfile", context / "Dockerfile")
        copy_harness_package(context)
        shutil.copy2(SCRIPT_DIR / "docker" / "build_context.dockerignore", context / ".dockerignore")
    except BaseException:
        shutil.rmtree(context, ignore_errors=True)
        raise
    return context


def docker_build(
    *,
    image: str,
    target: str,
    context: Path,
    uv_image: str,
    node_image: str,
    sdk_build_args: dict[str, str],
    agent_build_args: dict[str, str],
    no_cache: bool,
) -> None:
    cache_args = ["--no-cache"] if no_cache else []
    rendered_sdk_build_args = render_docker_build_args(sdk_build_args, allow_value_equals=True)
    rendered_agent_build_args = render_agent_build_args(agent_build_args)
    status = subprocess.call(
        [
            "docker",
            "build",
            *cache_args,
            "--target",
            target,
            "--build-arg",
            f"UV_IMAGE={uv_image}",
            "--build-arg",
            f"NODE_IMAGE={node_image}",
            *rendered_sdk_build_args,
            *rendered_agent_build_args,
            "-t",
            image,
            str(context),
        ]
    )
    if status != 0:
        raise SystemExit(status)


def render_docker_build_args(build_args: dict[str, str], *, allow_value_equals: bool = False) -> list[str]:
    rendered_build_args = []
    for key, value in sorted(build_args.items()):
        if "=" in str(key):
            raise ValueError(f"Docker build arg key must not contain '=': {key}")
        if "=" in str(value) and not allow_value_equals:
            raise ValueError(f"Docker build arg {key} value must not contain '='")
        rendered_build_args.extend(["--build-arg", f"{key}={value}"])
    return rendered_build_args


def render_agent_build_args(agent_build_args: dict[str, str]) -> list[str]:
    return render_docker_build_args(agent_build_args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build agent skills benchmark Docker images.")
    parser.add_argument(
        "--sdk-profile",
        default=DEFAULT_BENCHMARK_SDK,
        help=f"SDK profile name or YAML path. Defaults to {DEFAULT_BENCHMARK_SDK}.",
    )
    parser.add_argument(
        "--agent-profile",
        default=DEFAULT_BENCHMARK_AGENT,
        help=f"Agent profile name or YAML path. Defaults to {DEFAULT_BENCHMARK_AGENT}.",
    )
    parser.add_argument("--skip-skills-image", action="store_true", help="Do not build the skills image.")
    parser.add_argument("--skip-baseline-image", action="store_true", help="Do not build the baseline image.")
    parser.add_argument("--no-cache", action="store_true", help="Pass --no-cache to docker build.")
    parser.add_argument("--uv-image", default=DEFAULT_UV_IMAGE, help="uv image used as the Docker uv source stage.")
    parser.add_argument("--node-image", default=DEFAULT_NODE_IMAGE, help="Node runtime image used as the Docker base.")
    args = parser.parse_args(argv)

    try:
        adapter = load_agent_profile(args.agent_profile)
        sdk = load_sdk_profile(args.sdk_profile)
        wheel_build = sdk.wheel_build()
        skills_variant = sdk.wheel_variant("skills")
        baseline_variant = sdk.wheel_variant("baseline")
        skills_setup = resolve_sdk_skills_setup(sdk)
        targets = adapter.image_targets()
        sdk_build_args = sdk.docker_build_args()
        agent_build_args = adapter.build_args()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    image_name = targets.skills
    baseline_image_name = targets.baseline
    report_image_name = targets.report
    build_skills_image = not args.skip_skills_image
    build_baseline_image = not args.skip_baseline_image

    context = None
    try:
        emit("=== Preparing minimal Docker build context ===")
        context = prepare_build_context()
        emit(f"Agent profile: {args.agent_profile} -> {adapter.display_name} ({adapter.name})")
        emit(f"SDK profile: {args.sdk_profile} -> {sdk.display_name} ({sdk.package_name})")
        emit(f"SDK wheel build: {wheel_build.build_type}")
        if build_skills_image:
            emit(f"SDK skills setup: {skills_setup.setup_type}")
            stage_sdk_skills_setup(context, skills_setup)
        if build_skills_image or build_baseline_image:
            source = resolve_sdk_source(sdk)
            if source.source_type == "repo":
                emit(f"Using SDK repo: {source.repo_path}")
            else:
                emit("Using SDK wheels from profile.")

            if build_skills_image:
                skills_prepared = prepare_sdk_wheel(
                    source=source,
                    wheel_build=wheel_build,
                    sdk=sdk,
                    variant=skills_variant,
                    out_dir=context / "dist" / "skills",
                )
                emit(f"Using skills wheel: {skills_prepared.wheel.name}")
                write_wheel_metadata(
                    sdk=sdk,
                    variant=skills_variant,
                    wheel_build=wheel_build,
                    prepared=skills_prepared,
                    out_dir=context / "dist" / "skills",
                )

            if build_baseline_image:
                baseline_prepared = prepare_sdk_wheel(
                    source=source,
                    wheel_build=wheel_build,
                    sdk=sdk,
                    variant=baseline_variant,
                    out_dir=context / "dist" / "baseline",
                )
                emit(f"Using baseline wheel: {baseline_prepared.wheel.name}")
                write_wheel_metadata(
                    sdk=sdk,
                    variant=baseline_variant,
                    wheel_build=wheel_build,
                    prepared=baseline_prepared,
                    out_dir=context / "dist" / "baseline",
                )

        emit(f"Docker build context: {context}")
        emit(f"UV image: {args.uv_image}")
        emit(f"Node runtime image: {args.node_image}")
        for key, value in sorted(sdk_build_args.items()):
            emit(f"SDK build arg: {key}={value}")
        for key, value in sorted(agent_build_args.items()):
            emit(f"Agent build arg: {key}={value}")
        emit(f"Docker build no-cache: {str(args.no_cache).lower()}")
        if build_skills_image:
            emit(f"=== Building Docker skills image: {image_name} ===")
            docker_build(
                image=image_name,
                target="skills",
                context=context,
                uv_image=args.uv_image,
                node_image=args.node_image,
                sdk_build_args=sdk_build_args,
                agent_build_args=agent_build_args,
                no_cache=args.no_cache,
            )
        if build_baseline_image:
            emit(f"=== Building Docker baseline image: {baseline_image_name} ===")
            docker_build(
                image=baseline_image_name,
                target="baseline",
                context=context,
                uv_image=args.uv_image,
                node_image=args.node_image,
                sdk_build_args=sdk_build_args,
                agent_build_args=agent_build_args,
                no_cache=args.no_cache,
            )

        emit(f"Skills image: {image_name}")
        emit(f"Baseline image: {baseline_image_name}")
        emit(f"Report image: {report_image_name}")
        return 0
    finally:
        if context is not None:
            shutil.rmtree(context, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
