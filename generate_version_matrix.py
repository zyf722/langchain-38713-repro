from __future__ import annotations

import argparse
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, TypedDict, cast

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

CORE_PACKAGE = "langchain-core"
GRAPH_PACKAGE = "langgraph"
SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
CORE_PACKAGE_NAME = canonicalize_name(CORE_PACKAGE)

VersionString: TypeAlias = str
JsonObject: TypeAlias = dict[str, Any]
MarkerEnvironment: TypeAlias = dict[str, str]


class PyPIFile(TypedDict, total=False):
    yanked: bool
    requires_python: str | None


class PyPIIndexJson(TypedDict):
    releases: dict[VersionString, list[PyPIFile]]


class PyPIInfoJson(TypedDict, total=False):
    requires_dist: list[str] | None


class PyPIReleaseJson(TypedDict):
    info: PyPIInfoJson


class VersionPairJson(TypedDict):
    core: VersionString
    lg: VersionString


PackageVersionsJson = TypedDict(
    "PackageVersionsJson",
    {
        "langchain-core": list[VersionString],
        "langgraph": list[VersionString],
    },
)


class VersionMatrixJson(TypedDict):
    python_version: VersionString
    packages: PackageVersionsJson
    pairs: list[VersionPairJson]


@dataclass(frozen=True)
class MatrixConfig:
    core_count: int
    graph_count: int
    python_version: VersionString


@dataclass(frozen=True)
class CliArgs:
    config: MatrixConfig
    output: Path


@dataclass(frozen=True)
class VersionPair:
    core: VersionString
    lg: VersionString


@dataclass(frozen=True)
class VersionMatrix:
    python_version: VersionString
    core_versions: list[VersionString]
    graph_versions: list[VersionString]
    pairs: list[VersionPair]

    def to_json(self) -> VersionMatrixJson:
        return {
            "python_version": self.python_version,
            "packages": {
                CORE_PACKAGE: self.core_versions,
                GRAPH_PACKAGE: self.graph_versions,
            },
            "pairs": [{"core": pair.core, "lg": pair.lg} for pair in self.pairs],
        }


def version_key(version: VersionString) -> tuple[int, int, int]:
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)


def marker_environment(python_version: VersionString) -> MarkerEnvironment:
    env = {key: str(value) for key, value in default_environment().items()}
    parts = python_version.split(".")
    env["python_version"] = ".".join(parts[:2])
    env["python_full_version"] = (
        python_version if len(parts) >= 3 else f"{python_version}.0"
    )
    return env


def supports_python(requires_python: str | None, python_version: VersionString) -> bool:
    if not requires_python:
        return True
    try:
        return SpecifierSet(requires_python).contains(
            Version(python_version), prereleases=True
        )
    except (InvalidSpecifier, InvalidVersion):
        return False


def has_compatible_file(files: list[PyPIFile], python_version: VersionString) -> bool:
    return any(
        not file.get("yanked", False)
        and supports_python(file.get("requires_python"), python_version)
        for file in files
    )


def fetch_json(url: str) -> JsonObject:
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return payload


def fetch_package_index(package: str) -> PyPIIndexJson:
    return cast(PyPIIndexJson, fetch_json(f"https://pypi.org/pypi/{package}/json"))


def fetch_package_release(package: str, version: VersionString) -> PyPIReleaseJson:
    return cast(
        PyPIReleaseJson, fetch_json(f"https://pypi.org/pypi/{package}/{version}/json")
    )


def recent_versions(
    package: str, count: int, python_version: VersionString
) -> list[VersionString]:
    payload = fetch_package_index(package)

    versions: list[VersionString] = []
    for version, files in payload["releases"].items():
        if not SEMVER_PATTERN.match(version):
            continue
        if not files:
            continue
        if not has_compatible_file(files, python_version):
            continue
        versions.append(version)

    return sorted(versions, key=version_key)[-count:]


def required_core_specifier(
    graph_version: VersionString, env: MarkerEnvironment
) -> SpecifierSet:
    payload = fetch_package_release(GRAPH_PACKAGE, graph_version)
    requirements = payload["info"].get("requires_dist") or []
    specifiers: list[str] = []

    for requirement_text in requirements:
        requirement = Requirement(requirement_text)
        if canonicalize_name(requirement.name) != CORE_PACKAGE_NAME:
            continue
        if requirement.marker is not None and not requirement.marker.evaluate(
            environment=env
        ):
            continue
        if str(requirement.specifier):
            specifiers.append(str(requirement.specifier))

    return SpecifierSet(",".join(specifiers))


def is_compatible(core_version: VersionString, core_specifier: SpecifierSet) -> bool:
    return core_specifier.contains(Version(core_version), prereleases=True)


def build_matrix(config: MatrixConfig) -> VersionMatrix:
    env = marker_environment(config.python_version)
    core_versions = recent_versions(
        CORE_PACKAGE, config.core_count, config.python_version
    )
    graph_versions = recent_versions(
        GRAPH_PACKAGE, config.graph_count, config.python_version
    )
    core_specifiers = {
        graph_version: required_core_specifier(graph_version, env)
        for graph_version in graph_versions
    }

    pairs: list[VersionPair] = []
    for core_version in core_versions:
        for graph_version in graph_versions:
            core_specifier = core_specifiers[graph_version]
            print(
                f"checking {CORE_PACKAGE}=={core_version}, {GRAPH_PACKAGE}=={graph_version} "
                f"against {CORE_PACKAGE}{core_specifier or ''}...",
                flush=True,
            )
            if is_compatible(core_version, core_specifier):
                pairs.append(VersionPair(core=core_version, lg=graph_version))

    if not pairs:
        raise SystemExit("No compatible version pairs found.")

    return VersionMatrix(
        python_version=config.python_version,
        core_versions=core_versions,
        graph_versions=graph_versions,
        pairs=pairs,
    )


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Generate the committed GitHub Actions version matrix."
    )
    parser.add_argument(
        "--n-core",
        type=int,
        default=8,
        help=f"Number of recent {CORE_PACKAGE} releases to check.",
    )
    parser.add_argument(
        "--n-lg",
        type=int,
        default=8,
        help=f"Number of recent {GRAPH_PACKAGE} releases to check.",
    )
    parser.add_argument(
        "--python-version",
        default="3.12",
        help="Python version used to filter PyPI releases.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("version-pairs.json"),
        help="Path to write the matrix JSON.",
    )
    namespace = parser.parse_args()
    return CliArgs(
        config=MatrixConfig(
            core_count=namespace.n_core,
            graph_count=namespace.n_lg,
            python_version=namespace.python_version,
        ),
        output=namespace.output,
    )


def main() -> None:
    args = parse_args()
    matrix = build_matrix(args.config)
    matrix_json = matrix.to_json()
    args.output.write_text(json.dumps(matrix_json, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(matrix.pairs)} compatible pairs to {args.output}")


if __name__ == "__main__":
    main()
