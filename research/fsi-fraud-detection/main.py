"""CLI tool for bulk synthetic payment dataset generation.

Reads per-site YAML configs from ``config/{site}.yml``, generates normal
payment data, injects anomalies per each ``dataset_generation_config`` entry,
applies fraud probability thinning, and writes CSV files.

Usage::

    # Generate datasets for specific sites
    python main.py -s site1 -s siteB -o output/ -S 42

    # Generate for all sites found in config/
    python main.py -o output/
"""

import argparse
import csv
import hashlib
import logging
import typing
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from data_generation.anomaly_transformers import (
    Type1Config,
    Type2Config,
    add_fraud_columns,
    apply_fraud_with_probability,
    inject_all,
)
from data_generation.attributes import (
    get_payment_amount_attributes,
    get_payment_core_attributes,
    get_per_participant_attributes,
)
from data_generation.dataset import generate
from data_generation.rng.uniform_distribution import UniformDistributionSamplingConfig
from data_generation.static_data import country_static_data
from data_generation.synthetic_data_provider import (
    FakerSyntheticDataProvider,
    RandomChoiceDataProvider,
    UniformDistributionDataProvider,
)

log = logging.getLogger(__name__)

PREFIXES = ("DEBITOR", "CREDITOR")
CONFIG_DIR = Path(__file__).parent / "config"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic payment datasets with anomaly injection.",
    )
    parser.add_argument(
        "-s",
        "--site",
        action="append",
        dest="sites",
        help=(
            "Site name to generate data for (repeatable). "
            "Maps to config/{site}.yml. If omitted, all sites in config/ are used."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Root directory for CSV output (default: output/).",
    )
    parser.add_argument(
        "-S",
        "--seed",
        type=int,
        default=42,
        help="Global RNG seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "-c",
        "--checksum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate SHA256 checksums for produced dataset files (default: True).",
    )
    parser.add_argument(
        "-U",
        "--generate-universal-set",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Combine all scaling datasets into a single universal dataset (default: True).",
    )
    parser.add_argument(
        "-F",
        "--universal-set-file-path",
        type=Path,
        default=None,
        help=(
            "File path for the universal scaling dataset. "
            "Relative paths are resolved against the output directory. "
            "(default: {output_dir}/universal_scaling_datasets_all_banks.csv)"
        ),
    )
    parser.add_argument(
        "-C",
        "--universal-set-sample-row-count",
        type=int,
        default=10_000,
        help="Max rows to sample per site for the universal dataset (default: 10000). Use 0 for no sampling.",
    )
    return parser.parse_args(argv)


def discover_sites() -> list[str]:
    """Return sorted site names derived from ``config/*.yml`` filenames."""
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yml"))


def load_site_config(site_name: str) -> dict:
    """Load and return the parsed YAML for a single site."""
    config_path = CONFIG_DIR / f"{site_name}.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}. Available sites: {discover_sites()}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_providers(seed: int):
    """Create provider instances and the auto-mapping dict."""
    faker_provider = FakerSyntheticDataProvider(seed=seed)
    random_choice_provider = RandomChoiceDataProvider(seed=seed)
    uniform_provider = UniformDistributionDataProvider(seed=seed)

    provider_map = {
        FakerSyntheticDataProvider: faker_provider,
        RandomChoiceDataProvider: random_choice_provider,
        UniformDistributionDataProvider: uniform_provider,
    }
    return provider_map


def build_graph_and_providers(provider_map):
    """Build the dependency graph and map each attribute to its provider."""
    graph = get_per_participant_attributes(PREFIXES)
    get_payment_core_attributes(PREFIXES, graph)
    get_payment_amount_attributes(PREFIXES, graph)

    providers = {}
    for attr in graph:
        hints = typing.get_type_hints(attr.attribute_data_provider)
        provider_type = hints["provider"]
        providers[attr] = provider_map[provider_type]

    return graph, providers


def generate_site_datasets(
    site_name: str,
    site_config: dict,
    output_dir: Path,
    seed: int,
) -> list[Path]:
    """Generate all datasets for a single site."""
    generated_files: list[Path] = []
    fields = site_config["anomaly_generation_config"]["field"]
    dataset_configs = site_config["dataset_generation_config"]

    # Build distribution configs from site fields
    tower_lat_cfg = UniformDistributionSamplingConfig(
        low=fields["tower_lat"]["distributions"][0]["low"],
        high=fields["tower_lat"]["distributions"][0]["high"],
    )
    tower_lon_cfg = UniformDistributionSamplingConfig(
        low=fields["tower_long"]["distributions"][0]["low"],
        high=fields["tower_long"]["distributions"][0]["high"],
    )
    personal_amount_cfg = fields["normal_personal_acc_amount"]["distributions"][0]
    business_amount_cfg = fields["normal_business_acc_amount"]["distributions"][0]

    # Build anomaly configs
    type1_cfg = Type1Config.from_site_fields(fields)
    type2_cfg = Type2Config.from_site_fields(fields)
    anomaly_configs = {
        "type1": {"config": type1_cfg},
        "type2": {"config": type2_cfg},
    }

    # Load static data
    static_data = country_static_data.load_static_data("~/.cache/fsi_static")

    site_dir = output_dir / site_name
    site_dir.mkdir(parents=True, exist_ok=True)

    for ds_idx, ds_cfg in enumerate(dataset_configs):
        rule_stack = ds_cfg["fraud_insertion_rule_stack"]
        num_datasets = ds_cfg.get("num_datasets", 1)
        max_num_rows = ds_cfg.get("max_num_rows", 3000)
        apply_prob = ds_cfg.get("apply_probability", 1.0)
        fname_label = ds_cfg.get("fname_label", "")
        fraud_overlap_frac = ds_cfg.get("fraud_overlap_frac", -1)

        log.info(
            "  Generating %d dataset(s) for group %d: rules=%s, rows=%d, prob=%.2f",
            num_datasets,
            ds_idx,
            rule_stack,
            max_num_rows,
            apply_prob,
        )

        for i in range(1, num_datasets + 1):
            # Derive a unique seed per dataset for variety
            dataset_seed = seed + ds_idx * 100 + (i - 1)

            log.info(
                "  [%s] Generating dataset %d/%d for group %d, seed=%d",
                site_name,
                i,
                num_datasets,
                ds_idx,
                dataset_seed,
            )

            # Fresh providers per dataset so RNG state is independent
            provider_map = build_providers(dataset_seed)
            graph, providers = build_graph_and_providers(provider_map)

            # Generate normal data
            df = generate(
                graph,
                providers,
                max_num_rows,
                country_static_data=static_data,
                uniform_dist_config_lat=tower_lat_cfg,
                uniform_dist_config_lon=tower_lon_cfg,
                lognormal_personal_amount_config=personal_amount_cfg,
                lognormal_business_amount_config=business_amount_cfg,
            )
            log.info("    Generated %d rows × %d columns", len(df), len(df.columns))

            # Shuffle rows
            df = df.sample(frac=1, replace=False, random_state=dataset_seed).reset_index(drop=True)

            # Add fraud columns and inject anomalies
            add_fraud_columns(df)
            inject_all(
                df,
                anomaly_types=rule_stack,
                configs=anomaly_configs,
                fraud_overlap_frac=fraud_overlap_frac,
                seed=dataset_seed,
            )

            n_fraud = (df["FRAUD_FLAG"] == 1).sum()
            log.info(
                "    Fraud rows: %d / %d (%.2f%%)",
                n_fraud,
                len(df),
                n_fraud / len(df) * 100,
            )

            # Apply fraud probability thinning
            apply_fraud_with_probability(df, prob=apply_prob, random_state=dataset_seed)
            n_fraud_after = (df["FRAUD_FLAG"] == 1).sum()
            log.info(
                "    After probability thinning (prob=%.2f): %d fraud rows",
                apply_prob,
                n_fraud_after,
            )

            # Build filename matching reference notebook convention:
            # {site}_[{fraud_types}]_[app_frac_{prob}]_[{overlap}]_[{label}_{i}].csv
            fname_part_fraud_type = "_".join(rule_stack) if rule_stack else "no_fraud"
            fname_part_label = f"{fname_label}_{i}" if fname_label else str(i)
            fname_part_overlap = (
                f"pct_overlap_{round(fraud_overlap_frac * 100)}" if fraud_overlap_frac > 0 else "no_overlap"
            )
            fname_part_apply_probability = f"app_frac_{apply_prob}"

            csv_path = (
                site_dir
                / f"{site_name}_[{fname_part_fraud_type}]_[{fname_part_apply_probability}]_[{fname_part_overlap}]_[{fname_part_label}].csv"
            )
            df.to_csv(csv_path, index=False)
            log.info("    Written → %s", csv_path)
            generated_files.append(csv_path)

    return generated_files


_CHECKSUM_BLOCK_SIZE: int = 4096


def _calculate_sha256(file_path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with file_path.open("rb") as f:
        for byte_block in iter(lambda: f.read(_CHECKSUM_BLOCK_SIZE), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def write_site_checksums(generated_files: list[Path], site_dir: Path) -> None:
    """Write SHA256 checksums for the given files into the site directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checksum_file = site_dir / f"checksum_{timestamp}.csv"
    log.info("Computing SHA256 checksums for %d file(s)…", len(generated_files))
    checksums = [(f.name, _calculate_sha256(f)) for f in generated_files]
    with checksum_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["File", "SHA256"])
        writer.writerows(checksums)
    log.info("Checksums written → %s", checksum_file)


def combine_scaling_datasets(
    output_dir: Path,
    universal_set_file: Path,
    sample_rows: int,
) -> None:
    """Combine all scaling CSVs across site directories into one dataset."""
    scaling_files = sorted(output_dir.glob("*/*scaling*.csv"))
    scaling_files = [f for f in scaling_files if "archive" not in f.parts]

    if not scaling_files:
        log.warning("No scaling dataset files found under %s/*/", output_dir)
        return

    log.info("Found %d scaling dataset(s) to combine:", len(scaling_files))

    dfs: list[pd.DataFrame] = []
    for f in scaling_files:
        site = f.parent.name
        df = pd.read_csv(f)
        original_len = len(df)
        if sample_rows > 0 and len(df) > sample_rows:
            df = df.sample(n=sample_rows, random_state=42)
        df.insert(0, "SITE", site)
        log.info(
            "  [%s] %s → sampled %d / %d rows",
            site,
            f.name,
            len(df),
            original_len,
        )
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    log.info("Total rows combined: %d", len(combined))

    universal_set_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(universal_set_file, index=False)
    log.info("Universal scaling dataset written → %s", universal_set_file)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    sites = args.sites or discover_sites()
    log.info("Sites: %s", sites)
    log.info("Output dir: %s", args.output_dir)
    log.info("Seed: %d", args.seed)

    for site_name in sites:
        log.info("Processing site: %s", site_name)
        site_config = load_site_config(site_name)
        n_datasets = sum(cfg.get("num_datasets", 1) for cfg in site_config["dataset_generation_config"])
        log.info(
            "  %d dataset config entries → %d total files",
            len(site_config["dataset_generation_config"]),
            n_datasets,
        )
        generated_files = generate_site_datasets(
            site_name,
            site_config,
            args.output_dir,
            args.seed,
        )
        log.info("Site %s complete — %d file(s) produced.", site_name, len(generated_files))

        if args.checksum and generated_files:
            write_site_checksums(generated_files, args.output_dir / site_name)

    if args.generate_universal_set:
        universal_file = args.universal_set_file_path
        if universal_file is None:
            universal_file = args.output_dir / "universal_scaling_datasets_all_banks.csv"
        elif not universal_file.is_absolute():
            universal_file = args.output_dir / universal_file
        log.info("Generating universal scaling dataset → %s", universal_file)
        combine_scaling_datasets(
            args.output_dir,
            universal_file,
            args.universal_set_sample_row_count,
        )

    log.info("All done.")


if __name__ == "__main__":
    main()
