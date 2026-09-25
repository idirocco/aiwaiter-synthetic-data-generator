import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from config import CATEGORIES
from quality_dimensions import QUALITY_DIMENSIONS
from step1_generation import build_step1_output_path, generator_prompt_slug
from step3_human_labeling import build_step3_output_path
from step4_llm_as_judge import build_step4_output_path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:  # pragma: no cover - plotting is optional in minimal environments
    plt = None
    sns = None

OUTPUT_DIR = Path("output")
VISUALIZATION_DIR = Path("visualizations")
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    return [payload]


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _view_record(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if value is not None:
        return str(value)
    metadata = record.get("metadata") or {}
    return str(metadata.get(key, "unknown"))


def _safe_label_map(record: dict[str, Any], labels_key: str) -> dict[str, int]:
    if not isinstance(record, dict):
        return {dimension: 0 for dimension in QUALITY_DIMENSIONS}

    labels = record.get(labels_key) if labels_key else record
    if not isinstance(labels, dict):
        labels = {}

    normalized: dict[str, int] = {}
    for dimension in QUALITY_DIMENSIONS:
        value = labels.get(dimension)
        if value is None and isinstance(record, dict):
            value = record.get(dimension)
        if isinstance(value, dict):
            value = value.get("pass_", value.get("passed", False))
        normalized[dimension] = int(_normalize_bool(value))
    return normalized


def merge_step_outputs(
    generated_path: str | Path = OUTPUT_DIR / "step1_generated_qa.json",
    human_path: str | Path = OUTPUT_DIR / "step3_human_labels_default.json",
    llm_path: str | Path = OUTPUT_DIR / "step4_llm_judge_labels_default_default.json",
) -> list[dict[str, Any]]:
    generated_records = _read_json(generated_path)
    human_records = {r.get("trace_id"): r for r in _read_json(human_path) if r.get("trace_id")}
    llm_records = {r.get("trace_id"): r for r in _read_json(llm_path) if r.get("trace_id")}

    merged: list[dict[str, Any]] = []
    for index, record in enumerate(generated_records, start=1):
        trace_id = f"qa_{index:03d}"
        metadata = record.get("metadata") or {}
        human = human_records.get(trace_id, {})
        llm = llm_records.get(trace_id, {})

        human_labels = _safe_label_map(human, "human_labels") if "human_labels" in human else _safe_label_map(human, "")
        llm_labels = _safe_label_map(llm, "llm_judge_labels") if "llm_judge_labels" in llm else _safe_label_map(llm, "")

        merged.append(
            {
                "trace_id": trace_id,
                "category_name": metadata.get("category_name") or _view_record(record, "category_name"),
                "prompt_variant": metadata.get("prompt_variant") or "unknown",
                "human_labels": human_labels,
                "llm_judge_labels": llm_labels,
                "overall_pass": _normalize_bool(human.get("overall_pass")) if human else False,
                "llm_overall_pass": _normalize_bool(llm.get("overall_pass")) if llm else False,
            }
        )

    return merged


def compute_overall_metrics(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    record_list = list(records)
    if not record_list:
        return {
            dimension: {"human_pass_rate": 0.0, "llm_pass_rate": 0.0, "agreement_rate": 0.0}
            for dimension in QUALITY_DIMENSIONS + ["overall_pass"]
        }

    overall: dict[str, dict[str, float]] = {}
    for dimension in QUALITY_DIMENSIONS + ["overall_pass"]:
        human_total = 0.0
        llm_total = 0.0
        agreement_total = 0.0

        for record in record_list:
            if dimension == "overall_pass":
                human_value = int(_normalize_bool(record.get("overall_pass")))
                llm_value = int(_normalize_bool(record.get("llm_overall_pass")))
            else:
                human_labels = _safe_label_map(record, "human_labels")
                llm_labels = _safe_label_map(record, "llm_judge_labels")
                human_value = human_labels.get(dimension, 0)
                llm_value = llm_labels.get(dimension, 0)

            human_total += float(human_value)
            llm_total += float(llm_value)
            agreement_total += 1.0 if human_value == llm_value else 0.0

        total_records = float(len(record_list))
        overall[dimension] = {
            "human_pass_rate": round(human_total / total_records if total_records else 0.0, 4),
            "llm_pass_rate": round(llm_total / total_records if total_records else 0.0, 4),
            "agreement_rate": round(agreement_total / total_records if total_records else 0.0, 4),
        }

    return overall


def aggregate_segment_metrics(records: Iterable[dict[str, Any]], group_key: str = "category_name") -> dict[str, dict[str, dict[str, float]]]:
    group_stats: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: {
            dimension: {"human_pass_total": 0.0, "llm_pass_total": 0.0, "agreement_total": 0.0}
            for dimension in QUALITY_DIMENSIONS
        }
    )
    group_counts: dict[str, int] = defaultdict(int)

    for record in records:
        group_name = str(record.get(group_key) or record.get("metadata", {}).get(group_key) or "unknown")
        group_counts[group_name] += 1

        human_labels = _safe_label_map(record, "human_labels")
        llm_labels = _safe_label_map(record, "llm_judge_labels")

        for dimension in QUALITY_DIMENSIONS:
            human_value = human_labels.get(dimension, 0)
            llm_value = llm_labels.get(dimension, 0)
            group_stats[group_name][dimension]["human_pass_total"] += float(human_value)
            group_stats[group_name][dimension]["llm_pass_total"] += float(llm_value)
            group_stats[group_name][dimension]["agreement_total"] += 1.0 if human_value == llm_value else 0.0

        overall_human = int(_normalize_bool(record.get("overall_pass")))
        overall_llm = int(_normalize_bool(record.get("llm_overall_pass")))
        group_stats[group_name]["overall_pass"] = group_stats[group_name].get("overall_pass", {"human_pass_total": 0.0, "llm_pass_total": 0.0, "agreement_total": 0.0})
        group_stats[group_name]["overall_pass"]["human_pass_total"] += float(overall_human)
        group_stats[group_name]["overall_pass"]["llm_pass_total"] += float(overall_llm)
        group_stats[group_name]["overall_pass"]["agreement_total"] += 1.0 if overall_human == overall_llm else 0.0

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for group_name, stats in group_stats.items():
        group_total = group_counts.get(group_name, 0)
        dimension_summary: dict[str, dict[str, float]] = {}

        for dimension in QUALITY_DIMENSIONS:
            metric = stats[dimension]
            human_rate = metric["human_pass_total"] / group_total if group_total else 0.0
            llm_rate = metric["llm_pass_total"] / group_total if group_total else 0.0
            agreement_rate = metric["agreement_total"] / group_total if group_total else 0.0
            dimension_summary[dimension] = {
                "human_pass_rate": round(float(human_rate), 4),
                "llm_pass_rate": round(float(llm_rate), 4),
                "agreement_rate": round(float(agreement_rate), 4),
            }

        overall_metric = stats.get("overall_pass", {"human_pass_total": 0.0, "llm_pass_total": 0.0, "agreement_total": 0.0})
        dimension_summary["overall_pass"] = {
            "human_pass_rate": round(float(overall_metric["human_pass_total"] / group_total) if group_total else 0.0, 4),
            "llm_pass_rate": round(float(overall_metric["llm_pass_total"] / group_total) if group_total else 0.0, 4),
            "agreement_rate": round(float(overall_metric["agreement_total"] / group_total) if group_total else 0.0, 4),
        }
        summary[group_name] = dimension_summary

    return summary


def format_dimension_label(name: str) -> str:
    return name.replace("_", " ").title()


def build_segment_heatmap(records: Iterable[dict[str, Any]], group_key: str = "category_name") -> tuple[list[str], list[str], list[list[float]]]:
    summary = aggregate_segment_metrics(records, group_key=group_key)
    segments = sorted(summary)
    matrix: list[list[float]] = []
    for segment in segments:
        row: list[float] = []
        for dimension in QUALITY_DIMENSIONS:
            row.append(summary[segment][dimension]["llm_pass_rate"])
        matrix.append(row)
    return segments, QUALITY_DIMENSIONS, matrix


def save_segment_metrics_to_json(summary: dict[str, Any], output_path: str | Path = OUTPUT_DIR / "step5_segment_metrics.json") -> Path:
    file_path = Path(output_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved segment-level metrics to: {file_path.resolve()}")
    return file_path


def build_agreement_metrics(overall_metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        dimension: {
            "agreement_rate": overall_metrics[dimension]["agreement_rate"],
            "human_pass_rate": overall_metrics[dimension]["human_pass_rate"],
            "llm_pass_rate": overall_metrics[dimension]["llm_pass_rate"],
        }
        for dimension in QUALITY_DIMENSIONS
        if dimension in overall_metrics
    }


def save_agreement_metrics_to_json(
    overall_metrics: dict[str, dict[str, float]],
    output_path: str | Path = VISUALIZATION_DIR / "step5_human_llm_agreement.json",
) -> Path:
    file_path = Path(output_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_agreement_metrics(overall_metrics)
    file_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved human-LLM agreement metrics to: {file_path.resolve()}")
    return file_path


def plot_dimension_bar_chart(overall_metrics: dict[str, dict[str, float]], title: str, output_path: Path) -> None:
    if plt is None:
        return

    dimension_names = [name for name in QUALITY_DIMENSIONS if name in overall_metrics]
    display_names = [format_dimension_label(name) for name in dimension_names]
    human_rates = [overall_metrics[dimension]["human_pass_rate"] for dimension in dimension_names]
    llm_rates = [overall_metrics[dimension]["llm_pass_rate"] for dimension in dimension_names]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(dimension_names))
    ax.bar([i - 0.18 for i in x], human_rates, width=0.35, color="#9ecae1", label="Human")
    ax.bar([i + 0.18 for i in x], llm_rates, width=0.35, color="#2171b5", label="LLM judge")
    ax.set_title(title)
    ax.set_xlabel("Quality dimension")
    ax.set_ylabel("Pass rate")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(list(x))
    ax.set_xticklabels(display_names, rotation=25, ha="right")
    ax.legend(frameon=False)
    fig.subplots_adjust(bottom=0.26, top=0.9, left=0.08, right=0.98)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_agreement_bar_chart(overall_metrics: dict[str, dict[str, float]], output_path: Path) -> None:
    if plt is None:
        return

    dimension_names = [name for name in QUALITY_DIMENSIONS if name in overall_metrics]
    display_names = [format_dimension_label(name) for name in dimension_names]
    agreement_rates = [overall_metrics[dimension]["agreement_rate"] for dimension in dimension_names]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(display_names, agreement_rates, color="#f4a582")
    ax.set_title("Human vs. LLM agreement by dimension")
    ax.set_xlabel("Quality dimension")
    ax.set_ylabel("Agreement rate")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    for tick in ax.get_xticklabels():
        tick.set_rotation(25)
        tick.set_ha("right")
    fig.subplots_adjust(bottom=0.24, top=0.9, left=0.08, right=0.98)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_segment_heatmap(records: Iterable[dict[str, Any]], output_path: Path, group_key: str = "category_name") -> None:
    if plt is None:
        return

    segments, dimensions, matrix = build_segment_heatmap(records, group_key=group_key)
    if not segments:
        return

    fig, ax = plt.subplots(figsize=(max(12, len(dimensions) * 1.7), max(5, len(segments) * 0.9)))
    sns.heatmap(
        matrix,
        xticklabels=[format_dimension_label(d) for d in dimensions],
        yticklabels=segments,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Pass rate"},
        ax=ax,
    )
    ax.set_title(f"Segment pass rate heatmap by {group_key.replace('_', ' ')}")
    ax.set_xlabel("Quality dimension")
    ax.set_ylabel(group_key.replace("_", " ").title())
    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.2, right=0.98)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_category_distribution(records: Iterable[dict[str, Any]], output_path: Path) -> None:
    if plt is None:
        return

    counts: dict[str, int] = defaultdict(int)
    for record in records:
        category_name = record.get("category_name") or "unknown"
        counts[category_name] += 1

    categories = sorted(counts)
    generated_counts = [counts[category] for category in categories]
    total_records = sum(generated_counts)
    benchmark_counts = [max(1, round((total_records * 0.2))) for _ in categories]

    if len(CATEGORIES) == len(categories):
        benchmark_counts = [max(1, round(total_records / len(CATEGORIES))) for _ in categories]

    x = range(len(categories))
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.35
    ax.bar([i - width / 2 for i in x], generated_counts, width=width, label="Generated count")
    ax.bar([i + width / 2 for i in x], benchmark_counts, width=width, label="Benchmark count")
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_title("Generated category distribution vs benchmark")
    ax.set_xlabel("Category")
    ax.set_ylabel("Counts")
    ax.legend(frameon=False)
    fig.subplots_adjust(bottom=0.28, top=0.9, left=0.08, right=0.98)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_before_after_per_dimension(records: Iterable[dict[str, Any]], output_path: Path) -> None:
    if plt is None:
        return

    record_list = list(records)
    variant_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: {dimension: [] for dimension in QUALITY_DIMENSIONS})
    for record in record_list:
        prompt_variant = str(record.get("prompt_variant") or "unknown")
        human_labels = _safe_label_map(record, "human_labels")
        llm_labels = _safe_label_map(record, "llm_judge_labels")
        for dimension in QUALITY_DIMENSIONS:
            variant_values[prompt_variant][dimension].append(human_labels.get(dimension, 0))
            variant_values[prompt_variant][dimension].append(llm_labels.get(dimension, 0))

    if not variant_values:
        return

    variants = sorted(variant_values)
    dimension_names = QUALITY_DIMENSIONS
    means: list[list[float]] = []
    for variant in variants:
        row = []
        for dimension in dimension_names:
            values = variant_values[variant][dimension]
            row.append(sum(values) / len(values) if values else 0.0)
        means.append(row)

    x = range(len(dimension_names))
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.75 / max(1, len(variants))
    for index, variant in enumerate(variants):
        offset = (index - (len(variants) - 1) / 2) * width
        values = [means[index][position] for position in range(len(dimension_names))]
        ax.bar([i + offset for i in x], values, width=width, label=str(variant))

    ax.set_xticks(list(x))
    ax.set_xticklabels([format_dimension_label(d) for d in dimension_names], rotation=25, ha="right")
    ax.set_title("Per-dimension comparison by prompt variant")
    ax.set_xlabel("Quality dimension")
    ax.set_ylabel("Mean pass rate")
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(0, 1.05)
    fig.subplots_adjust(bottom=0.26, top=0.9, left=0.08, right=0.98)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_step5_output_path(
    stem: str,
    extension: str,
    generator_prompt: str | None = None,
    judge_prompt: str | None = None,
    output_dir: str | Path = VISUALIZATION_DIR,
) -> Path:
    prompt_slug = f"{generator_prompt_slug(generator_prompt)}_{generator_prompt_slug(judge_prompt)}"
    return Path(output_dir) / f"{stem}_{prompt_slug}.{extension}"


def generate_step5_visualizations(
    records: Iterable[dict[str, Any]],
    output_dir: str | Path = VISUALIZATION_DIR,
    generator_prompt: str | None = None,
    judge_prompt: str | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    record_list = list(records)
    overall_metrics = compute_overall_metrics(record_list)
    summary = aggregate_segment_metrics(record_list, group_key="category_name")
    metrics_path = save_segment_metrics_to_json(
        summary,
        build_step5_output_path(
            "step5_segment_metrics",
            "json",
            generator_prompt=generator_prompt,
            judge_prompt=judge_prompt,
            output_dir=output_dir,
        ),
    )

    agreement_path = save_agreement_metrics_to_json(
        overall_metrics,
        build_step5_output_path(
            "step5_human_llm_agreement",
            "json",
            generator_prompt=generator_prompt,
            judge_prompt=judge_prompt,
            output_dir=output_dir,
        ),
    )

    saved_paths = {
        "segment_metrics": metrics_path,
        "agreement_by_dimension_json": agreement_path,
        # "segment_heatmap": build_step5_output_path(
        #     "step5_segment_heatmap", "png", generator_prompt, judge_prompt, output_dir
        # ),
        # "dimension_pass_rates": build_step5_output_path(
        #     "step5_dimension_pass_rates", "png", generator_prompt, judge_prompt, output_dir
        # ),
        "agreement_by_dimension": build_step5_output_path(
            "step5_human_llm_agreement", "png", generator_prompt, judge_prompt, output_dir
        ),
        # "category_distribution": build_step5_output_path(
        #     "step5_category_distribution", "png", generator_prompt, judge_prompt, output_dir
        # ),
        # "before_after": build_step5_output_path(
        #     "step5_before_after_per_dimension", "png", generator_prompt, judge_prompt, output_dir
        # ),
    }

    # plot_segment_heatmap(record_list, saved_paths["segment_heatmap"], group_key="category_name")
    # plot_dimension_bar_chart(overall_metrics, "Per-dimension pass rate (LLM judge)", saved_paths["dimension_pass_rates"])
    plot_agreement_bar_chart(overall_metrics, saved_paths["agreement_by_dimension"])
    # plot_category_distribution(record_list, saved_paths["category_distribution"])
    # plot_before_after_per_dimension(record_list, saved_paths["before_after"])

    print(f"Saved step 5 visualizations to: {output_dir.resolve()}")
    return saved_paths


def run_step5_analysis(
    generated_path: str | Path | None = None,
    human_path: str | Path | None = None,
    llm_path: str | Path | None = None,
    output_dir: str | Path = VISUALIZATION_DIR,
    generator_prompt: str | None = None,
    judge_prompt: str | None = None,
) -> dict[str, Any]:
    if generated_path is None:
        generated_path = build_step1_output_path(generator_prompt)
    if human_path is None:
        human_path = build_step3_output_path(generator_prompt)
    if llm_path is None:
        llm_path = build_step4_output_path(generator_prompt, judge_prompt)

    merged_records = merge_step_outputs(generated_path=generated_path, human_path=human_path, llm_path=llm_path)
    summary = aggregate_segment_metrics(merged_records, group_key="category_name")
    plots = generate_step5_visualizations(
        merged_records,
        output_dir=output_dir,
        generator_prompt=generator_prompt,
        judge_prompt=judge_prompt,
    )
    return {
        "records": merged_records,
        "summary": summary,
        "plots": plots,
    }


if __name__ == "__main__":
    run_step5_analysis()

