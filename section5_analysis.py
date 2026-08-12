

from pathlib import Path
import sys
import warnings
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


DEFAULT_INPUT = "Dyslexia Study Responses (11).xlsx"
INPUT_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_INPUT)
OUTPUT_DIR = Path("section5_outputs_complete")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_PARTICIPANTS = 20
EXPECTED_ITEMS = 83

CATEGORY_ORDER = [
    "Analytical Thinking & Problem Solving",
    "Attention Difficulties",
    "Communication & Expression",
    "Planning & Organization / Time Management",
    "Reading & Decoding Difficulties",
    "Working Memory Limitations",
    "Writing & Spelling Difficulties",
]

CATEGORY_COLORS = {
    "Analytical Thinking & Problem Solving": "#A9A9A9",
    "Attention Difficulties": "#DAA520",
    "Communication & Expression": "#DB7093",
    "Planning & Organization / Time Management": "#87CEEB",
    "Reading & Decoding Difficulties": "#1E90FF",
    "Working Memory Limitations": "#20B2AA",
    "Writing & Spelling Difficulties": "#D2691E",
}

EXPECTED_CATEGORY_ITEMS = {
    "Writing & Spelling Difficulties": 12,
    "Reading & Decoding Difficulties": 24,
    "Working Memory Limitations": 17,
    "Analytical Thinking & Problem Solving": 9,
    "Communication & Expression": 9,
    "Attention Difficulties": 7,
    "Planning & Organization / Time Management": 5,
}


def first_non_null(series):
    """Return the first non-missing value in a participant-level field."""
    s = series.dropna()
    return s.iloc[0] if not s.empty else np.nan


def save_df(df, filename, index=False):
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=index)
    return path


def section(title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def r2(value):
    """Publication rounding to two decimals using conventional half-up rounding."""
    return Decimal(str(float(value))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# 1. LOAD RAW DATA

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE.resolve()}\n"
        f"Place the Excel file beside this script or pass its path as an argument."
    )

raw = pd.read_excel(INPUT_FILE)

required_columns = {
    "timestamp_iso", "participant_id", "category_id", "category_label",
    "barrier_index", "barrier_text", "frequency_code", "impact_score",
    "years_experience", "age_range", "gender", "dyslexia_diagnosis",
    "identified_conditions",
}
missing_columns = required_columns.difference(raw.columns)
if missing_columns:
    raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

raw["timestamp_dt"] = pd.to_datetime(raw["timestamp_iso"], utc=True, errors="coerce")
raw["frequency_code"] = pd.to_numeric(raw["frequency_code"], errors="coerce")
raw["impact_score"] = pd.to_numeric(raw["impact_score"], errors="coerce")

section("RAW DATA")
print(f"Input file: {INPUT_FILE.resolve()}")
print(f"Original rows: {len(raw)}")
print(f"Unique participant IDs: {raw['participant_id'].nunique(dropna=True)}")


# 2.DEMOGRAPHIC SUMMARY

demographic_fields = [
    "years_experience", "age_range", "gender", "dyslexia_diagnosis",
    "identified_conditions", "other_conditions_affecting_reading",
    "other_conditions_details",
]

participants = (
    raw.groupby("participant_id", as_index=False)
       .agg({field: first_non_null for field in demographic_fields})
)

n_participants = len(participants)
if n_participants != EXPECTED_PARTICIPANTS:
    warnings.warn(
        f"Expected {EXPECTED_PARTICIPANTS} participants, found {n_participants}."
    )

# Standard demographic counts.
experience_counts = participants["years_experience"].value_counts(dropna=False)
age_counts = participants["age_range"].value_counts(dropna=False)
gender_counts = participants["gender"].value_counts(dropna=False)
diagnosis_counts = participants["dyslexia_diagnosis"].value_counts(dropna=False)

# Neurodivergent / related conditions can contain comma-separated labels.
condition_counter = {}
for value in participants["identified_conditions"].dropna():
    for condition in [x.strip() for x in str(value).split(",") if x.strip()]:
        condition_counter[condition] = condition_counter.get(condition, 0) + 1
condition_counts = pd.Series(condition_counter).sort_values(ascending=False)

section("PARTICIPANT SUMMARY")
print(f"Participants: n = {n_participants}")
print("\nYears of experience:")
print(experience_counts.to_string())
print("\nAge:")
print(age_counts.to_string())
print("\nGender:")
print(gender_counts.to_string())
print("\nDyslexia diagnosis status:")
print(diagnosis_counts.to_string())
print("\nIdentified conditions:")
print(condition_counts.to_string())

save_df(participants, "section5_participant_level_demographics.csv")
condition_counts.rename("count").to_csv(
    OUTPUT_DIR / "section5_condition_counts.csv",
    header=True,
)


# 3. CLEAN DATA
section("DATA CLEANING AND SCORING")


follow_up_mask = (
    raw["category_id"].eq("follow_up")
    | raw["category_label"].eq("Follow-up Contact")
)
follow_up_rows = int(follow_up_mask.sum())
barriers = raw.loc[~follow_up_mask].copy()

print(f"Follow-up/contact rows removed: {follow_up_rows}")
print(f"Barrier-response rows before de-duplication: {len(barriers)}")

duplicate_key = ["participant_id", "category_id", "barrier_index"]

repeat_sizes = (
    barriers.groupby(duplicate_key, dropna=False)
            .size()
            .reset_index(name="stored_record_count")
)
repeat_keys = repeat_sizes.loc[repeat_sizes["stored_record_count"] > 1, duplicate_key]

if not repeat_keys.empty:
    repeated_records = (
        barriers.merge(repeat_keys.assign(_repeat=True), on=duplicate_key, how="inner")
                .sort_values(duplicate_key + ["timestamp_dt"], na_position="first")
    )
    save_df(
        repeated_records[
            [
                "participant_id", "category_id", "category_label", "barrier_index",
                "barrier_text", "timestamp_iso", "frequency_code", "impact_score",
            ]
        ],
        "section5_repeated_stored_records_audit.csv",
    )
else:
    repeated_records = pd.DataFrame()

n_repeat_combinations = len(repeat_keys)
n_extra_records = int(
    repeat_sizes.loc[repeat_sizes["stored_record_count"] > 1, "stored_record_count"]
                .sub(1)
                .sum()
)

print(f"Repeated participant-item combinations: {n_repeat_combinations}")
print(f"Extra stored records resolved: {n_extra_records}")

barriers = (
    barriers.sort_values("timestamp_dt", na_position="first")
            .drop_duplicates(subset=duplicate_key, keep="last")
            .copy()
)

print(f"Unique participant-item records after de-duplication: {len(barriers)}")

barriers["frequency_clean"] = barriers["frequency_code"]
barriers["impact_clean"] = barriers["impact_score"]

never_nonzero = barriers.loc[
    barriers["frequency_clean"].eq(0)
    & barriers["impact_clean"].notna()
    & barriers["impact_clean"].ne(0),
    [
        "participant_id", "category_id", "category_label", "barrier_index",
        "barrier_text", "frequency_clean", "impact_clean", "timestamp_iso",
    ],
].copy()

if not never_nonzero.empty:
    save_df(
        never_nonzero,
        "section5_frequency_zero_nonzero_impact_audit.csv",
    )


barriers.loc[barriers["frequency_clean"].eq(0), "impact_clean"] = 0

valid_frequency = int(barriers["frequency_clean"].notna().sum())
missing_frequency = int(barriers["frequency_clean"].isna().sum())
never_frequency = int(barriers["frequency_clean"].eq(0).sum())
valid_impact = int(barriers["impact_clean"].notna().sum())
missing_impact = int(barriers["impact_clean"].isna().sum())

print(f"Valid Frequency responses: {valid_frequency}")
print(f"Missing Frequency responses: {missing_frequency}")
print(f"Frequency = 0 (Never): {never_frequency}")
print(f"Valid Impact responses after Never rule: {valid_impact}")
print(f"Missing Impact responses after Never rule: {missing_impact}")


item_key = ["category_id", "category_label", "barrier_index", "barrier_text"]

item_stats = (
    barriers.groupby(item_key, dropna=False)
            .agg(
                n_frequency=("frequency_clean", "count"),
                mean_frequency=("frequency_clean", "mean"),
                sd_frequency=("frequency_clean", "std"),
                n_impact=("impact_clean", "count"),
                mean_impact=("impact_clean", "mean"),
                sd_impact=("impact_clean", "std"),
            )
            .reset_index()
)

if len(item_stats) != EXPECTED_ITEMS:
    raise AssertionError(
        f"Expected {EXPECTED_ITEMS} barrier items, found {len(item_stats)}."
    )

section("ITEM-LEVEL STATISTICS")
print(f"Barrier items: {len(item_stats)}")
print(
    f"Frequency valid n per item: {item_stats['n_frequency'].min()}–"
    f"{item_stats['n_frequency'].max()}"
)
print(
    f"Impact valid n per item: {item_stats['n_impact'].min()}–"
    f"{item_stats['n_impact'].max()}"
)

grand_mean_frequency = float(item_stats["mean_frequency"].mean())
grand_mean_impact = float(item_stats["mean_impact"].mean())

sd_item_frequency = float(item_stats["mean_frequency"].std(ddof=1))
sd_item_impact = float(item_stats["mean_impact"].std(ddof=1))

frequency_min = float(item_stats["mean_frequency"].min())
frequency_max = float(item_stats["mean_frequency"].max())
impact_min = float(item_stats["mean_impact"].min())
impact_max = float(item_stats["mean_impact"].max())

section("5.1.1 OVERALL DISTRIBUTION")
print(f"Mean Frequency across 83 item means: {grand_mean_frequency:.12f} -> {r2(grand_mean_frequency)}")
print(f"Frequency SD across item means:      {sd_item_frequency:.12f} -> {r2(sd_item_frequency)}")
print(f"Frequency item-mean range:           {frequency_min:.12f} to {frequency_max:.12f}")
print(f"Mean Impact across 83 item means:    {grand_mean_impact:.12f} -> {r2(grand_mean_impact)}")
print(f"Impact SD across item means:         {sd_item_impact:.12f} -> {r2(sd_item_impact)}")
print(f"Impact item-mean range:              {impact_min:.12f} to {impact_max:.12f}")


category_stats = (
    item_stats.groupby("category_label", dropna=False)
              .agg(
                  n_items=("barrier_text", "size"),
                  M_frequency=("mean_frequency", "mean"),
                  SD_frequency=("mean_frequency", "std"),
                  M_impact=("mean_impact", "mean"),
                  SD_impact=("mean_impact", "std"),
              )
              .reset_index()
)

category_stats["frequency_rank"] = (
    category_stats["M_frequency"].rank(method="min", ascending=False).astype(int)
)
category_stats["impact_rank"] = (
    category_stats["M_impact"].rank(method="min", ascending=False).astype(int)
)
category_stats = category_stats.sort_values("frequency_rank").reset_index(drop=True)

actual_category_items = dict(
    zip(category_stats["category_label"], category_stats["n_items"])
)
if actual_category_items != EXPECTED_CATEGORY_ITEMS:
    raise AssertionError(
        "Category item counts do not match the instrument.\n"
        f"Expected: {EXPECTED_CATEGORY_ITEMS}\n"
        f"Actual:   {actual_category_items}"
    )

section("5.1.2 CATEGORY-LEVEL RANKINGS — TABLE 11")
print(
    category_stats[
        [
            "category_label", "n_items", "frequency_rank", "M_frequency",
            "SD_frequency", "impact_rank", "M_impact", "SD_impact",
        ]
    ].to_string(index=False)
)



item_stats["frequency_rank"] = (
    item_stats["mean_frequency"].rank(method="min", ascending=False).astype(int)
)
item_stats["impact_rank"] = (
    item_stats["mean_impact"].rank(method="min", ascending=False).astype(int)
)

top_frequency = (
    item_stats.loc[item_stats["frequency_rank"] <= 5]
              .sort_values(
                  ["frequency_rank", "mean_frequency", "mean_impact", "barrier_text"],
                  ascending=[True, False, False, True],
              )
              .copy()
)

top_impact = (
    item_stats.loc[item_stats["impact_rank"] <= 5]
              .sort_values(
                  ["impact_rank", "mean_impact", "mean_frequency", "barrier_text"],
                  ascending=[True, False, False, True],
              )
              .copy()
)

section("5.1.3 ITEM-LEVEL ANALYSIS — TABLE 12")
print("TOP FREQUENCY ITEMS (ties retained):")
print(
    top_frequency[
        [
            "frequency_rank", "barrier_text", "category_label",
            "n_frequency", "mean_frequency", "n_impact", "mean_impact",
        ]
    ].to_string(index=False)
)
print("\nTOP IMPACT ITEMS:")
print(
    top_impact[
        [
            "impact_rank", "barrier_text", "category_label",
            "n_frequency", "mean_frequency", "n_impact", "mean_impact",
        ]
    ].to_string(index=False)
)


pearson_r, pearson_p = pearsonr(
    item_stats["mean_frequency"].to_numpy(),
    item_stats["mean_impact"].to_numpy(),
)
pearson_r = float(pearson_r)
pearson_p = float(pearson_p)

section("5.1.4 FREQUENCY–IMPACT RELATIONSHIP")
print(f"Pearson r = {pearson_r:.12f} -> {pearson_r:.3f}")
print(f"Exact p    = {pearson_p:.12e}")
print("Publication form: p < .001" if pearson_p < 0.001 else f"Publication form: p = {pearson_p:.3f}")


high_frequency = item_stats["mean_frequency"] >= grand_mean_frequency
high_impact = item_stats["mean_impact"] >= grand_mean_impact

item_stats["quadrant"] = np.select(
    [
        high_frequency & high_impact,
        (~high_frequency) & (~high_impact),
        (~high_frequency) & high_impact,
        high_frequency & (~high_impact),
    ],
    [
        "High Frequency / High Impact",
        "Low Frequency / Low Impact",
        "Low Frequency / High Impact",
        "High Frequency / Low Impact",
    ],
    default="Unclassified",
)

quadrant_order = [
    "High Frequency / High Impact",
    "Low Frequency / Low Impact",
    "Low Frequency / High Impact",
    "High Frequency / Low Impact",
]

quadrant_counts = (
    item_stats["quadrant"]
    .value_counts()
    .reindex(quadrant_order, fill_value=0)
)
quadrant_summary = quadrant_counts.rename("n_items").to_frame()
quadrant_summary["percentage"] = quadrant_summary["n_items"] / EXPECTED_ITEMS * 100
quadrant_summary.index.name = "quadrant"

quadrant_category = (
    item_stats.groupby(["quadrant", "category_label"])
              .size()
              .rename("n_items")
              .reset_index()
)

section("QUADRANT ANALYSIS")
print(f"Frequency reference line = {grand_mean_frequency:.12f} -> {r2(grand_mean_frequency)}")
print(f"Impact reference line    = {grand_mean_impact:.12f} -> {r2(grand_mean_impact)}")
print("\nQuadrant totals:")
print(quadrant_summary.to_string())
print("\nCategory composition by quadrant:")
for quadrant in quadrant_order:
    print(f"\n{quadrant}")
    q = (
        quadrant_category.loc[quadrant_category["quadrant"].eq(quadrant)]
                         .sort_values(["n_items", "category_label"], ascending=[False, True])
    )
    print(q[["category_label", "n_items"]].to_string(index=False))

if int(quadrant_summary["n_items"].sum()) != EXPECTED_ITEMS:
    raise AssertionError("Quadrant counts do not sum to 83 items.")


save_df(item_stats, "section5_all_83_item_statistics.csv")
save_df(category_stats, "section5_table11_category_rankings.csv")
save_df(top_frequency, "section5_table12_top_frequency.csv")
save_df(top_impact, "section5_table12_top_impact.csv")
quadrant_summary.to_csv(OUTPUT_DIR / "section5_quadrant_summary.csv")
save_df(quadrant_category, "section5_quadrant_category_breakdown.csv")

scatter_coordinates = item_stats[
    [
        "category_label", "barrier_index", "barrier_text",
        "n_frequency", "mean_frequency", "n_impact", "mean_impact",
        "frequency_rank", "impact_rank", "quadrant",
    ]
].copy()
save_df(scatter_coordinates, "section5_scatter_coordinates.csv")


fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for category in CATEGORY_ORDER:
    subset = item_stats.loc[item_stats["category_label"].eq(category)]
    if subset.empty:
        continue
    ax.scatter(
        subset["mean_frequency"],
        subset["mean_impact"],
        s=180,
        c=CATEGORY_COLORS[category],
        alpha=0.85,
        edgecolors="black",
        linewidths=0.8,
        label=category,
        zorder=3,
    )

ax.axvline(
    grand_mean_frequency,
    linestyle="--",
    linewidth=1.5,
    color="gray",
    alpha=0.75,
    zorder=1,
)
ax.axhline(
    grand_mean_impact,
    linestyle="--",
    linewidth=1.5,
    color="gray",
    alpha=0.75,
    zorder=1,
)


def quadrant_box(edge_color):
    return dict(
        boxstyle="round,pad=0.3",
        facecolor="white",
        edgecolor=edge_color,
        linewidth=1.5,
    )


# Descriptive quadrant labels.
ax.text(
    0.94, 0.93, "High Frequency\nHigh Impact",
    transform=ax.transAxes, ha="right", va="top",
    fontsize=12, fontweight="bold", color="#D2691E",
    bbox=quadrant_box("#D2691E"), zorder=5,
)
ax.text(
    0.10, 0.16, "Low Frequency\nLow Impact",
    transform=ax.transAxes, ha="left", va="bottom",
    fontsize=12, fontweight="bold", color="#20B2AA",
    bbox=quadrant_box("#20B2AA"), zorder=5,
)
ax.text(
    0.94, 0.16, "High Frequency\nLow Impact",
    transform=ax.transAxes, ha="right", va="bottom",
    fontsize=12, fontweight="bold", color="#1E90FF",
    bbox=quadrant_box("#1E90FF"), zorder=5,
)
ax.text(
    0.10, 0.64, "Low Frequency\nHigh Impact",
    transform=ax.transAxes, ha="left", va="top",
    fontsize=12, fontweight="bold", color="#DB7093",
    bbox=quadrant_box("#DB7093"), zorder=5,
)

ax.set_xlabel("Mean Frequency Score", fontsize=14, fontweight="bold")
ax.set_ylabel("Mean Impact Score", fontsize=14, fontweight="bold")
ax.set_title(
    "Barrier Frequency vs. Impact by Category\n"
    f"(83 barrier items; total participant sample n = {n_participants})",
    fontsize=16,
    fontweight="bold",
    pad=18,
)

ax.legend(
    title="Barrier Category",
    title_fontsize=12,
    fontsize=10,
    loc="upper left",
    frameon=True,
    edgecolor="gray",
    facecolor="white",
)

ax.set_xlim(0.5, 3.5)
ax.set_ylim(0.5, 3.2)
ax.set_xticks(np.arange(1.0, 3.5, 0.5))
ax.set_yticks(np.arange(0.5, 3.5, 0.5))
ax.grid(True, alpha=0.16)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=11)

plt.tight_layout()

scatter_png = OUTPUT_DIR / "figure4_frequency_impact_scatter_400dpi.png"
scatter_pdf = OUTPUT_DIR / "figure4_frequency_impact_scatter.pdf"
scatter_svg = OUTPUT_DIR / "figure4_frequency_impact_scatter.svg"

fig.savefig(scatter_png, dpi=400, bbox_inches="tight")
fig.savefig(scatter_pdf, bbox_inches="tight")
fig.savefig(scatter_svg, bbox_inches="tight")
plt.close(fig)



verified_values = {
    "grand_mean_frequency": (grand_mean_frequency, 2.243754037127531, 1e-12),
    "grand_mean_impact": (grand_mean_impact, 1.677927694795165, 1e-12),
    "sd_item_frequency": (sd_item_frequency, 0.44692751908026346, 1e-12),
    "sd_item_impact": (sd_item_impact, 0.46222094487626886, 1e-12),
    "pearson_r": (pearson_r, 0.8987543397705828, 1e-12),
}

for name, (actual, expected, atol) in verified_values.items():
    if not np.isclose(actual, expected, atol=atol, rtol=0):
        warnings.warn(
            f"Verification mismatch for {name}: actual={actual!r}, expected={expected!r}"
        )

expected_quadrants = {
    "High Frequency / High Impact": 36,
    "Low Frequency / Low Impact": 35,
    "Low Frequency / High Impact": 7,
    "High Frequency / Low Impact": 5,
}
for q, expected in expected_quadrants.items():
    actual = int(quadrant_counts.loc[q])
    if actual != expected:
        warnings.warn(f"Quadrant mismatch for {q}: actual={actual}, expected={expected}")


publication_report = OUTPUT_DIR / "section5_publication_values.txt"
with publication_report.open("w", encoding="utf-8") as f:
    f.write("SECTION 5 VERIFIED PUBLICATION VALUES\n")
    f.write("=" * 50 + "\n")
    f.write(f"Participants: n = {n_participants}\n")
    f.write(f"Barrier items: {len(item_stats)}\n\n")

    f.write("Overall distribution (83 item means)\n")
    f.write(f"Frequency: M = {r2(grand_mean_frequency)}, SD = {r2(sd_item_frequency)}, ")
    f.write(f"range = {r2(frequency_min)}–{r2(frequency_max)}\n")
    f.write(f"Impact: M = {r2(grand_mean_impact)}, SD = {r2(sd_item_impact)}, ")
    f.write(f"range = {r2(impact_min)}–{r2(impact_max)}\n\n")

    f.write("Pearson correlation\n")
    f.write(f"r = {pearson_r:.3f}, exact p = {pearson_p:.12e}\n\n")

    f.write("Quadrants\n")
    for q in quadrant_order:
        n = int(quadrant_counts.loc[q])
        pct = n / EXPECTED_ITEMS * 100
        f.write(f"{q}: {n} ({pct:.1f}%)\n")

    f.write("\nCategory rankings\n")
    for _, row in category_stats.iterrows():
        f.write(
            f"{row['category_label']}: "
            f"Freq rank {int(row['frequency_rank'])}, M={r2(row['M_frequency'])}, "
            f"SD={r2(row['SD_frequency'])}; "
            f"Impact rank {int(row['impact_rank'])}, M={r2(row['M_impact'])}, "
            f"SD={r2(row['SD_impact'])}\n"
        )


section("FINAL VERIFIED SECTION 5 VALUES")
print(f"Participants: n = {n_participants}")
print(f"Barrier items: {len(item_stats)}")
print(
    f"Frequency: M = {r2(grand_mean_frequency)}, SD = {r2(sd_item_frequency)}, "
    f"range = {r2(frequency_min)}–{r2(frequency_max)}"
)
print(
    f"Impact:    M = {r2(grand_mean_impact)}, SD = {r2(sd_item_impact)}, "
    f"range = {r2(impact_min)}–{r2(impact_max)}"
)
print(f"Pearson:   r = {pearson_r:.3f}, p < .001")
print("Quadrants:")
for q in quadrant_order:
    n = int(quadrant_counts.loc[q])
    print(f"  {q}: {n} ({n / EXPECTED_ITEMS * 100:.1f}%)")

print("\nFiles created:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {path}")

print("\nAnalysis completed successfully.")
