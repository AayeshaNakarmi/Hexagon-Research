import ollama
import pandas as pd
import time
import re
import os
from tqdm import tqdm


# ============================================================
# SETTINGS
# ============================================================

MODEL = "mistral"

# IMPORTANT:
# 250 = 250 ACTUAL STATEMENTS, NOT 250 CSV ROWS
BATCH_SIZE = 250

# Small delay between statements
DELAY = 0.5

# Rest after every 250 REAL statements
BATCH_BREAK = 15


# ============================================================
# VERIFIER
# ============================================================

class StatementVerifier:

    def __init__(
        self,
        model=MODEL,
        delay=DELAY,
        batch_size=BATCH_SIZE
    ):
        self.model = model
        self.delay = delay
        self.batch_size = batch_size

    # --------------------------------------------------------
    # CHECK ONE STATEMENT
    # --------------------------------------------------------

    def classify_statement(self, statement, language):

        if language == "Nepali":

            language_instruction = """
The statement is written in Nepali.
Understand the Nepali statement accurately before deciding.
"""

        else:

            language_instruction = """
The statement is written in English.
"""

        prompt = f"""
You are a fact-checking assistant.

{language_instruction}

Determine whether the following statement is factually TRUE or FALSE.

Statement:
{statement}

Instructions:

- Decide based on factual accuracy.
- Respond with ONLY one word:
  True
  or
  False

Do not explain.
Do not provide any other text.

Answer:
"""

        try:

            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            answer = response["message"]["content"].strip()

            match = re.search(
                r"\b(true|false)\b",
                answer,
                re.IGNORECASE
            )

            if match:
                return match.group(1).capitalize()

            return "Unknown"

        except Exception as e:

            print(f"\nMistral error: {e}")

            return "Error"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def has_statement(value):

    if pd.isna(value):
        return False

    return str(value).strip() != ""


def is_finished(value):

    return str(value).strip() in [
        "True",
        "False",
        "Unknown",
        "Error"
    ]


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(file_path):

    print(f"\nLoading: {file_path}")

    df = pd.read_csv(
        file_path,
        encoding="utf-8-sig"
    )

    print(f"Total CSV rows: {len(df)}")

    required = [
        "id",
        "statement"
    ]

    for column in required:

        if column not in df.columns:

            raise ValueError(
                f"Column '{column}' was not found "
                f"in {file_path}\n"
                f"Available columns: {list(df.columns)}"
            )

    return df


# ============================================================
# PREPARE / RESUME EXISTING RESULTS
# ============================================================

def prepare_output(
    original_df,
    output_file
):

    # --------------------------------------------------------
    # If result file already exists, load it
    # --------------------------------------------------------

    if os.path.exists(output_file):

        print("\nExisting result file found:")
        print(output_file)

        existing = pd.read_csv(
            output_file,
            encoding="utf-8-sig"
        )

        # Same number of rows = probably same dataset
        if (
            len(existing) == len(original_df)
            and "id" in existing.columns
        ):

            print(
                "Using existing results "
                "to continue from where you stopped."
            )

            df = existing.copy()

            # Make sure original columns still exist
            for column in original_df.columns:

                if column not in df.columns:

                    df[column] = original_df[column].values

            return df

        print(
            "Existing result file does not match "
            "the current dataset."
        )

    # --------------------------------------------------------
    # No previous result
    # --------------------------------------------------------

    df = original_df.copy()

    df["mistral_result"] = ""

    return df


# ============================================================
# MARK EMPTY STATEMENTS
# ============================================================

def mark_empty_statements(df):

    if "mistral_result" not in df.columns:

        df["mistral_result"] = ""

    for index in df.index:

        statement = df.at[
            index,
            "statement"
        ]

        if not has_statement(statement):

            df.at[
                index,
                "mistral_result"
            ] = "No Statement"

    return df


# ============================================================
# PROCESS DATASET
#
# IMPORTANT:
# BATCHES ARE BASED ON REAL STATEMENTS.
# EMPTY ROWS DO NOT COUNT.
# ============================================================

def process_dataset(
    verifier,
    df,
    output_file,
    language
):

    print("\n")
    print("#" * 70)
    print(f"{language.upper()} PROCESSING")
    print("#" * 70)

    # --------------------------------------------------------
    # Make sure result column exists
    # --------------------------------------------------------

    if "mistral_result" not in df.columns:

        df["mistral_result"] = ""

    # --------------------------------------------------------
    # Mark empty statements
    # --------------------------------------------------------

    df = mark_empty_statements(df)

    # Save immediately
    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Count actual statements
    # --------------------------------------------------------

    statement_mask = df["statement"].apply(
        has_statement
    )

    total_real_statements = statement_mask.sum()

    print(
        f"\nTotal CSV rows: "
        f"{len(df)}"
    )

    print(
        f"Actual {language} statements: "
        f"{total_real_statements}"
    )

    print(
        f"Empty {language} rows: "
        f"{len(df) - total_real_statements}"
    )

    # --------------------------------------------------------
    # Find unfinished REAL statements
    # --------------------------------------------------------

    unfinished_mask = (
        statement_mask
        &
        ~df["mistral_result"].apply(
            is_finished
        )
    )

    unfinished_indices = list(
        df.index[unfinished_mask]
    )

    completed_count = (
        total_real_statements
        -
        len(unfinished_indices)
    )

    print(
        f"\nAlready completed: "
        f"{completed_count}"
    )

    print(
        f"Still remaining: "
        f"{len(unfinished_indices)}"
    )

    # --------------------------------------------------------
    # PROCESS 250 REAL STATEMENTS AT A TIME
    # --------------------------------------------------------

    batch_number = 0

    for start in range(
        0,
        len(unfinished_indices),
        verifier.batch_size
    ):

        batch_number += 1

        batch_indices = unfinished_indices[
            start:
            start + verifier.batch_size
        ]

        print("\n" + "=" * 70)

        print(
            f"{language} BATCH {batch_number}"
        )

        print(
            f"REAL STATEMENTS IN THIS BATCH: "
            f"{len(batch_indices)}"
        )

        print(
            f"Overall completed BEFORE this batch: "
            f"{completed_count}"
        )

        print("=" * 70)

        # ----------------------------------------------------
        # CHECK EACH REAL STATEMENT
        # ----------------------------------------------------

        for index in tqdm(
            batch_indices,
            desc=f"{language} Batch {batch_number}"
        ):

            statement = df.at[
                index,
                "statement"
            ]

            if not has_statement(statement):

                df.at[
                    index,
                    "mistral_result"
                ] = "No Statement"

                continue

            result = verifier.classify_statement(
                str(statement),
                language
            )

            df.at[
                index,
                "mistral_result"
            ] = result

            time.sleep(
                verifier.delay
            )

        # ----------------------------------------------------
        # SAVE AFTER BATCH
        # ----------------------------------------------------

        df.to_csv(
            output_file,
            index=False,
            encoding="utf-8-sig"
        )

        completed_count += len(batch_indices)

        print("\n" + "-" * 70)

        print(
            f"Saved after {completed_count} "
            f"/ {total_real_statements} "
            f"real statements."
        )

        print(
            f"File: {output_file}"
        )

        # ----------------------------------------------------
        # REST AFTER BATCH
        # ----------------------------------------------------

        if completed_count < total_real_statements:

            print(
                f"\nTaking a {BATCH_BREAK}-second break "
                f"for the laptop..."
            )

            time.sleep(
                BATCH_BREAK
            )

    print("\n" + "#" * 70)

    print(
        f"{language.upper()} PROCESSING COMPLETE"
    )

    print("#" * 70)

    return df


# ============================================================
# GET IDS THAT HAVE REAL NEPALI STATEMENTS
# ============================================================

def get_nepali_statement_ids(nepali_df):

    mask = nepali_df["statement"].apply(
        has_statement
    )

    ids = set(
        nepali_df.loc[
            mask,
            "id"
        ]
    )

    print("\n" + "=" * 70)
    print("NEPALI IDs FOR ENGLISH FILTERING")
    print("=" * 70)

    print(
        f"IDs with actual Nepali statements: "
        f"{len(ids)}"
    )

    return ids


# ============================================================
# FILTER ENGLISH
#
# ONLY IDS THAT HAVE A NEPALI STATEMENT
# ============================================================

def filter_english(
    english_df,
    valid_nepali_ids
):

    filtered = english_df[
        english_df["id"].isin(
            valid_nepali_ids
        )
    ].copy()

    filtered.reset_index(
        drop=True,
        inplace=True
    )

    print("\n" + "=" * 70)
    print("ENGLISH FILTER")
    print("=" * 70)

    print(
        f"Original English rows: "
        f"{len(english_df)}"
    )

    print(
        f"English rows that WILL be checked: "
        f"{len(filtered)}"
    )

    print(
        f"English rows NOT checked: "
        f"{len(english_df) - len(filtered)}"
    )

    return filtered


# ============================================================
# COMPARE RESULTS
#
# THIS IS THE NEW / UPDATED PART
# ============================================================

def compare_results(
    nepali_df,
    english_df,
    output_file
):

    print("\n")
    print("#" * 80)
    print("FINAL ENGLISH vs NEPALI COMPARISON")
    print("#" * 80)

    # ========================================================
    # PREPARE DATA
    # ========================================================

    nepali = nepali_df.rename(
        columns={
            "label": "nepali_label",
            "statement": "nepali_statement",
            "mistral_result": "nepali_result"
        }
    )

    english = english_df.rename(
        columns={
            "label": "english_label",
            "statement": "english_statement",
            "mistral_result": "english_result"
        }
    )

    # ========================================================
    # ONLY COMPARE IDs THAT EXIST IN BOTH
    # ========================================================

    comparison = pd.merge(
        nepali[
            [
                "id",
                "nepali_label",
                "nepali_statement",
                "nepali_result"
            ]
        ],

        english[
            [
                "id",
                "english_label",
                "english_statement",
                "english_result"
            ]
        ],

        on="id",
        how="inner"
    )

    # ========================================================
    # CLEAN RESULTS
    # ========================================================

    comparison["nepali_result"] = (
        comparison["nepali_result"]
        .astype(str)
        .str.strip()
        .str.capitalize()
    )

    comparison["english_result"] = (
        comparison["english_result"]
        .astype(str)
        .str.strip()
        .str.capitalize()
    )

    comparison["nepali_label"] = (
        comparison["nepali_label"]
        .astype(str)
        .str.strip()
        .str.capitalize()
    )

    comparison["english_label"] = (
        comparison["english_label"]
        .astype(str)
        .str.strip()
        .str.capitalize()
    )

    # ========================================================
    # SAME / DIFFERENT FOR EACH LANGUAGE
    # ========================================================

    comparison["nepali_same_different"] = (
        comparison["nepali_result"]
        ==
        comparison["nepali_label"]
    ).map({
        True: "Same",
        False: "Different"
    })

    comparison["english_same_different"] = (
        comparison["english_result"]
        ==
        comparison["english_label"]
    ).map({
        True: "Same",
        False: "Different"
    })

    # ========================================================
    # ONLY VALID TRUE/FALSE RESULTS
    # ========================================================

    valid_nepali = comparison[
        comparison["nepali_result"].isin(
            ["True", "False"]
        )
        &
        comparison["nepali_label"].isin(
            ["True", "False"]
        )
    ].copy()

    valid_english = comparison[
        comparison["english_result"].isin(
            ["True", "False"]
        )
        &
        comparison["english_label"].isin(
            ["True", "False"]
        )
    ].copy()

    # ========================================================
    # ENGLISH DETAILED BREAKDOWN
    # ========================================================

    english_breakdown = (
        valid_english
        .groupby(
            [
                "english_result",
                "english_label",
                "english_same_different"
            ]
        )
        .size()
        .reset_index(name="number")
    )

    english_breakdown = english_breakdown.rename(
        columns={
            "english_result": "mistral_result",
            "english_label": "dataset_label",
            "english_same_different": "same_or_different"
        }
    )

    english_breakdown.insert(
        0,
        "language",
        "English"
    )

    # ========================================================
    # NEPALI DETAILED BREAKDOWN
    # ========================================================

    nepali_breakdown = (
        valid_nepali
        .groupby(
            [
                "nepali_result",
                "nepali_label",
                "nepali_same_different"
            ]
        )
        .size()
        .reset_index(name="number")
    )

    nepali_breakdown = nepali_breakdown.rename(
        columns={
            "nepali_result": "mistral_result",
            "nepali_label": "dataset_label",
            "nepali_same_different": "same_or_different"
        }
    )

    nepali_breakdown.insert(
        0,
        "language",
        "Nepali"
    )

    # ========================================================
    # COMBINE ENGLISH + NEPALI
    # ========================================================

    detailed_breakdown = pd.concat(
        [
            english_breakdown,
            nepali_breakdown
        ],
        ignore_index=True
    )

    # Make sure all four possible combinations appear,
    # even if their count is ZERO.

    expected_rows = []

    for language in ["English", "Nepali"]:

        for result in ["True", "False"]:

            for label in ["True", "False"]:

                if result == label:
                    same_diff = "Same"
                else:
                    same_diff = "Different"

                expected_rows.append({
                    "language": language,
                    "mistral_result": result,
                    "dataset_label": label,
                    "same_or_different": same_diff
                })

    expected_df = pd.DataFrame(
        expected_rows
    )

    detailed_breakdown = pd.merge(
        expected_df,
        detailed_breakdown,
        on=[
            "language",
            "mistral_result",
            "dataset_label",
            "same_or_different"
        ],
        how="left"
    )

    detailed_breakdown["number"] = (
        detailed_breakdown["number"]
        .fillna(0)
        .astype(int)
    )

    # ========================================================
    # PRINT DETAILED BREAKDOWN
    # ========================================================

    print("\n")
    print("=" * 80)
    print("DETAILED TRUE/FALSE + SAME/DIFFERENT")
    print("=" * 80)

    print(
        detailed_breakdown.to_string(
            index=False
        )
    )

    # ========================================================
    # ENGLISH SUMMARY
    # ========================================================

    english_total = len(valid_english)

    english_true = (
        valid_english["english_result"]
        == "True"
    ).sum()

    english_false = (
        valid_english["english_result"]
        == "False"
    ).sum()

    english_same = (
        valid_english["english_same_different"]
        == "Same"
    ).sum()

    english_different = (
        valid_english["english_same_different"]
        == "Different"
    ).sum()

    english_true_same = (
        (valid_english["english_result"] == "True")
        &
        (valid_english["english_label"] == "True")
    ).sum()

    english_true_different = (
        (valid_english["english_result"] == "True")
        &
        (valid_english["english_label"] == "False")
    ).sum()

    english_false_same = (
        (valid_english["english_result"] == "False")
        &
        (valid_english["english_label"] == "False")
    ).sum()

    english_false_different = (
        (valid_english["english_result"] == "False")
        &
        (valid_english["english_label"] == "True")
    ).sum()

    english_accuracy = (
        english_same
        /
        english_total
        *
        100
        if english_total > 0
        else 0
    )

    # ========================================================
    # NEPALI SUMMARY
    # ========================================================

    nepali_total = len(valid_nepali)

    nepali_true = (
        valid_nepali["nepali_result"]
        == "True"
    ).sum()

    nepali_false = (
        valid_nepali["nepali_result"]
        == "False"
    ).sum()

    nepali_same = (
        valid_nepali["nepali_same_different"]
        == "Same"
    ).sum()

    nepali_different = (
        valid_nepali["nepali_same_different"]
        == "Different"
    ).sum()

    nepali_true_same = (
        (valid_nepali["nepali_result"] == "True")
        &
        (valid_nepali["nepali_label"] == "True")
    ).sum()

    nepali_true_different = (
        (valid_nepali["nepali_result"] == "True")
        &
        (valid_nepali["nepali_label"] == "False")
    ).sum()

    nepali_false_same = (
        (valid_nepali["nepali_result"] == "False")
        &
        (valid_nepali["nepali_label"] == "False")
    ).sum()

    nepali_false_different = (
        (valid_nepali["nepali_result"] == "False")
        &
        (valid_nepali["nepali_label"] == "True")
    ).sum()

    nepali_accuracy = (
        nepali_same
        /
        nepali_total
        *
        100
        if nepali_total > 0
        else 0
    )

    # ========================================================
    # PRINT ENGLISH
    # ========================================================

    print("\n")
    print("=" * 80)
    print("🇬🇧 ENGLISH")
    print("=" * 80)

    print(
        f"Total valid statements: {english_total}"
    )

    print(
        f"\nMistral TRUE: {english_true}"
    )

    print(
        f"  TRUE + Dataset TRUE  = Same: "
        f"{english_true_same}"
    )

    print(
        f"  TRUE + Dataset FALSE = Different: "
        f"{english_true_different}"
    )

    print(
        f"\nMistral FALSE: {english_false}"
    )

    print(
        f"  FALSE + Dataset FALSE = Same: "
        f"{english_false_same}"
    )

    print(
        f"  FALSE + Dataset TRUE  = Different: "
        f"{english_false_different}"
    )

    print(
        f"\nTotal Same: {english_same}"
    )

    print(
        f"Total Different: {english_different}"
    )

    print(
        f"Accuracy: {english_accuracy:.2f}%"
    )

    # ========================================================
    # PRINT NEPALI
    # ========================================================

    print("\n")
    print("=" * 80)
    print("🇳🇵 NEPALI")
    print("=" * 80)

    print(
        f"Total valid statements: {nepali_total}"
    )

    print(
        f"\nMistral TRUE: {nepali_true}"
    )

    print(
        f"  TRUE + Dataset TRUE  = Same: "
        f"{nepali_true_same}"
    )

    print(
        f"  TRUE + Dataset FALSE = Different: "
        f"{nepali_true_different}"
    )

    print(
        f"\nMistral FALSE: {nepali_false}"
    )

    print(
        f"  FALSE + Dataset FALSE = Same: "
        f"{nepali_false_same}"
    )

    print(
        f"  FALSE + Dataset TRUE  = Different: "
        f"{nepali_false_different}"
    )

    print(
        f"\nTotal Same: {nepali_same}"
    )

    print(
        f"Total Different: {nepali_different}"
    )

    print(
        f"Accuracy: {nepali_accuracy:.2f}%"
    )

    # ========================================================
    # ENGLISH vs NEPALI — SAME IDS
    # ========================================================

    both_valid = comparison[
        comparison["english_result"].isin(
            ["True", "False"]
        )
        &
        comparison["nepali_result"].isin(
            ["True", "False"]
        )
    ].copy()

    # --------------------------------------------------------
    # Four possible prediction combinations
    # --------------------------------------------------------

    both_true = (
        (both_valid["english_result"] == "True")
        &
        (both_valid["nepali_result"] == "True")
    ).sum()

    both_false = (
        (both_valid["english_result"] == "False")
        &
        (both_valid["nepali_result"] == "False")
    ).sum()

    english_true_nepali_false = (
        (both_valid["english_result"] == "True")
        &
        (both_valid["nepali_result"] == "False")
    ).sum()

    english_false_nepali_true = (
        (both_valid["english_result"] == "False")
        &
        (both_valid["nepali_result"] == "True")
    ).sum()

    same_prediction = (
        both_valid["english_result"]
        ==
        both_valid["nepali_result"]
    ).sum()

    different_prediction = (
        len(both_valid)
        -
        same_prediction
    )

    # ========================================================
    # PRINT CROSS-LANGUAGE COMPARISON
    # ========================================================

    print("\n")
    print("=" * 80)
    print("🔄 ENGLISH vs NEPALI — SAME IDs")
    print("=" * 80)

    print(
        f"Both languages valid: "
        f"{len(both_valid)}"
    )

    print(
        f"\nBoth predicted TRUE: "
        f"{both_true}"
    )

    print(
        f"Both predicted FALSE: "
        f"{both_false}"
    )

    print(
        f"English TRUE / Nepali FALSE: "
        f"{english_true_nepali_false}"
    )

    print(
        f"English FALSE / Nepali TRUE: "
        f"{english_false_nepali_true}"
    )

    print(
        f"\nSame prediction: "
        f"{same_prediction}"
    )

    print(
        f"Different prediction: "
        f"{different_prediction}"
    )

    # ========================================================
    # ACCURACY DIFFERENCE
    # ========================================================

    accuracy_difference = abs(
        english_accuracy
        -
        nepali_accuracy
    )

    print("\n")
    print("=" * 80)
    print("📊 ACCURACY COMPARISON")
    print("=" * 80)

    print(
        f"English accuracy: "
        f"{english_accuracy:.2f}%"
    )

    print(
        f"Nepali accuracy: "
        f"{nepali_accuracy:.2f}%"
    )

    print(
        f"Difference: "
        f"{accuracy_difference:.2f} percentage points"
    )

    if english_accuracy > nepali_accuracy:

        print(
            "Better accuracy: ENGLISH"
        )

    elif nepali_accuracy > english_accuracy:

        print(
            "Better accuracy: NEPALI"
        )

    else:

        print(
            "Better accuracy: EQUAL"
        )

    # ========================================================
    # ADD CROSS-LANGUAGE COMPARISON TO DATAFRAME
    # ========================================================

    comparison[
        "language_prediction_comparison"
    ] = "Unknown"

    valid_both_mask = (
        comparison["english_result"].isin(
            ["True", "False"]
        )
        &
        comparison["nepali_result"].isin(
            ["True", "False"]
        )
    )

    comparison.loc[
        valid_both_mask
        &
        (
            comparison["english_result"]
            ==
            comparison["nepali_result"]
        ),
        "language_prediction_comparison"
    ] = "Same"

    comparison.loc[
        valid_both_mask
        &
        (
            comparison["english_result"]
            !=
            comparison["nepali_result"]
        ),
        "language_prediction_comparison"
    ] = "Different"

    # ========================================================
    # SAVE FULL COMPARISON
    # ========================================================

    comparison.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # SAVE DETAILED BREAKDOWN
    # ========================================================

    breakdown_file = (
        "Hexagon-English-Nepali-Detailed-Breakdown.csv"
    )

    detailed_breakdown.to_csv(
        breakdown_file,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # SAVE SIMPLE SUMMARY
    # ========================================================

    summary_data = pd.DataFrame({

        "Language": [
            "English",
            "Nepali"
        ],

        "Total_Valid_Statements": [
            english_total,
            nepali_total
        ],

        "Mistral_True": [
            english_true,
            nepali_true
        ],

        "True_Dataset_True_Same": [
            english_true_same,
            nepali_true_same
        ],

        "True_Dataset_False_Different": [
            english_true_different,
            nepali_true_different
        ],

        "Mistral_False": [
            english_false,
            nepali_false
        ],

        "False_Dataset_False_Same": [
            english_false_same,
            nepali_false_same
        ],

        "False_Dataset_True_Different": [
            english_false_different,
            nepali_false_different
        ],

        "Total_Same": [
            english_same,
            nepali_same
        ],

        "Total_Different": [
            english_different,
            nepali_different
        ],

        "Accuracy_Percent": [
            english_accuracy,
            nepali_accuracy
        ]
    })

    summary_file = (
        "Hexagon-English-Nepali-Summary.csv"
    )

    summary_data.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # SAVE CROSS-LANGUAGE SUMMARY
    # ========================================================

    cross_language_summary = pd.DataFrame({

        "Metric": [
            "Both predicted TRUE",
            "Both predicted FALSE",
            "English TRUE / Nepali FALSE",
            "English FALSE / Nepali TRUE",
            "Same prediction",
            "Different prediction",
            "English accuracy",
            "Nepali accuracy",
            "Accuracy difference"
        ],

        "Number": [
            both_true,
            both_false,
            english_true_nepali_false,
            english_false_nepali_true,
            same_prediction,
            different_prediction,
            english_accuracy,
            nepali_accuracy,
            accuracy_difference
        ]
    })

    cross_file = (
        "Hexagon-English-Nepali-Cross-Language-Summary.csv"
    )

    cross_language_summary.to_csv(
        cross_file,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # PRINT FILES
    # ========================================================

    print("\n")
    print("=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print(
        f"\n1. Full row-by-row comparison:"
        f"\n   {output_file}"
    )

    print(
        f"\n2. Detailed True/False breakdown:"
        f"\n   {breakdown_file}"
    )

    print(
        f"\n3. English/Nepali summary:"
        f"\n   {summary_file}"
    )

    print(
        f"\n4. Cross-language comparison:"
        f"\n   {cross_file}"
    )

    return comparison


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # INPUT FILES
    # --------------------------------------------------------

    nepali_file = (
        "Hexagon-Dataset - Nepali-PolyFEVER.csv"
    )

    english_file = (
        "Hexagon-Dataset - English-PolyFEVER.csv"
    )

    # --------------------------------------------------------
    # OUTPUT FILES
    # --------------------------------------------------------

    nepali_output = (
        "Hexagon-Nepali-Mistral-Results.csv"
    )

    english_output = (
        "Hexagon-English-Filtered-Mistral-Results.csv"
    )

    comparison_output = (
        "Hexagon-Nepali-English-Comparison.csv"
    )

    # --------------------------------------------------------
    # START VERIFIER
    # --------------------------------------------------------

    verifier = StatementVerifier(
        model=MODEL,
        delay=DELAY,
        batch_size=BATCH_SIZE
    )

    # ========================================================
    # STEP 1 — LOAD NEPALI
    # ========================================================

    nepali_original = load_dataset(
        nepali_file
    )

    # ========================================================
    # STEP 2 — PREPARE / RESUME NEPALI
    # ========================================================

    nepali_df = prepare_output(
        nepali_original,
        nepali_output
    )

    # ========================================================
    # STEP 3 — PROCESS NEPALI
    #
    # 250 ACTUAL STATEMENTS AT A TIME
    # ========================================================

    nepali_df = process_dataset(
        verifier,
        nepali_df,
        nepali_output,
        "Nepali"
    )

    # ========================================================
    # STEP 4 — GET IDS WITH REAL NEPALI STATEMENTS
    # ========================================================

    valid_nepali_ids = get_nepali_statement_ids(
        nepali_df
    )

    # ========================================================
    # STEP 5 — LOAD ENGLISH
    # ========================================================

    english_original = load_dataset(
        english_file
    )

    # ========================================================
    # STEP 6 — FILTER ENGLISH
    #
    # ONLY IDs THAT HAVE NEPALI STATEMENTS
    # ========================================================

    english_filtered = filter_english(
        english_original,
        valid_nepali_ids
    )

    # ========================================================
    # STEP 7 — PREPARE / RESUME ENGLISH
    # ========================================================

    english_df = prepare_output(
        english_filtered,
        english_output
    )

    # ========================================================
    # STEP 8 — PROCESS ENGLISH
    #
    # 250 ACTUAL STATEMENTS AT A TIME
    # ========================================================

    english_df = process_dataset(
        verifier,
        english_df,
        english_output,
        "English"
    )

    # ========================================================
    # STEP 9 — COMPARE ENGLISH + NEPALI
    # ========================================================

    compare_results(
        nepali_df,
        english_df,
        comparison_output
    )

    # ========================================================
    # FINISHED
    # ========================================================

    print("\n")
    print("#" * 70)
    print("🎉 EVERYTHING IS COMPLETE")
    print("#" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()