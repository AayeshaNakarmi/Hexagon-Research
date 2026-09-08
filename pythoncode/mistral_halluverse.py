import pandas as pd
import requests
import json
import time
import os
import re
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

API_KEY = ""

API_URL = ""

MODEL = "mistral-small-latest"

# IMPORTANT:
# 250 = 250 VALID PAIRED ROWS
# Not simply CSV row numbers.
MAX_ROWS = 250

# Small pause between API calls
DELAY = 1

# Pause after every 10 API calls
BREAK_AFTER = 10
BREAK_SECONDS = 5

# Number of retries if API fails
MAX_RETRIES = 3


# ============================================================
# BASIC HELPERS
# ============================================================

def has_text(value):

    if pd.isna(value):
        return False

    return str(value).strip() != ""


def clean_text(value):

    if pd.isna(value):
        return ""

    return str(value).strip()


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(file_path):

    print("\n" + "=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    print(f"File: {file_path}")

    df = pd.read_csv(
        file_path,
        encoding="utf-8-sig"
    )

    print(f"Total CSV rows: {len(df)}")

    required_columns = [
        "dialogue",
        "nepali",
        "summary",
        "summary.1",
        "topic"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"\nMissing columns: {missing}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    # Clean important columns

    for column in required_columns:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    # ========================================================
    # IMPORTANT FILTER
    #
    # We only use rows where:
    #
    # English dialogue exists
    # Nepali dialogue exists
    # English summary exists
    # Nepali summary exists
    #
    # This guarantees that English is NOT evaluated
    # when Nepali is unavailable.
    # ========================================================

    valid_mask = (

        df["dialogue"].notna()
        &
        df["nepali"].notna()
        &
        df["summary"].notna()
        &
        df["summary.1"].notna()

        &

        (df["dialogue"] != "")
        &
        (df["nepali"] != "")
        &
        (df["summary"] != "")
        &
        (df["summary.1"] != "")
    )

    valid_df = df[
        valid_mask
    ].copy()

    valid_df.reset_index(
        drop=True,
        inplace=True
    )

    print("\nDATASET CHECK")
    print("-" * 70)

    print(
        f"Total rows: {len(df)}"
    )

    print(
        f"Rows with BOTH English + Nepali data: "
        f"{len(valid_df)}"
    )

    print(
        f"Rows that will be processed: "
        f"{min(MAX_ROWS, len(valid_df))}"
    )

    print(
        "\nRows without Nepali dialogue are NOT evaluated "
        "in English either."
    )

    return valid_df


# ============================================================
# MISTRAL API CALL
# ============================================================

def call_mistral_api(
    prompt,
    model=MODEL,
    retries=MAX_RETRIES
):

    headers = {

        "Authorization":
            f"Bearer {API_KEY}",

        "Content-Type":
            "application/json"
    }

    data = {

        "model": model,

        "messages": [

            {
                "role": "user",
                "content": prompt
            }

        ],

        "temperature": 0.2,

        "max_tokens": 300
    }

    for attempt in range(retries):

        try:

            response = requests.post(

                API_URL,

                headers=headers,

                json=data,

                timeout=90
            )

            response.raise_for_status()

            result = response.json()

            answer = (
                result["choices"][0]
                ["message"]["content"]
            )

            return answer.strip()

        except requests.exceptions.RequestException as e:

            print(
                f"\nAPI error "
                f"(attempt {attempt + 1}/{retries}):"
            )

            print(e)

            if attempt < retries - 1:

                wait_time = 2 ** attempt

                print(
                    f"Retrying in "
                    f"{wait_time} seconds..."
                )

                time.sleep(wait_time)

            else:

                print(
                    "Maximum retries reached."
                )

                return None

        except Exception as e:

            print(
                f"\nUnexpected API error: {e}"
            )

            return None

    return None


# ============================================================
# GENERATION PROMPT — ENGLISH
# ============================================================

def create_english_generation_prompt(dialogue):

    return f"""
You are summarizing a dialogue for a factual evaluation dataset.

Read the dialogue carefully.

Generate:

1. A concise English summary.
2. The main topic of the dialogue.

IMPORTANT:

- The summary must represent the actual meaning of the dialogue.
- Preserve the important facts, events, people, objects, and entities.
- Do not change the main subject.
- Do not replace one entity with another.
- Do not invent information.
- Do not add facts that are not supported by the dialogue.
- Keep the summary concise.
- The topic should describe what the dialogue is mainly about.

Return ONLY valid JSON in exactly this format:

{{
    "summary": "your English summary here",
    "topic": "main topic here"
}}

Dialogue:

{dialogue}
"""


# ============================================================
# GENERATION PROMPT — NEPALI
# ============================================================

def create_nepali_generation_prompt(dialogue):

    return f"""
तपाईं तथ्यमा आधारित संवादको मूल्याङ्कनका लागि सारांश तयार गर्ने सहायक हुनुहुन्छ।

तल दिइएको नेपाली संवादलाई ध्यानपूर्वक पढ्नुहोस्।

तपाईंले दुई कुरा तयार गर्नुपर्छ:

1. संवादको छोटो र स्पष्ट नेपाली सारांश।
2. संवादको मुख्य विषय।

महत्त्वपूर्ण निर्देशनहरू:

- सारांशले संवादको वास्तविक अर्थ सही रूपमा प्रस्तुत गर्नुपर्छ।
- मुख्य तथ्य, घटना, व्यक्ति, वस्तु र महत्वपूर्ण जानकारी सुरक्षित राख्नुहोस्।
- मुख्य विषय परिवर्तन नगर्नुहोस्।
- एउटा व्यक्ति, वस्तु वा संस्थालाई अर्को व्यक्ति, वस्तु वा संस्थासँग नबदल्नुहोस्।
- संवादमा नभएको जानकारी नथप्नुहोस्।
- संवादसँग मेल नखाने कुरा नबनाउनुहोस्।
- सारांश छोटो र स्पष्ट राख्नुहोस्।
- विषयले संवाद मुख्य रूपमा के बारेमा हो भन्ने देखाउनुपर्छ।

ONLY valid JSON मा तलको format प्रयोग गरेर उत्तर दिनुहोस्:

{{
    "summary": "यहाँ नेपाली सारांश लेख्नुहोस्",
    "topic": "यहाँ मुख्य विषय लेख्नुहोस्"
}}

संवाद:

{dialogue}
"""


# ============================================================
# PARSE GENERATION JSON
# ============================================================

def parse_generation_response(response):

    if response is None:

        return "", ""

    text = response.strip()

    # Remove markdown code fences if model accidentally adds them

    text = text.replace(
        "```json",
        ""
    )

    text = text.replace(
        "```",
        ""
    )

    text = text.strip()

    try:

        data = json.loads(text)

        summary = clean_text(
            data.get("summary", "")
        )

        topic = clean_text(
            data.get("topic", "")
        )

        return summary, topic

    except Exception:

        print(
            "\nCould not parse generation JSON."
        )

        print(
            "Raw response:"
        )

        print(response)

        return "", ""


# ============================================================
# SEMANTIC JUDGE PROMPT
# ============================================================

def create_judge_prompt(
    dialogue,
    dataset_summary,
    dataset_topic,
    llm_summary,
    llm_topic,
    language
):

    return f"""
You are a semantic evaluator checking whether an LLM-generated
summary and topic correctly represent a dialogue.

Language:
{language}

The ORIGINAL DIALOGUE is the primary source of truth.

Your job is NOT to compare exact wording.

Different:

- grammar
- sentence structure
- synonyms
- wording
- phrasing

are completely acceptable if the meaning is the same.

For example:

Reference:
"The discussion is about tomato farming."

Generated:
"The conversation focuses on growing tomatoes."

This is SAME MEANING.

However:

Reference:
"The discussion is about tomato farming."

Generated:
"The conversation focuses on growing apples."

This is DIFFERENT MEANING.

You MUST evaluate the meaning and concept.

Check:

1. Does the generated summary represent the same main idea?
2. Does it preserve important facts?
3. Does it preserve important people, objects, places, or entities?
4. Does it describe the same subject?
5. Does the generated topic refer to the same main topic?
6. Did the model accidentally replace one entity with another?
7. Did the model introduce an important fact that is not in the dialogue?
8. Did the model omit something so important that the meaning changes?
9. Does the generated result contradict the dialogue?

DO NOT mark something as correct merely because it shares
some words with the reference.

The generated summary does NOT need to use the same words.

============================================================
ORIGINAL DIALOGUE
============================================================

{dialogue}

============================================================
DATASET REFERENCE SUMMARY
============================================================

{dataset_summary}

============================================================
DATASET REFERENCE TOPIC
============================================================

{dataset_topic}

============================================================
LLM GENERATED SUMMARY
============================================================

{llm_summary}

============================================================
LLM GENERATED TOPIC
============================================================

{llm_topic}

============================================================

Return ONLY valid JSON.

Use exactly this format:

{{
    "summary_result": "Same Meaning",
    "topic_result": "Same Meaning",
    "overall_result": "Same Meaning",
    "score": 0.95,
    "reason": "The generated summary and topic preserve the same meaning as the dialogue and reference."
}}

Rules:

summary_result MUST be either:
"Same Meaning"
or
"Different Meaning"

topic_result MUST be either:
"Same Meaning"
or
"Different Meaning"

overall_result MUST be either:
"Same Meaning"
or
"Different Meaning"

score must be between 0 and 1.

Do not include markdown.
Do not include anything outside the JSON.
"""


# ============================================================
# JUDGE RESPONSE
# ============================================================

def judge_result(
    dialogue,
    dataset_summary,
    dataset_topic,
    llm_summary,
    llm_topic,
    language
):

    if not llm_summary:

        return (
            "Different Meaning",
            "Different Meaning",
            "Different Meaning",
            0.0,
            "No summary generated."
        )

    prompt = create_judge_prompt(

        dialogue=dialogue,

        dataset_summary=dataset_summary,

        dataset_topic=dataset_topic,

        llm_summary=llm_summary,

        llm_topic=llm_topic,

        language=language
    )

    response = call_mistral_api(
        prompt
    )

    if response is None:

        return (
            "Judge Error",
            "Judge Error",
            "Judge Error",
            0.0,
            "Judge API failed."
        )

    text = response.strip()

    text = text.replace(
        "```json",
        ""
    )

    text = text.replace(
        "```",
        ""
    )

    text = text.strip()

    try:

        data = json.loads(text)

        summary_result = data.get(
            "summary_result",
            "Judge Error"
        )

        topic_result = data.get(
            "topic_result",
            "Judge Error"
        )

        overall_result = data.get(
            "overall_result",
            "Judge Error"
        )

        score = float(
            data.get(
                "score",
                0
            )
        )

        reason = clean_text(
            data.get(
                "reason",
                ""
            )
        )

        return (
            summary_result,
            topic_result,
            overall_result,
            score,
            reason
        )

    except Exception as e:

        print(
            "\nCould not parse judge response:"
        )

        print(response)

        return (
            "Judge Error",
            "Judge Error",
            "Judge Error",
            0.0,
            f"Parsing error: {e}"
        )


# ============================================================
# PROCESS ONE ROW
# ============================================================

def process_one_row(
    row,
    row_number,
    total
):

    print("\n" + "=" * 70)

    print(
        f"PROCESSING {row_number}/{total}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # ID
    # --------------------------------------------------------

    if "id" in row.index:

        row_id = row["id"]

    elif "Unnamed: 0" in row.index:

        row_id = row["Unnamed: 0"]

    else:

        row_id = row_number

    # ========================================================
    # ENGLISH
    # ========================================================

    print("\n[ENGLISH]")

    print(
        "Generating English summary + topic..."
    )

    english_prompt = (
        create_english_generation_prompt(
            row["dialogue"]
        )
    )

    english_response = call_mistral_api(
        english_prompt
    )

    english_summary, english_topic = (
        parse_generation_response(
            english_response
        )
    )

    time.sleep(DELAY)

    print(
        "Judging English summary + topic..."
    )

    (
        english_summary_result,
        english_topic_result,
        english_overall_result,
        english_score,
        english_reason

    ) = judge_result(

        dialogue=row["dialogue"],

        dataset_summary=row["summary"],

        dataset_topic=row["topic"],

        llm_summary=english_summary,

        llm_topic=english_topic,

        language="English"
    )

    time.sleep(DELAY)

    # ========================================================
    # NEPALI
    # ========================================================

    print("\n[NEPALI]")

    print(
        "Generating Nepali summary + topic..."
    )

    nepali_prompt = (
        create_nepali_generation_prompt(
            row["nepali"]
        )
    )

    nepali_response = call_mistral_api(
        nepali_prompt
    )

    nepali_summary, nepali_topic = (
        parse_generation_response(
            nepali_response
        )
    )

    time.sleep(DELAY)

    print(
        "Judging Nepali summary + topic..."
    )

    (
        nepali_summary_result,
        nepali_topic_result,
        nepali_overall_result,
        nepali_score,
        nepali_reason

    ) = judge_result(

        dialogue=row["nepali"],

        dataset_summary=row["summary.1"],

        dataset_topic=row["topic"],

        llm_summary=nepali_summary,

        llm_topic=nepali_topic,

        language="Nepali"
    )

    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "row_id": row_id,

        # -------------------------
        # English original
        # -------------------------

        "english_dialogue":
            row["dialogue"],

        "english_dataset_summary":
            row["summary"],

        "english_dataset_topic":
            row["topic"],

        # -------------------------
        # English Mistral output
        # -------------------------

        "english_llm_summary":
            english_summary,

        "english_llm_topic":
            english_topic,

        # -------------------------
        # English evaluation
        # -------------------------

        "english_summary_result":
            english_summary_result,

        "english_topic_result":
            english_topic_result,

        "english_overall_result":
            english_overall_result,

        "english_score":
            english_score,

        "english_judge_reason":
            english_reason,

        # -------------------------
        # Nepali original
        # -------------------------

        "nepali_dialogue":
            row["nepali"],

        "nepali_dataset_summary":
            row["summary.1"],

        "nepali_dataset_topic":
            row["topic"],

        # -------------------------
        # Nepali Mistral output
        # -------------------------

        "nepali_llm_summary":
            nepali_summary,

        "nepali_llm_topic":
            nepali_topic,

        # -------------------------
        # Nepali evaluation
        # -------------------------

        "nepali_summary_result":
            nepali_summary_result,

        "nepali_topic_result":
            nepali_topic_result,

        "nepali_overall_result":
            nepali_overall_result,

        "nepali_score":
            nepali_score,

        "nepali_judge_reason":
            nepali_reason
    }


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(
    df,
    output_file
):

    # --------------------------------------------------------
    # Check for previous results
    # --------------------------------------------------------

    if os.path.exists(output_file):

        print("\nExisting result file found.")

        old_results = pd.read_csv(
            output_file,
            encoding="utf-8-sig"
        )

        # Resume using row_id where possible

        if "row_id" in old_results.columns:

            completed_ids = set(
                old_results[
                    "row_id"
                ].astype(str)
            )

        else:

            completed_ids = set()

    else:

        old_results = pd.DataFrame()

        completed_ids = set()

    # --------------------------------------------------------
    # Limit to first 250 VALID rows
    # --------------------------------------------------------

    df_to_process = df.iloc[
        :MAX_ROWS
    ].copy()

    total = len(
        df_to_process
    )

    # --------------------------------------------------------
    # Remove already completed rows
    # --------------------------------------------------------

    remaining = []

    for _, row in df_to_process.iterrows():

        if str(
            row.get(
                "id",
                row.get(
                    "Unnamed: 0",
                    ""
                )
            )
        ) not in completed_ids:

            remaining.append(row)

    print("\n" + "=" * 70)

    print(
        f"VALID ROWS TO EVALUATE: {total}"
    )

    print(
        f"ALREADY COMPLETED: "
        f"{total - len(remaining)}"
    )

    print(
        f"REMAINING: {len(remaining)}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    new_results = []

    api_call_counter = 0

    for i, row in enumerate(
        remaining,
        start=1
    ):

        result = process_one_row(

            row=row,

            row_number=i,

            total=len(remaining)
        )

        new_results.append(
            result
        )

        api_call_counter += 3

        # ----------------------------------------------------
        # Save after EVERY ROW
        #
        # This is intentional.
        #
        # If your laptop/API stops,
        # you don't lose previous rows.
        # ----------------------------------------------------

        combined = pd.concat(

            [
                old_results,
                pd.DataFrame(
                    new_results
                )
            ],

            ignore_index=True
        )

        combined.to_csv(

            output_file,

            index=False,

            encoding="utf-8-sig"
        )

        print(
            f"\nSaved progress to:"
            f"\n{output_file}"
        )

        # ----------------------------------------------------
        # Pause
        # ----------------------------------------------------

        time.sleep(DELAY)

        if (
            api_call_counter
            >= BREAK_AFTER
        ):

            print(
                f"\nTaking a "
                f"{BREAK_SECONDS}-second break..."
            )

            time.sleep(
                BREAK_SECONDS
            )

            api_call_counter = 0

    # --------------------------------------------------------
    # Final dataframe
    # --------------------------------------------------------

    final_results = pd.concat(

        [
            old_results,
            pd.DataFrame(
                new_results
            )
        ],

        ignore_index=True
    )

    final_results.to_csv(

        output_file,

        index=False,

        encoding="utf-8-sig"
    )

    return final_results


# ============================================================
# CALCULATE ACCURACY
# ============================================================

def calculate_accuracy(
    results,
    language
):

    if language == "English":

        column = (
            "english_overall_result"
        )

    else:

        column = (
            "nepali_overall_result"
        )

    valid = results[
        results[column].isin(
            [
                "Same Meaning",
                "Different Meaning"
            ]
        )
    ]

    total = len(valid)

    same = (
        valid[column]
        ==
        "Same Meaning"
    ).sum()

    different = (
        valid[column]
        ==
        "Different Meaning"
    ).sum()

    accuracy = (

        same / total * 100

        if total > 0

        else 0
    )

    return {
        "language": language,
        "total": total,
        "same_meaning": same,
        "different_meaning": different,
        "accuracy": accuracy
    }


# ============================================================
# SUMMARY/TOPIC ACCURACY SEPARATELY
# ============================================================

def calculate_component_accuracy(
    results,
    language
):

    if language == "English":

        summary_column = (
            "english_summary_result"
        )

        topic_column = (
            "english_topic_result"
        )

    else:

        summary_column = (
            "nepali_summary_result"
        )

        topic_column = (
            "nepali_topic_result"
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary_valid = results[
        results[summary_column].isin(
            [
                "Same Meaning",
                "Different Meaning"
            ]
        )
    ]

    summary_total = len(
        summary_valid
    )

    summary_same = (
        summary_valid[
            summary_column
        ]
        ==
        "Same Meaning"
    ).sum()

    summary_accuracy = (

        summary_same
        /
        summary_total
        *
        100

        if summary_total > 0

        else 0
    )

    # --------------------------------------------------------
    # TOPIC
    # --------------------------------------------------------

    topic_valid = results[
        results[topic_column].isin(
            [
                "Same Meaning",
                "Different Meaning"
            ]
        )
    ]

    topic_total = len(
        topic_valid
    )

    topic_same = (
        topic_valid[
            topic_column
        ]
        ==
        "Same Meaning"
    ).sum()

    topic_accuracy = (

        topic_same
        /
        topic_total
        *
        100

        if topic_total > 0

        else 0
    )

    return {

        "language": language,

        "summary_total":
            summary_total,

        "summary_same":
            summary_same,

        "summary_different":
            summary_total - summary_same,

        "summary_accuracy":
            summary_accuracy,

        "topic_total":
            topic_total,

        "topic_same":
            topic_same,

        "topic_different":
            topic_total - topic_same,

        "topic_accuracy":
            topic_accuracy
    }


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

def print_final_results(
    results
):

    english = calculate_accuracy(
        results,
        "English"
    )

    nepali = calculate_accuracy(
        results,
        "Nepali"
    )

    english_components = (
        calculate_component_accuracy(
            results,
            "English"
        )
    )

    nepali_components = (
        calculate_component_accuracy(
            results,
            "Nepali"
        )
    )

    print("\n\n")
    print("#" * 70)
    print("FINAL RESULTS")
    print("#" * 70)

    # ========================================================
    # OVERALL
    # ========================================================

    print("\nOVERALL SEMANTIC ACCURACY")
    print("-" * 70)

    print(
        f"English:"
    )

    print(
        f"  Total evaluated: "
        f"{english['total']}"
    )

    print(
        f"  Same Meaning: "
        f"{english['same_meaning']}"
    )

    print(
        f"  Different Meaning: "
        f"{english['different_meaning']}"
    )

    print(
        f"  Accuracy: "
        f"{english['accuracy']:.2f}%"
    )

    print()

    print(
        f"Nepali:"
    )

    print(
        f"  Total evaluated: "
        f"{nepali['total']}"
    )

    print(
        f"  Same Meaning: "
        f"{nepali['same_meaning']}"
    )

    print(
        f"  Different Meaning: "
        f"{nepali['different_meaning']}"
    )

    print(
        f"  Accuracy: "
        f"{nepali['accuracy']:.2f}%"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\nSUMMARY ACCURACY")
    print("-" * 70)

    print(
        f"English summary:"
        f" {english_components['summary_accuracy']:.2f}%"
    )

    print(
        f"  Same: "
        f"{english_components['summary_same']}"
    )

    print(
        f"  Different: "
        f"{english_components['summary_different']}"
    )

    print()

    print(
        f"Nepali summary:"
        f" {nepali_components['summary_accuracy']:.2f}%"
    )

    print(
        f"  Same: "
        f"{nepali_components['summary_same']}"
    )

    print(
        f"  Different: "
        f"{nepali_components['summary_different']}"
    )

    # ========================================================
    # TOPIC
    # ========================================================

    print("\nTOPIC ACCURACY")
    print("-" * 70)

    print(
        f"English topic:"
        f" {english_components['topic_accuracy']:.2f}%"
    )

    print(
        f"  Same: "
        f"{english_components['topic_same']}"
    )

    print(
        f"  Different: "
        f"{english_components['topic_different']}"
    )

    print()

    print(
        f"Nepali topic:"
        f" {nepali_components['topic_accuracy']:.2f}%"
    )

    print(
        f"  Same: "
        f"{nepali_components['topic_same']}"
    )

    print(
        f"  Different: "
        f"{nepali_components['topic_different']}"
    )

    # ========================================================
    # LANGUAGE COMPARISON
    # ========================================================

    print("\nLANGUAGE COMPARISON")
    print("-" * 70)

    difference = abs(
        english["accuracy"]
        -
        nepali["accuracy"]
    )

    print(
        f"English accuracy: "
        f"{english['accuracy']:.2f}%"
    )

    print(
        f"Nepali accuracy: "
        f"{nepali['accuracy']:.2f}%"
    )

    print(
        f"Difference: "
        f"{difference:.2f} percentage points"
    )

    if (
        english["accuracy"]
        >
        nepali["accuracy"]
    ):

        print(
            "Better semantic accuracy: English"
        )

    elif (
        nepali["accuracy"]
        >
        english["accuracy"]
    ):

        print(
            "Better semantic accuracy: Nepali"
        )

    else:

        print(
            "Both languages have equal accuracy."
        )

    # ========================================================
    # TABLE
    # ========================================================

    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)

    table = pd.DataFrame([

        {
            "Language":
                "English",

            "Total":
                english["total"],

            "Same Meaning":
                english["same_meaning"],

            "Different Meaning":
                english["different_meaning"],

            "Accuracy %":
                round(
                    english["accuracy"],
                    2
                )
        },

        {
            "Language":
                "Nepali",

            "Total":
                nepali["total"],

            "Same Meaning":
                nepali["same_meaning"],

            "Different Meaning":
                nepali["different_meaning"],

            "Accuracy %":
                round(
                    nepali["accuracy"],
                    2
                )
        }

    ])

    print(
        table.to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # INPUT
    # ========================================================

    file_path = (
        "Hexagon-Dataset - HalluVerseM3.csv"
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    output_file = (
        "HalluVerseM3-Mistral-API-Results.csv"
    )

    # ========================================================
    # API KEY CHECK
    # ========================================================

    if (
        API_KEY
        ==
        "YOUR_MISTRAL_API_KEY"
    ):

        print(
            "\nPlease put your Mistral API key "
            "into API_KEY first."
        )

        return

    # ========================================================
    # LOAD
    # ========================================================

    df = load_dataset(
        file_path
    )

    if len(df) == 0:

        print(
            "\nNo valid paired rows found."
        )

        return

    # ========================================================
    # PROCESS
    # ========================================================

    results = process_dataset(

        df=df,

        output_file=output_file
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print_final_results(
        results
    )

    print("\n" + "#" * 70)

    print(
        "ALL PROCESSING COMPLETE"
    )

    print("#" * 70)

    print(
        f"\nResults file:"
        f"\n{output_file}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print(
        "\nStarting HalluVerseM3 "
        "Mistral API evaluation..."
    )

    print(
        "Started:",
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    main()

    print(
        "\nFinished:",
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )