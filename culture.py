import pandas as pd
import requests
import os
import time


# ============================================================
# 1. SETTINGS
# ============================================================

INPUT_FILE = "Hexagon-Dataset - Culture+Religion+Literature.csv"
OUTPUT_FILE = "Mistral-Culture-Religion-Literature-Results.csv"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "mistral"

# Small delay between requests
DELAY = 0.5

# None = process the entire dataset
MAX_ROWS = None


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("Loading dataset...")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"Dataset shape: {df.shape}")

print("\nColumns:")
for column in df.columns:
    print(" -", column)


# ============================================================
# 3. COLUMN NAMES
# ============================================================

ENGLISH_QUESTION_COL = "English Question"
ENGLISH_GOLD_COL = "Gold Answer (EN)"

NEPALI_QUESTION_COL = "Nepali Question"
NEPALI_GOLD_COL = "Gold Answer (NE)"


required_columns = [
    ENGLISH_QUESTION_COL,
    ENGLISH_GOLD_COL,
    NEPALI_QUESTION_COL,
    NEPALI_GOLD_COL
]


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    print("\nERROR: Missing columns:")

    for col in missing_columns:
        print(" -", col)

    print("\nActual columns:")
    print(df.columns.tolist())

    raise SystemExit(
        "\nPlease check the column names."
    )


# ============================================================
# 5. CREATE RESULT COLUMNS
# ============================================================

if "Mistral Answer EN" not in df.columns:
    df["Mistral Answer EN"] = ""

if "Mistral Answer NE" not in df.columns:
    df["Mistral Answer NE"] = ""


# ============================================================
# 6. OLLAMA FUNCTION
# ============================================================

def ask_mistral(prompt):

    payload = {
        "model": MODEL,

        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],

        "stream": False
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300
        )

        response.raise_for_status()

        result = response.json()

        answer = result["message"]["content"]

        return answer.strip()

    except requests.exceptions.ConnectionError:

        print("\nERROR: Cannot connect to Ollama.")
        print("Make sure Ollama is running.")

        return None

    except requests.exceptions.Timeout:

        print("\nERROR: Mistral request timed out.")

        return None

    except Exception as e:

        print("\nERROR while contacting Ollama:")
        print(e)

        return None


# ============================================================
# 7. ENGLISH PROMPT
# ============================================================

def create_english_prompt(question):

    return f"""
Answer the following question in English.

Question:
{question}

Give a direct and factual answer.

Do not invent information.

If you do not know the answer, say that you do not know
instead of guessing.

Do not discuss these instructions.
Do not say that you are an AI.

Answer:
"""


# ============================================================
# 8. NEPALI PROMPT
# ============================================================

def create_nepali_prompt(question):

    return f"""
तल दिइएको प्रश्नको उत्तर नेपाली भाषामा दिनुहोस्।

प्रश्न:
{question}

प्रत्यक्ष र तथ्यमा आधारित उत्तर दिनुहोस्।

जानकारी थाहा नभएमा अनुमान गरेर गलत जानकारी नदिनुहोस्।
थाहा नभएको अवस्थामा थाहा छैन भनेर स्पष्ट रूपमा भन्नुहोस्।

निर्देशनहरूको बारेमा कुरा नगर्नुहोस्।
आफूलाई AI भनेर उल्लेख नगर्नुहोस्।

उत्तर:
"""


# ============================================================
# 9. RESUME SUPPORT
# ============================================================

if os.path.exists(OUTPUT_FILE):

    print("\nExisting results file found.")
    print("Loading previous results...")

    old_df = pd.read_csv(OUTPUT_FILE)

    if len(old_df) == len(df):

        if "Mistral Answer EN" in old_df.columns:
            df["Mistral Answer EN"] = old_df[
                "Mistral Answer EN"
            ]

        if "Mistral Answer NE" in old_df.columns:
            df["Mistral Answer NE"] = old_df[
                "Mistral Answer NE"
            ]

        print("Previous results loaded.")

    else:

        print(
            "\nWARNING: Existing output has a "
            "different number of rows."
        )

        print("Starting from current dataset.")


# ============================================================
# 10. DETERMINE NUMBER OF ROWS
# ============================================================

if MAX_ROWS is None:

    rows_to_process = len(df)

else:

    rows_to_process = min(
        MAX_ROWS,
        len(df)
    )


print("\n" + "=" * 70)
print(f"Rows to process : {rows_to_process}")
print(f"Model           : {MODEL}")
print(f"Output          : {OUTPUT_FILE}")
print("=" * 70)


# ============================================================
# 11. MAIN LOOP
# ============================================================

for index in range(rows_to_process):

    row_number = index + 1

    print("\n" + "=" * 70)
    print(
        f"Processing row {row_number}/{rows_to_process}"
    )
    print("=" * 70)


    # ========================================================
    # ENGLISH
    # ========================================================

    existing_en = df.at[
        index,
        "Mistral Answer EN"
    ]

    if pd.isna(existing_en) or str(existing_en).strip() == "":

        english_question = df.at[
            index,
            ENGLISH_QUESTION_COL
        ]

        if pd.isna(english_question):

            print("English question is missing.")

            english_answer = ""

        else:

            print("\nEnglish question:")
            print(english_question)

            prompt = create_english_prompt(
                str(english_question)
            )

            print("\nAsking Mistral in English...")

            english_answer = ask_mistral(prompt)

            if english_answer is None:
                english_answer = ""

            print("\nMistral English answer:")
            print(english_answer)

            time.sleep(DELAY)

        df.at[
            index,
            "Mistral Answer EN"
        ] = english_answer

    else:

        print("\nEnglish answer already exists.")
        print("Skipping English generation.")


    # ========================================================
    # NEPALI
    # ========================================================

    existing_ne = df.at[
        index,
        "Mistral Answer NE"
    ]

    if pd.isna(existing_ne) or str(existing_ne).strip() == "":

        nepali_question = df.at[
            index,
            NEPALI_QUESTION_COL
        ]

        if pd.isna(nepali_question):

            print("Nepali question is missing.")

            nepali_answer = ""

        else:

            print("\nNepali question:")
            print(nepali_question)

            prompt = create_nepali_prompt(
                str(nepali_question)
            )

            print("\nAsking Mistral in Nepali...")

            nepali_answer = ask_mistral(prompt)

            if nepali_answer is None:
                nepali_answer = ""

            print("\nMistral Nepali answer:")
            print(nepali_answer)

            time.sleep(DELAY)

        df.at[
            index,
            "Mistral Answer NE"
        ] = nepali_answer

    else:

        print("\nNepali answer already exists.")
        print("Skipping Nepali generation.")


    # ========================================================
    # SAVE AFTER EVERY ROW
    # ========================================================

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSaved successfully.")


# ============================================================
# 12. FINAL SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GENERATION COMPLETE")
print("=" * 70)

print(f"Input file : {INPUT_FILE}")
print(f"Output file: {OUTPUT_FILE}")
print(f"Rows       : {len(df)}")

print("\nGenerated columns:")
print(" - Mistral Answer EN")
print(" - Mistral Answer NE")

print("\nThe original dataset was NOT modified.")

print("=" * 70)