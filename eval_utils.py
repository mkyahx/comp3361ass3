from io import StringIO
import re
import string
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import numpy as np
from tqdm.notebook import tqdm
import pandas as pd


def normalize_number_str(number_str: str) -> float:
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        return float("inf")


def split_string(
    s: str,
    char_list: list[str] = [",", ";"],
) -> list[str]:
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


def is_float(element) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


def normalize_str(input_str, remove_punct=True) -> str:
    no_spaces = re.sub(r"\s", "", input_str)
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    else:
        return no_spaces.lower()


def extract_numbers(text: str) -> List[str]:
    pattern = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    return [el.replace(",", "") for el in re.findall(pattern, text)]


def get_question_score(model_answer: str, ground_truth: str) -> bool:
    if is_float(ground_truth):
        normalized_answer = normalize_number_str(str(model_answer))
        return np.isclose(normalized_answer, float(ground_truth), rtol=1e-2, atol=1)

    elif any(char in ground_truth for char in [",", ";"]):
        gt_elems = split_string(ground_truth)
        ma_elems = split_string(str(model_answer))

        if len(gt_elems) != len(ma_elems):
            return False

        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if is_float(gt_elem):
                normalized_ma_elem = normalize_number_str(ma_elem.strip())
                comparisons.append(np.isclose(normalized_ma_elem, float(gt_elem), rtol=1e-2, atol=1))
            else:
                comparisons.append(
                    normalize_str(ma_elem, remove_punct=False) == normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)

    else:
        return normalize_str(str(model_answer)) == normalize_str(ground_truth)


def get_correct(row):
    source = row.get("source", "")
    if source in ["GSM8K", "MATH", "ARC"]:
        numbers_answer = extract_numbers(str(row["answer"]))
        if len(numbers_answer) == 0:
            return False
        return np.isclose(float(numbers_answer[-1]), float(row["true_answer"]), rtol=1e-5, atol=1e-7)
    else:  # SimpleQA, GAIA, etc.
        return get_question_score(str(row["answer"]), str(row["true_answer"]))


def score_answers_subset(answers_file):
    try:
        print(answers_file)
        df = pd.read_json(StringIO(open(answers_file, "r").read()), lines=True)
        df["correct"] = df.apply(get_correct, axis=1)
        acc = df["correct"].mean().item()
        result = df.loc[0, ["model_id", "agent_action_type", "source"]].to_dict()
        result["acc"] = acc
        result["total"] = len(df)
        result["correct_count"] = df["correct"].sum().item()
        return result
    except Exception as e:
        print(f"Error with {answers_file}: {e}")
        return None


def score_answers(answers_files):
    results = []
    with ThreadPoolExecutor(max_workers=16) as exe:
        futures = [
            exe.submit(score_answers_subset, answers_file) for answers_file in answers_files
        ]
        for f in tqdm(as_completed(futures), total=len(answers_files), desc="Processing tasks"):
            result = f.result()
            if result:
                results.append(result)
    df = pd.DataFrame(results)
    return df
