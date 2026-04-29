import os
import ast
import pandas as pd
from typing import List
from huggingface_hub import create_tag
from datasets import load_dataset, Dataset

def fix_data_format(
        dataset: pd.DataFrame,
        reformat_columns: List[str] = ['generations', 'model_names', 'ratings', 'rationales', 'distilabel_metadata']
        ) -> pd.DataFrame:
    """
    Reformats specified columns in a Pandas DataFrame by converting string
    representations of Python objects (lists, dictionaries) into actual Python objects.
    This is particularly useful when loading data where complex data types were
    saved as strings.

    Args:
        dataset (pd.DataFrame): The input DataFrame containing the columns to reformat.
        reformat_columns (List[str]): A list of column names that should be
                                       reformatted. The function will attempt to
                                       convert string representations (e.g., '["a", "b"]'
                                       or '{"key": "value"}') into actual Python objects
                                       using `ast.literal_eval`.

    Returns:
        pd.DataFrame: The DataFrame with the specified columns reformatted.
    """

    print("ℹ️ Reformatting columns to Python objects...")
    # Apply ast.literal_eval to convert string representations to actual Python objects
    for col in reformat_columns:
        if col in dataset.columns:
            print(f"  - Processing column: '{col}' (current type: {dataset[col].dtype})")
            # Use ast.literal_eval, handling potential NaNs gracefully
            dataset[col] = dataset[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) and (x.startswith('[') or x.startswith('{')) else x
            )
            print(f"    --> Fixed type: {dataset[col].apply(type).iloc[0] if not dataset[col].empty else 'empty'}")

    return dataset

def push_and_tag(artifact, repo_id: str, artifact_type: str, version_tag: str, commit_msg: str ='Placeholder'):
    # -- Push the data to the 'main' branch --
    print(f"ℹ️Pushing to {repo_id}")
    artifact.push_to_hub(repo_id, private=False, commit_message=commit_msg)

    # -- Create a version tag for this specific commit --
    print(f"Creating tag {version_tag}")
    create_tag(
        repo_id=repo_id,
        tag=version_tag,
        repo_type=artifact_type,
        tag_message=f"Release of {version_tag}"
        )
    print("✅Done!")
