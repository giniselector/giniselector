import json
import os
from typing import Any, Dict

import pandas as pd


def str_to_dict(string: str) -> Dict[str, Any]:
    return json.loads(string)


def append_results_to_file(results, filename="results.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    results = {k: [v] for k, v in results.items()}
    results = pd.DataFrame.from_dict(results, orient="columns")

    if not os.path.isfile(filename):
        results.to_csv(filename, header=True, index=False)
    else:  # it exists, so append without writing the header
        results.to_csv(filename, mode="a", header=False, index=False)
