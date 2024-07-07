import json
import pandas as pd
import git

git.Git("data/origin_data/").clone("https://github.com/google/BEGIN-dataset.git")
git.Git("data/origin_data/").clone("https://github.com/RUCAIBox/HaluEval.git")

begin_dev_paths = [
    "data/origin_data/BEGIN-dataset/topicalchat/begin_dev_tc.tsv",
    "data/origin_data/BEGIN-dataset/cmu-dog/begin_dev_cmu.tsv",
    "data/origin_data/BEGIN-dataset/wow/begin_dev_wow.tsv"
    ]

begin_test_paths = [
    "data/origin_data/BEGIN-dataset/topicalchat/begin_test_tc.tsv",
    "data/origin_data/BEGIN-dataset/cmu-dog/begin_test_cmu.tsv",
    "data/origin_data/BEGIN-dataset/wow/begin_test_wow.tsv"
    ]

def load_begin_csv(paths):
    data = []
    for path in paths:
        print(path)
        df = pd.read_csv(path, sep='\t', header=0).fillna('')
        for i in range(len(df)):
            knowledge = df.iloc[i]["knowledge"]
            message = df.iloc[i]["message"]
            response = df.iloc[i]["response"]
            label = df.iloc[i]["begin_label"]
            d = {
                "input": message,
                "system_output": response,
                "grounding": knowledge,
                "label": label,
                "index": len(data)
            }
            data.append(d)
    return data

begin_dev = load_begin_csv(begin_dev_paths)
begin_test = load_begin_csv(begin_test_paths)

open("data/begin_dev.json", "w").write(json.dumps(begin_dev, indent="\t"))
open("data/begin_test.json", "w").write(json.dumps(begin_test, indent="\t"))