import json
import random
random.seed(1234)

synthetic_hallucination_path = "../generation/outputs_begin/hallucination.json"
synthetic_faithful_path = "../generation/outputs_begin/faithful.json"
synthetic_generic_path = "../generation/outputs_begin/generic.json"

output_dir = "data/begin"

data_synthetic_hallucination_origin = json.loads(open(synthetic_hallucination_path).read())
data_synthetic_faithful_origin = json.loads(open(synthetic_faithful_path).read())
data_synthetic_generic_origin = json.loads(open(synthetic_generic_path).read())

data_test = []
data_dev = []
data_synthetic_hallucination = []
data_synthetic_faithful = []
data_synthetic_generic = []


for d in data_synthetic_faithful_origin:
    if "Dialogue History" not in d["generated_faithful_output"] and "Knowledge" not in d["generated_faithful_output"]:
        data_synthetic_faithful.append({
            "index": len(data_synthetic_faithful),
            "grounding": d["grounding"],
            "input": d["input"],
            "generated_text": d["generated_faithful_output"]
            })

for d in data_synthetic_hallucination_origin:
    if "Dialogue History" not in d["generated_hallucination_output"] and "Knowledge" not in d["generated_hallucination_output"]:

        data_synthetic_hallucination.append({
            "index": len(data_synthetic_hallucination),
            "grounding": d["grounding"],
            "input": d["input"],
            "generated_text": d["generated_hallucination_output"]
            })

for d in data_synthetic_generic_origin:
    if "Dialogue History" not in d["generated_generic_output"] and "Knowledge" not in d["generated_generic_output"]:

        data_synthetic_generic.append({
            "index": len(data_synthetic_generic),
            "grounding": d["grounding"],
            "input": d["input"],
            "generated_text": d["generated_generic_output"]
            })

open(f"{output_dir}/train_generated_faithful.json", "w").write(json.dumps(data_synthetic_faithful, indent="\t"))
open(f"{output_dir}/train_generated_hallucination.json", "w").write(json.dumps(data_synthetic_hallucination, indent="\t"))
open(f"{output_dir}/train_generated_generic.json", "w").write(json.dumps(data_synthetic_generic, indent="\t"))