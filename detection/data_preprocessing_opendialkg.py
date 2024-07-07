import json
import random
random.seed(1234)

system_response_path = "../generation/outputs_opendialkg/simulated_responses.json"
generated_faithful_path = "../generation/outputs_opendialkg/faithful.json"
generated_hallucination_path = "../generation/outputs_opendialkg/hallucination.json"

output_dir = "data/opendialkg"


def add_spaces_to_repeated_substrings(sentence):
    """
    This function is used to resolve the formatting issue of HaluEval dataset where the knowledge source is
    concatenated without any separation marker.
    """
    i = 0

    # Initialize a list to hold the characters of the modified sentence
    modified_sentence_chars = []

    # Iterate through the sentence while the index is less than the length of the sentence
    while i < len(sentence):
        # Initialize a flag to indicate if a repetition was found
        repetition_found = False

        # Try all possible lengths for the repeated substring, starting from 1 to the maximum possible length
        for length in range(5, (len(sentence) - i)//2 + 1):
            # Extract the substring of the current length starting at the current index
            substring = sentence[i:i+length]

            # Check if the substring is immediately repeated
            if sentence.startswith(substring, i + length):
                # If a repetition is found, append a space before and after the repeated substring
                # Only append a space before if it's not the start of the string
                modified_sentence_chars.append(substring)
                modified_sentence_chars.append('. ')
                modified_sentence_chars.append(substring)

                # Update the index to jump past the repeated substring
                i += 2*length
                repetition_found = True
                break

        # If no repetition was found for any length, just add the current character and move to the next one
        if not repetition_found:
            modified_sentence_chars.append(sentence[i])
            i += 1

    # Convert the list of characters back into a final modified sentence
    modified_sentence = ''.join(modified_sentence_chars)
    return modified_sentence


data_origin = json.loads(open(system_response_path).read())
data_origin_faithful = json.loads(open(generated_faithful_path).read())
data_origin_hallucination = json.loads(open(generated_hallucination_path).read())


data_human = []
data_system = []
data_faithful = []
data_hallucination = []
data_halueval = []

for d in data_origin:
    if d["index"] < 2000:
        data_human.append({
            "index": d["index"],
            "grounding": d["grounding"], #add_spaces_to_repeated_substrings(d["grounding"]),
            "input": d["input"],
            "generated_text": d["ground_truth_output"]
        })
        data_system.append({
            "index": d["index"],
            "grounding": d["grounding"], #add_spaces_to_repeated_substrings(d["grounding"]),
            "input": d["input"],
            "generated_text": d["system_output"]
        })
        data_halueval.append({
            "index": d["index"],
            "grounding": d["grounding"], #add_spaces_to_repeated_substrings(d["grounding"]),
            "input": d["input"],
            "generated_text": d["hallucinated_response(HaluEval)"],
        })


for d in data_origin_faithful:
    if d["index"] >= 2000:
        continue
    data_faithful.append({
        "index": d["index"],
        "grounding": d["grounding"], #add_spaces_to_repeated_substrings(d["grounding"]),
        "input": d["input"],
        "generated_text": d["generated_faithful_output"]
    })


for d in data_origin_hallucination:
    if d["index"] >= 2000:
        continue

    data_hallucination.append({
            "index": d["index"],
            "grounding": d["grounding"], #add_spaces_to_repeated_substrings(d["grounding"]),
            "input": d["input"],
            "generated_text": d["generated_hallucination_output"]
        })


open(f"{output_dir}/train_human_written_responses.json", "w").write(json.dumps(data_human, indent="\t"))
open(f"{output_dir}/train_system.json", "w").write(json.dumps(data_system, indent="\t"))
open(f"{output_dir}/train_generated_faithful.json", "w").write(json.dumps(data_faithful, indent="\t"))
open(f"{output_dir}/train_generated_hallucination.json", "w").write(json.dumps(data_hallucination, indent="\t"))
open(f"{output_dir}/train_halueval.json", "w").write(json.dumps(data_halueval, indent="\t"))