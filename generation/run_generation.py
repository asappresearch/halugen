import time

from tools.args import ArgParser
from prompts import prompt_faithful, prompt_generic, prompt_hallucinate

import os
import json
import logging
from copy import deepcopy

def processing(args):

    total_cost = 0
    # load data
    data_list = json.loads(open(args.input_path).read())
    if args.num_input == -1:
        args.num_input = len(data_list)
    data_list = data_list[:args.num_input]

    output_path = f"{args.output}/{args.mode}.json"
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # try to load from the output file to skip generated indices.
    generated_data_list = []
    generated_outputs_dict = {}
    if os.path.exists(output_path):
        generated_data_list = json.loads(open(output_path).read())
        for d in generated_data_list:
            if d['index'] not in generated_outputs_dict:
                generated_outputs_dict[d['index']] = []
            generated_outputs_dict[d['index']].append(d)
            total_cost += d["cost (usd)"]

    # inference
    model_name = f"{args.mode}"
    if args.mode == "faithful":
        model = prompt_faithful.generate
    elif args.mode == "generic":
        model = prompt_generic.generate
    else:
        model = prompt_hallucinate.generate

    for i_d, d in enumerate(data_list):
        if i_d  >= args.num_input:
            continue

        logging.info(f"----- data {i_d} / {len(data_list)} -----")

        # skip if already exist in the output file in order to continue from where the program stopped previously.
        if i_d in generated_outputs_dict:
            continue

        logging.info(f">>>>>> data {i_d} / {len(data_list)} | generating >>>>>> ")
        count_try = 0
        while count_try < args.num_samples:
            detection_result = ""

            price = 0
            count_try += 1

            # hallucination generation
            output = ""
            target_text = d["system_output"]
            count_generate = 0
            while (output == "" or output == target_text) and count_generate < 5:
                try:
                    logging.info("attempt to generate")
                    output, price_ = model(
                        d['input'], d['grounding'], target_text, args.base_model, args.temp)
                    price += price_
                except:
                    time.sleep(10)
                count_generate += 1

            logging.info(f"data {i_d} / {len(data_list)}, system_output: {output}")
            logging.info(f"data {i_d} / {len(data_list)}, {model_name} {count_try}-th generation: {output}")


            # save data to output file
            generated_data_list.append(deepcopy(d))
            generated_data_list[-1]["index"] = i_d
            generated_data_list[-1]["method"] = model_name
            generated_data_list[-1][f"{model_name} generation cost (usd)"] = price
            generated_data_list[-1]["num_retries"] = count_try - 1
            generated_data_list[-1][f"generated_{model_name}_output"] = output
            total_cost += price
            open(output_path, "w").write(json.dumps(generated_data_list, indent=4))
        logging.info(f"<<<<<< generated by {model_name} <<<<<<")
        logging.info(f"-------total cost: USD {total_cost}--------")


if __name__ == "__main__":
    parser = ArgParser(description="Run synthetic generation")
    parser.add_argument(
        "--input_path",
        default="outputs_opendialkg/simulated_responses.json",
        help="you can either use simulated convos over OpenDialKG from 'outputs_opendialkg/simulated_responses.json'"
             "or use BEGIN dev set from '../data/begin_dev.json'"
    )
    parser.add_argument(
        "--num_input",
        default=-1,
        type=int,
        help="end data index to generate. The data[:num_input] are used"
    )
    parser.add_argument(
        "--num_samples",
        default=1,
        type=int,
        help="number of samples for each input, per model. default is 1"
    )
    parser.add_argument(
        "--mode",
        default="faithful",
        choices=["faithful", "generic", "hallucination"],
        help="generate one of the following synthetic outputs: faithful, generic, or hallucination"
    )
    parser.add_argument(
        "--temp",
        default=0.0,
        type=float,
        help="temperature to generate synthetic data"
    )
    parser.add_argument(
        "--base_model",
        default="gpt-4-1106-preview",
        choices=["gpt-4", "gpt-4-1106-preview", "gpt-4-0613", "gpt-4-0314",
                 "gpt-3.5-turbo", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613",
                 "gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613"]
    )
    parser.add_argument(
        "--output",
        default="outputs_opendialkg"
    )

    args = parser.parse_args()
    processing(args)