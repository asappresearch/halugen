# HaluGen

[https://github.com/asappresearch/halugen](https://github.com/asappresearch/halugen)

This repo contains code and data used for paper:

"Enhancing Hallucination Detection through Perturbation-Based Synthetic Data Generation in System Responses", ACL 2024 Findings

This repo contains two parts: first generate synthetic data, then use the generated data to finetune a hallucination detector.

Make sure to first run `python data_clone_preprocessing.py` to clone the BEGIN repo and HaluEval repo under `data/`

## Synthetic data generation

Synthetic data used in the paper has been saved under `generation/outputs_opendialkg/` and `generation/outputs_begin/`. 

If you prefer to regenerate the data, you can go through the following steps (we use opendialkg as an example, you can also load BEGIN dataset):

- Export your openai key to env

`export OPENAI_API_KEY=<your_openai_key>`

- Go to the subdirectory

`cd ./generation` 

- (Optional) OpenDialKG dataset only include human annotated responses and does not provide any system responses. To simulate responses from a chatbot system for opendialkg dataset, run `python run_opendialkg_simulator.py` which will cost around 35 USD to generate 10k responses.

- Generate synthetic responses based on system responses, which will cost around 180 USD for faithful and hallucination generation on OpenDialKG system responses: 

`python run_generation.py --input_path outputs_opendialkg/simulated_responses.json --mode faithful --output outputs_opendialkg`

`python run_generation.py --input_path outputs_opendialkg/simulated_responses.json --mode hallucination --output outputs_opendialkg`


## Finetune a T5 based hallucination detector 

In the following, we use opendialkg as an example, you can also load BEGIN dataset.

- Go to the subdirectory

`cd detection`

- Data preprocessing: convert original data and synthetic data into proper format for finetuning. 

`python data_preprocessing_opendialkg.py`

- T5-base Finetuning: 

`python run_t5_opendialkg.py --train_positive_path data/opendialkg/train_generated_faithful.json --train_negative_path data/opendialkg/train_generated_hallucination.json --test_path ../data/opendialkg_eval_dev.json --mode train`

- testing:

`python run_t5_opendialkg.py --test_path ../data/opendialkg_eval_test.json --model_path models/opendialkg_lr_1e-4_epoch_4 --output_path predictions/opendialkg_test --mode test`
