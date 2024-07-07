import json
import logging
import os
import random
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import torch
from peft import (LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model,
                  prepare_model_for_int8_training)
from torch import cuda
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          default_data_collator,
                          get_linear_schedule_with_warmup)

from tools.args import ArgParser

device = "cuda" if cuda.is_available() else "cpu"


class OpenDialKGDataset(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(self, data_list, tokenizer, source_len=2048, target_len=16):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
        """
        self.tokenizer = tokenizer
        self.data = data_list
        self.source_len = source_len
        self.target_len = target_len

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.data)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        response_text = (
            self.data[index]["generated_text"]
            if "generated_text" in self.data[index]
            else self.data[index]["system_output"]
        )
        source_text = f"Knowledge: {self.data[index]['grounding']}\nInput: {self.data[index]['input']}\nOutput: {response_text}\nOutput is faithful: "
        target_text = (
            "No" if self.data[index]["label"] == "hallucination" else "Yes"
        )  # 'Yes' - output is faithful. 'No' - output is hallucination.
        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.target_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
            "index": self.data[index]["index"],
        }


def train(epoch, tokenizer, model, device, loader, optimizer, lr_scheduler):
    """
    Function to be called for training with the parameters passed from main function

    """

    model.train()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        lm_labels = y.clone().detach()
        lm_labels[y == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)
        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            labels=lm_labels,
        )
        loss = outputs[0]

        if _ % 10 == 0:
            logging.info(f"epoch {epoch} step {_} loss {loss}")

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()


def inference(tokenizer, model, device, loader, output_path):
    """
    Function to evaluate model for predictions

    """
    predictions = []
    prediction_scores = []
    actuals = []
    indices = []
    durations = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(loader, 0)):
            # print(data)
            start_time = time.time()
            indices.extend([int(d) for d in data["index"]])

            ids = data["source_ids"].to(device, dtype=torch.long)
            mask = data["source_mask"].to(device, dtype=torch.long)
            y = data["target_ids"].to(device, dtype=torch.long)
            generated_outputs = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=2,
                return_dict_in_generate=True,
                output_scores=True,
            )
            generated_ids = generated_outputs["sequences"]
            preds = [
                tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for g in generated_ids
            ]
            logging.info(preds)
            predictions.extend(preds)
            scores = torch.exp(generated_outputs["scores"][-1][:, 2163]) / (
                torch.exp(generated_outputs["scores"][-1][:, 465])
                + torch.exp(generated_outputs["scores"][-1][:, 2163])
            )

            scores = scores.cpu().numpy()
            prediction_scores.extend([str(f) for f in list(scores)])

            target = [
                tokenizer.decode(
                    t, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for t in y
            ]
            durations.append(time.time() - start_time)
            actuals.extend(target)
            outputs = list(zip(indices, predictions, prediction_scores, actuals))
            open(output_path, "w").write(json.dumps(outputs, indent="\t"))
            logging.info(
                f"average inference time per example: {np.mean(durations)} seconds"
            )


def calc_metrics(pred_path):
    data = json.loads(open(pred_path).read())
    binary_preds = [float(d[2]) > 0.5 for d in data]
    binary_labels = [d[3] == "Yes" for d in data]

    acc = accuracy_score(binary_labels, binary_preds)
    logging.critical(f" accuracy  = {acc:.3f}")

    macro_prec, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        binary_labels, binary_preds, average="macro", zero_division=0
    )
    logging.critical(
        f" | (macro) p = {macro_prec:.3f}, r = {macro_recall:.3f}, f1  = {macro_f1:.3f}"
    )

    micro_prec, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        binary_labels, binary_preds, average="micro", zero_division=0
    )
    logging.critical(
        f" | (micro) p = {micro_prec:.3f}, r = {micro_recall:.3f}, f1  = {micro_f1:.3f}"
    )

    pos_prec, pos_recall, pos_f1, _ = precision_recall_fscore_support(
        binary_labels, binary_preds, pos_label=False, average="binary", zero_division=0
    )
    logging.critical(
        f" | (hallucination) p = {pos_prec:.3f}, r = {pos_recall:.3f}, f1  = {pos_f1:.3f}"
    )

    pos_prec, pos_recall, pos_f1, _ = precision_recall_fscore_support(
        binary_labels, binary_preds, pos_label=True, average="binary", zero_division=0
    )
    logging.critical(
        f" | (faithful) p = {pos_prec:.3f}, r = {pos_recall:.3f}, f1  = {pos_f1:.3f}"
    )


def run(args):
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    if args.mode == "train":

        tokenizer = AutoTokenizer.from_pretrained("t5-base")

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            "t5-base", load_in_8bit=True, device_map="auto"
        )
        model = prepare_model_for_int8_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        model = model.to(device)

        # Creation of Dataloaders for train
        negative_data_list = json.loads(open(args.train_negative_path).read())[
            : int(args.n)
        ]
        for d in negative_data_list:
            d["label"] = "hallucination"
        positive_data_list = json.loads(open(args.train_positive_path).read())[
            : int(args.n)
        ]
        for d in positive_data_list:
            d["label"] = "faithful"
        logging.info(f"positive data length {len(positive_data_list)}")
        logging.info(f"negative data length {len(negative_data_list)}")
        train_data_list = positive_data_list + negative_data_list
        random.shuffle(train_data_list)

        training_set = OpenDialKGDataset(train_data_list, tokenizer)

        train_params = {
            "batch_size": args.batch_size,
            "shuffle": True,
            "num_workers": 0,
        }

        training_loader = DataLoader(training_set, **train_params)

        # Creation of Dataloaders for test
        if args.test_path != "":
            test_data_list = json.loads(open(args.test_path).read())
            test_set = OpenDialKGDataset(test_data_list, tokenizer)

            test_params = {
                "batch_size": args.batch_size,
                "shuffle": False,
                "num_workers": 0,
            }
            test_loader = DataLoader(test_set, **test_params)

        # Defining the optimizer that will be used to tune the weights of the network in the training session.
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(training_loader) * args.epochs),
        )

        if args.test_path != "":
            logging.info("before training, testing")
            inference(
                tokenizer, model, device, test_loader, f"{args.output_path}.init.json"
            )
            calc_metrics(f"{args.output_path}.init.json")
        for epoch in range(args.epochs):
            train(
                epoch,
                tokenizer,
                model,
                device,
                training_loader,
                optimizer,
                lr_scheduler,
            )
            # Saving the model after training
            model.save_pretrained(f"{args.model_path}_epoch_{epoch}")
            tokenizer.save_pretrained(f"{args.model_path}_epoch_{epoch}")
            if args.test_path != "":
                logging.info(f"Epoch {epoch}, testing")
                inference(
                    tokenizer,
                    model,
                    device,
                    test_loader,
                    f"{args.output_path}.epoch_{epoch}.json",
                )
                calc_metrics(f"{args.output_path}.epoch_{epoch}.json")

    else:
        if args.model_path != "":
            # load trained model for validation
            # Load peft config for pre-trained checkpoint etc.
            config = PeftConfig.from_pretrained(args.model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

            # Load the Lora model
            model = PeftModel.from_pretrained(model, args.model_path)
            model = model.to(device)
            model.eval()

        test_data_list = json.loads(open(args.test_path).read())[:1000]
        test_set = OpenDialKGDataset(test_data_list, tokenizer)
        test_params = {
            "batch_size": args.batch_size,
            "shuffle": False,
            "num_workers": 0,
        }
        test_loader = DataLoader(test_set, **test_params)
        inference(tokenizer, model, device, test_loader, args.output_path)
        calc_metrics(args.output_path)


if __name__ == "__main__":
    parser = ArgParser(description="Run evaluation of hallucination detection")
    parser.add_argument(
        "--train_positive_path",
        default="",
        help="path to the data file which contains positive training data",
    )
    parser.add_argument(
        "--train_negative_path",
        default="",
        help="path to the data file which contains negative training data",
    )
    parser.add_argument(
        "--test_path", default="", help="path to the data file which contains test data"
    )
    parser.add_argument(
        "--model_path",
        default="models/opendialkg_lr_1e-4",
        help="path to the folder to save (if mode is 'train') / load (if mode is 'test') the model",
    )
    parser.add_argument(
        "--output_path",
        default="predictions/opendialkg_lr_1e-4",
        help="path to the json file for saving predictions",
    )
    parser.add_argument("-n", default=2000, type=int, help="number of data to train.")
    parser.add_argument("--mode", default="test", choices=["train", "test"])
    parser.add_argument("--seed", default=1234, type=int, help="random seed")
    parser.add_argument("--batch_size", default=4, type=int, help="batch size")
    parser.add_argument(
        "--epochs", default=5, type=int, help="number of epochs for training"
    )
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")

    args = parser.parse_args()
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
    run(args)
