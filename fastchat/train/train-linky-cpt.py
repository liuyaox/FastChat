# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from fastchat.conversation import get_conv_template

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False
    data_format: str = "vicuna"


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()


def preprocess(
    sources, tokenizer: transformers.PreTrainedTokenizer, data_format: str = "vicuna"
) -> Dict:
    # Apply prompt templates
    conversations = []
    if data_format == "koala_dialogue":
        conv = get_conv_template("koala_v0.2")
        roles = {
            "user": conv.roles[0],
            "assistant": conv.roles[1],
            "system": conv.roles[2],
        }
        for i, source in enumerate(sources):
            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())
    else:
        conv = get_conversation_template("vicuna")
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    if data_format == "koala_dialogue":
        assert conv.sep_style == SeparatorStyle.KOALA_DIALOGUE
    else:
        assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))
            exit()

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                # todo: mismatch
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def preprocess_pretrain(
    sources, tokenizer: transformers.PreTrainedTokenizer, data_format: str = "vicuna"
) -> Dict:
    conversaiotns = []
    conv = get_conv_template("pretrain")
    for i, source in enumerate(sources):    # YAO：sources是一个只包含1个元素的列表
        conv.messages = []
        for j, sentence in enumerate(source):   # YAO: source是一个只包含半轮对话(assistant:xxx)的列表
            conv.append_message(sentence["from"], sentence["value"])
        conversaiotns.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversaiotns,              # YAO: 注意，conversations来自上面的source，它会被max_length截断或padding！
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    input_ids, targets = process_pretrain_tensor(
        input_ids, tokenizer.pad_token_id, IGNORE_TOKEN_ID
    )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def process_pretrain_tensor(input_ids, pad_token_id, IGNORE_TOKEN_ID):
    raw_input_ids = input_ids[0].tolist()

    num_two = raw_input_ids.count(2)
    if num_two % 2 != 0:
        raw_input_ids[-1] = 2
        num_two += 1

    original_len = len(raw_input_ids)
    target_ids, new_input_ids = [], []
    flag = False
    for i, num in enumerate(raw_input_ids):
        if num == 2:
            if flag:
                new_input_ids.append(num)
                flag = False
                target_ids.append(num)
            else:
                flag = True
        else:
            if flag:
                target_ids.append(num)
            else:
                target_ids.append(IGNORE_TOKEN_ID)
            new_input_ids.append(num)

    new_input_ids = new_input_ids + [pad_token_id] * (original_len - len(new_input_ids))
    target_ids = target_ids + [IGNORE_TOKEN_ID] * (original_len - len(target_ids))

    input_tensor = torch.tensor(new_input_ids, dtype=input_ids.dtype)
    target_tensor = torch.tensor(target_ids, dtype=input_ids.dtype)
    input_tensor = input_tensor.unsqueeze(0)
    target_tensor = target_tensor.unsqueeze(0)

    return input_tensor, target_tensor


def preprocess_only_assistant(
    sources, tokenizer: transformers.PreTrainedTokenizer, data_format: str = "vicuna"
) -> Dict:
    # Apply prompt templates
    conversations = []
    mask_list = []
    conv = get_conv_template("only_assistant")
    roles = {"user": conv.roles[0], "assistant": conv.roles[1], "system": conv.roles[2]}
    for i, source in enumerate(sources):
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            conv.append_message(role, sentence["value"])
            # {
            #     "from": "assistant",
            #     "value": "Greetings ...",
            #     "ignore": true
            # }
            if sentence["from"] == "assistant":
                if "ignore" in sentence and sentence["ignore"]:
                    mask_list.append(False)
                else:
                    mask_list.append(True)
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    input_ids, targets = process_tensor(
        input_ids, tokenizer.pad_token_id, IGNORE_TOKEN_ID, mask_list
    )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def process_tensor(input_ids, pad_token_id, IGNORE_TOKEN_ID, mask_list=None):
    raw_input_ids = input_ids[0].tolist()

    num_two = raw_input_ids.count(2)
    if num_two % 2 != 0:
        raw_input_ids[-1] = 2
        num_two += 1

    original_len = len(raw_input_ids)
    target_ids, new_input_ids = [], []
    flag = False
    mask_idx = 0
    for i, num in enumerate(raw_input_ids):
        if num == 2:
            if flag:
                new_input_ids.append(num)
                flag = False
                if mask_list and mask_list[mask_idx]:
                    target_ids.append(num)
                else:
                    target_ids.append(IGNORE_TOKEN_ID)
            else:
                flag = True
        else:
            if flag:
                if mask_list and mask_list[mask_idx]:
                    target_ids.append(num)
                else:
                    target_ids.append(IGNORE_TOKEN_ID)
            else:
                target_ids.append(IGNORE_TOKEN_ID)
            new_input_ids.append(num)

        if not flag and num == 2:
            mask_idx += 1

    new_input_ids = new_input_ids + [pad_token_id] * (original_len - len(new_input_ids))
    target_ids = target_ids + [IGNORE_TOKEN_ID] * (original_len - len(target_ids))

    input_tensor = torch.tensor(new_input_ids, dtype=input_ids.dtype)
    target_tensor = torch.tensor(target_ids, dtype=input_ids.dtype)
    input_tensor = input_tensor.unsqueeze(0)
    target_tensor = target_tensor.unsqueeze(0)

    return input_tensor, target_tensor


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data,
        tokenizer: transformers.PreTrainedTokenizer,
        data_format: str = "vicuna",
    ):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.data_format = data_format

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        if self.data_format == "sft":
            ret = preprocess_only_assistant(
                [self.raw_data[i]["conversations"]], self.tokenizer, self.data_format
            )
        elif self.data_format == "pretrain":
            # YAO: 预训练数据，也是每条样本单独取出、单独preprocess
            # YAO: preprocess_pretrain第1个输入参数是列表，但列表只有1个元素，即1个conversations（1个小说的某个片段）
            ret = preprocess_pretrain(
                [self.raw_data[i]["conversations"]], self.tokenizer, self.data_format
            )
        elif self.data_format == "mix":
            if "type" in self.raw_data[i] and self.raw_data[i]["type"] == "pretrain":
                ret = preprocess_pretrain(
                    [self.raw_data[i]["conversations"]],
                    self.tokenizer,
                    self.data_format,
                )
            else:
                ret = preprocess_only_assistant(
                    [self.raw_data[i]["conversations"]],
                    self.tokenizer,
                    self.data_format,
                )
        else:
            ret = preprocess(
                [self.raw_data[i]["conversations"]], self.tokenizer, self.data_format
            )
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(
        train_json, tokenizer=tokenizer, data_format=data_args.data_format
    )

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(
            eval_json, tokenizer=tokenizer, data_format=data_args.data_format
        )
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    if local_rank == 0:
        print("data_module: ", data_module)
        sample_0 = data_module["train_dataset"][0]
        print("sample_0: ", sample_0)
        print(
            f"input_ids: {print_using_tokens(sample_0['input_ids'], tokenizer.pad_token_id)}"
        )
        print(f"labels: {print_using_tokens(sample_0['labels'], IGNORE_TOKEN_ID)}")

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    model.config.use_cache = True
    trainer.save_state()
    trainer_save_model_safe(trainer)


def debug_train():
    global local_rank
    model_args, data_args, training_args = (
        ModelArguments(),
        DataArguments(),
        TrainingArguments(output_dir="koala_vicuna1.5_13b"),
    )
    local_rank = training_args.local_rank

    model_args = ModelArguments(
        model_name_or_path="/mnt/data/gyl/pretrained_models/vicuna-13b-v1.5"
    )
    data_args = DataArguments(
        data_path="/mnt/data/gyl/data/koala-sft/v0.1-mix/train_filter.json",
        data_format="mix",
        lazy_preprocess=True,
    )
    training_args = TrainingArguments(
        bf16=True,
        output_dir="koala_vicuna1.5_13b",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        eval_steps=15,
        save_strategy="steps",
        save_steps=15,
        save_total_limit=8,
        learning_rate=2e-5,
        weight_decay=0.0,
        warmup_ratio=0.04,
        lr_scheduler_type="cosine",
        logging_steps=1,
        fsdp="full_shard auto_wrap offload",
        fsdp_transformer_layer_cls_to_wrap="LlamaDecoderLayer",
        tf32=True,
        model_max_length=4096,
        gradient_checkpointing=True,
    )

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    for i in range(0, max(10, len(data_module["train_dataset"]))):
        print(f"***************** {i} *****************")
        cur_data = data_module["train_dataset"][i]
        input_ids = print_using_tokens(
            cur_data["input_ids"].tolist(), tokenizer.pad_token_id
        )
        label_ids = print_using_tokens(cur_data["labels"].tolist(), IGNORE_TOKEN_ID)
        print(f"input_ids: {input_ids}")
        print(f"label_ids: {label_ids}")
        print(f"input string: {tokenizer.decode(input_ids)}")
        output_label_ids = split_and_filter(label_ids)
        for cur_label_ids in output_label_ids:
            print(f"cur_label_ids: {cur_label_ids}")
            print(f"cur_label string: {tokenizer.decode(cur_label_ids)}")


def split_and_filter(input_list):
    result = []
    temp_list = []

    for item in input_list:
        if item != -100:
            temp_list.append(item)
        elif temp_list:
            result.append(temp_list)
            temp_list = []

    if temp_list:
        result.append(temp_list)

    return result


def print_using_tokens(original_list: list, ignore_id: int) -> list:
    n = 0
    for x in reversed(original_list):
        if x == ignore_id:
            n += 1
        else:
            break

    if n > 0:
        new_list = original_list[:-n]
    else:
        new_list = original_list[:]

    return new_list


if __name__ == "__main__":
    debug_train()
