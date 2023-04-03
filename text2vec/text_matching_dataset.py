# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
from torch.utils.data import Dataset
from loguru import logger
from transformers import PreTrainedTokenizer
from datasets import load_dataset


def load_train_data(path):
    data = []
    if not os.path.isfile(path):
        return data
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip().split("\t")
            # 需要有三列
            if len(line) != 3:
                logger.warning(f"line size not match, pass: {line}")
                continue
            # 最后一列是分数
            score = int(line[2])
            if "STS" in path.upper():
                # 有 STS-B 任务, 需要转换成 0/1 标签
                score = int(score > 2.5)
            data.append((line[0], line[1], score))
    return data


def load_test_data(path):
    data = []
    if not os.path.isfile(path):
        return data
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip().split("\t")
            if len(line) != 3:
                logger.warning(f"line size not match, pass: {line}")
                continue
            # 读取测试集的时候, 就是直接将分数转成 int, 没了 sts 的判断过程
            data.append((line[0], line[1], int(line[2])))
    return data


class TextMatchingTrainDataset(Dataset):
    """文本匹配训练数据集, 重写__getitem__和__len__方法"""

    def __init__(self, tokenizer: PreTrainedTokenizer, data: list, max_len: int = 64):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        return self.tokenizer(text, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")

    def __getitem__(self, index: int):
        line = self.data[index]
        return self.text_2_id(line[0]), self.text_2_id(line[1]), line[2]


class TextMatchingTestDataset(Dataset):
    """文本匹配测试数据集, 重写__getitem__和__len__方法
    没看出来和 TextMatchingTrainDataset 有什么区别
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, data: list, max_len: int = 64):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        return self.tokenizer(text, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")

    def __getitem__(self, index: int):
        line = self.data[index]
        return self.text_2_id(line[0]), self.text_2_id(line[1]), line[2]


class HFTextMatchingTrainDataset(Dataset):
    """Load HuggingFace datasets to SBERT format

    Args:
        tokenizer (PreTrainedTokenizer): tokenizer
        name (str): dataset name
        max_len (int): max length of sentence
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, name="STS-B", max_len: int = 64):
        self.tokenizer = tokenizer
        # 数据集还是同一份, 和 cosent 的数据集是一样的, 即 HFCosentTrainDataset 类
        self.data = load_dataset("shibing624/nli_zh", name.upper(), split="train")
        self.max_len = max_len
        self.name = name.upper()

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        return self.tokenizer(text, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")

    def __getitem__(self, index: int):
        line = self.data[index]
        # STS-B convert to 0/1 label
        return (
            self.text_2_id(line["sentence1"]),
            self.text_2_id(line["sentence2"]),
            int(line["label"] > 2.5) if "STS" in self.name else line["label"],
        )


class HFTextMatchingTestDataset(Dataset):
    """Load HuggingFace datasets to SBERT format

    Args:
        tokenizer (PreTrainedTokenizer): tokenizer
        name (str): dataset name
        max_len (int): max length of sentence
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, name="STS-B", max_len: int = 64, split="validation"):
        self.tokenizer = tokenizer
        # 用了不同的 split, 测试集用的是 validation
        self.data = load_dataset("shibing624/nli_zh", name.upper(), split=split)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        return self.tokenizer(text, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")

    def __getitem__(self, index: int):
        line = self.data[index]
        return self.text_2_id(line["sentence1"]), self.text_2_id(line["sentence2"]), line["label"]


"""
预先 tokenized 的数据集
"""


def my_load_dataset(path):
    data = {
        "input1": [],
        "input2": [],
        "label": [],
    }
    if not os.path.isfile(path):
        return data
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip().split("\t")
            # 需要有三列
            if len(line) != 3:
                logger.warning(f"line size not match, pass: {line}")
                continue
            # 最后一列是分数
            score = int(line[2])
            if "STS" in path.upper():
                # 有 STS-B 任务, 需要转换成 0/1 标签
                score = int(score > 2.5)
            data["input1"].append(line[0])
            data["input2"].append(line[1])
            data["label"].append(score)
    return data


class MyTextMatchingTrainDataset(Dataset):
    """文本匹配训练数据集, 重写__getitem__和__len__方法"""

    def __init__(self, tokenizer: PreTrainedTokenizer, data: list, max_len: int = 64):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len

        # 预先分词好
        self.data["input1"] = [self.text_2_id(text) for text in self.data["input1"]]
        self.data["input2"] = [self.text_2_id(text) for text in self.data["input2"]]

    def __len__(self):
        return len(self.data["label"])

    def text_2_id(self, text: str):
        return self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    def __getitem__(self, index: int):
        return self.data["input1"][index], self.data["input2"][index], self.data["label"][index]
