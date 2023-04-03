import math
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from loguru import logger
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

# 需要将根目录加进来, 即 D:\code\github\text2vec
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from text2vec.sentence_model import SentenceModel
from text2vec.text_matching_dataset import (
    HFTextMatchingTestDataset,
    HFTextMatchingTrainDataset,
    TextMatchingTestDataset,
    TextMatchingTrainDataset,
    load_test_data,
    load_train_data,
)
from text2vec.utils.stats_util import set_seed


class SentenceBertModel(SentenceModel):
    def __init__(
        self,
        model_name_or_path: str = "hfl/chinese-macbert-base",
        encoder_type: str = "MEAN",
        max_seq_length: int = 128,
        num_classes: int = 2,
        device: str = None,
    ):
        """
        Initializes a SentenceBert Model.

        Args:
            model_name_or_path: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            encoder_type: encoder type, set by model name
            max_seq_length: The maximum total input sequence length after tokenization.
            num_classes: Number of classes for classification.
            device: CPU or GPU
        """
        super().__init__(model_name_or_path, encoder_type, max_seq_length, device)
        # 多了一个分类器
        self.classifier = nn.Linear(self.bert.config.hidden_size * 3, num_classes).to(self.device)

    def __str__(self):
        return (
            f"<SentenceBertModel: {self.model_name_or_path}, encoder_type: {self.encoder_type}, "
            f"max_seq_length: {self.max_seq_length}>"
        )

    def concat_embeddings(self, source_embeddings, target_embeddings):
        """
        Output the bert sentence embeddings, pass to classifier module. Applies different
        concats and finally the linear layer to produce class scores
        :param source_embeddings:
        :param target_embeddings:
        :return: embeddings
        """
        # (u, v, |u - v|)
        embs = [source_embeddings, target_embeddings, torch.abs(source_embeddings - target_embeddings)]
        # input_embs shape: [batch_size, 3 * hidden_size]
        input_embs = torch.cat(embs, 1)
        # fc layer
        # logits shape: [batch_size, num_classes]
        logits = self.classifier(input_embs)
        return logits

    def calc_loss(self, y_true, y_pred):
        """
        交叉熵
        Calc loss with two sentence embeddings, Softmax loss
        """
        loss = nn.CrossEntropyLoss()(y_pred, y_true)
        return loss


def profile_main():
    """
    看看性能瓶颈在哪里
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentence_bert_model = SentenceBertModel(
        model_name_or_path="bert-base-chinese", encoder_type="POOLER", device=device
    )
    # 神之偷懒
    self = sentence_bert_model

    # 从本地加载数据
    train_file = r"D:\code\github\text2vec\examples\data\STS-B\STS-B.train.data"
    eval_file = r"D:\code\github\text2vec\examples\data\STS-B\STS-B.valid.data"
    train_dataset = TextMatchingTrainDataset(self.tokenizer, load_train_data(train_file), self.max_seq_length)
    eval_dataset = TextMatchingTestDataset(self.tokenizer, load_test_data(eval_file), self.max_seq_length)

    # 参数设置
    batch_size: int = 8
    num_epochs: int = 1
    weight_decay: float = 0.01
    seed: int = 42
    warmup_ratio: float = 0.1
    lr: float = 2e-5
    eps: float = 1e-6
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    max_steps: int = -1
    output_dir: str = "./temp"
    verbose: bool = True

    self.bert.to(self.device)
    set_seed(seed)

    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, pin_memory=True)
    total_steps = len(train_dataloader) * num_epochs
    param_optimizer = list(self.bert.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    warmup_steps = math.ceil(total_steps * warmup_ratio)  # by default 10% of _train data for warm-up
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    self.bert.train()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True,
        with_modules=True,
    ) as prof:
        count = 0
        for step, batch in enumerate(train_dataloader):
            with record_function("# get input data"):
                source, target, labels = batch
                # source        [batch, 1, seq_len] -> [batch, seq_len]
                source_input_ids = source.get("input_ids").squeeze(1).to(self.device)
                source_attention_mask = source.get("attention_mask").squeeze(1).to(self.device)
                source_token_type_ids = source.get("token_type_ids").squeeze(1).to(self.device)
                # target        [batch, 1, seq_len] -> [batch, seq_len]
                target_input_ids = target.get("input_ids").squeeze(1).to(self.device)
                target_attention_mask = target.get("attention_mask").squeeze(1).to(self.device)
                target_token_type_ids = target.get("token_type_ids").squeeze(1).to(self.device)
                labels = labels.to(self.device)

            with record_function("# forward"):
                source_embeddings = self.get_sentence_embeddings(
                    source_input_ids, source_attention_mask, source_token_type_ids
                )
                target_embeddings = self.get_sentence_embeddings(
                    target_input_ids, target_attention_mask, target_token_type_ids
                )

            with record_function("# calc loss"):
                logits = self.concat_embeddings(source_embeddings, target_embeddings)
                loss = self.calc_loss(labels, logits)

            with record_function("# backward"):
                loss.backward()

            with record_function("# optimizer"):
                torch.nn.utils.clip_grad_norm_(self.bert.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad()

            # 循环 10 次
            count += 1
            if count >= 10:
                break

    # 打印性能报告
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))


if __name__ == "__main__":
    profile_main()
