# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Create Sentence-BERT model for text matching task
"""

import math
import os
import sys

import pandas as pd
import torch
import torch.distributed as dist
from loguru import logger
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
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


def train_loop(global_rank, world_size):
    logger.info("global_rank: {}, world_size: {}".format(global_rank, world_size))
    # windows 用 gloo, linux 用 nccl
    dist.init_process_group(
        backend="gloo", init_method="tcp://localhost:23456", rank=global_rank, world_size=world_size
    )

    device = torch.device("cuda:{}".format(global_rank))
    sentence_bert_model = SentenceBertModel(
        model_name_or_path="bert-base-chinese", encoder_type="POOLER", device=device
    )
    # 神之偷懒
    self = sentence_bert_model

    # 从 huggingface 的数据集中加载数据
    # hf_dataset_name: str = "STS-B"
    # train_dataset = HFTextMatchingTrainDataset(self.tokenizer, hf_dataset_name, max_len=self.max_seq_length)
    # eval_dataset = HFTextMatchingTestDataset(self.tokenizer, hf_dataset_name, max_len=self.max_seq_length)

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

    os.makedirs(output_dir, exist_ok=True)
    logger.debug("Use pytorch device: {}".format(self.device))
    self.bert.to(self.device)
    set_seed(seed)

    # 同样是不使用 shuffle
    sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, sampler=sampler)
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

    # 该用 DDP 替换了. 先不用 find_unused_parameters
    self.bert = DDP(
        self.bert,
        device_ids=[global_rank],
        output_device=global_rank,
        find_unused_parameters=False,
        # 加了 broadcast_buffers 可以防止 RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
        broadcast_buffers=False,
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Batch size = {batch_size}")
    logger.info(f"  Num steps = {total_steps}")
    logger.info(f"  Warmup-steps: {warmup_steps}")

    logger.info("  Training started")
    global_step = 0
    self.bert.zero_grad()
    epoch_number = 0
    best_eval_metric = 0
    steps_trained_in_current_epoch = 0
    epochs_trained = 0

    # 这些步骤和 cosent 一样
    if self.model_name_or_path and os.path.exists(self.model_name_or_path):
        try:
            # set global_step to global_step of last saved checkpoint from model path
            checkpoint_suffix = self.model_name_or_path.split("/")[-1].split("-")
            if len(checkpoint_suffix) > 2:
                checkpoint_suffix = checkpoint_suffix[1]
            else:
                checkpoint_suffix = checkpoint_suffix[-1]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // gradient_accumulation_steps)
            logger.info("   Continuing training from checkpoint, will skip to saved global_step")
            logger.info("   Continuing training from epoch %d" % epochs_trained)
            logger.info("   Continuing training from global step %d" % global_step)
            logger.info("   Will skip the first %d steps in the current epoch" % steps_trained_in_current_epoch)
        except ValueError:
            logger.info("   Starting fine-tuning.")

    # 判断下当前是否主进程, 只在主进程下打印日志和保存模型等
    is_main_process = global_rank == -1 or dist.get_rank() == 0
    training_progress_scores = {
        "global_step": [],
        "train_loss": [],
        "eval_spearman": [],
        "eval_pearson": [],
    }
    for current_epoch in trange(int(num_epochs), desc="Epoch", disable=not is_main_process, mininterval=0):
        self.bert.train()
        current_loss = 0
        if epochs_trained > 0:
            epochs_trained -= 1
            continue
        batch_iterator = tqdm(
            train_dataloader,
            desc=f"Running Epoch {epoch_number + 1} of {num_epochs}",
            disable=not is_main_process,
            mininterval=0,
        )
        for step, batch in enumerate(batch_iterator):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            # 当需要调试的时候, 可以设置为 True, 但是会影响性能
            with torch.autograd.set_detect_anomaly(False):
                # 输入居然是不一样的, 现在有三个输入了, 需要重新确认下数据集
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

                # get sentence embeddings of BERT encoder
                # https://github.com/huggingface/transformers/issues/7848
                source_embeddings = self.get_sentence_embeddings(
                    source_input_ids, source_attention_mask, source_token_type_ids
                )
                target_embeddings = self.get_sentence_embeddings(
                    target_input_ids, target_attention_mask, target_token_type_ids
                )
                # shape: (batch * 2, seq_len) 合并了两个输入, 训练会更快些
                # input_ids = torch.cat([source_input_ids, target_input_ids], dim=0)
                # attention_mask = torch.cat([source_attention_mask, target_attention_mask], dim=0)
                # token_type_ids = torch.cat([source_token_type_ids, target_token_type_ids], dim=0)
                # embeddings = self.get_sentence_embeddings(input_ids, attention_mask, token_type_ids)
                # source_embeddings, target_embeddings = torch.split(embeddings, source_input_ids.size(0), dim=0)
                # source_embeddings, target_embeddings = torch.chunk(embeddings, 2, dim=0)
                # 结合了两个输出
                logits = self.concat_embeddings(source_embeddings, target_embeddings)
                loss = self.calc_loss(labels, logits)
                current_loss = loss.item()
                if verbose:
                    batch_iterator.set_description(
                        f"Epoch: {epoch_number + 1}/{num_epochs}, Batch:{step}/{len(train_dataloader)}, Loss: {current_loss:9.4f}"
                    )

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.bert.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    optimizer.zero_grad()
                    global_step += 1
        epoch_number += 1

        if is_main_process:
            output_dir_current = os.path.join(output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number))
            results = self.eval_model(eval_dataset, output_dir_current, verbose=verbose, batch_size=batch_size)
            self.save_model(output_dir_current, model=self.bert, results=results)
            training_progress_scores["global_step"].append(global_step)
            training_progress_scores["train_loss"].append(current_loss)
            for key in results:
                training_progress_scores[key].append(results[key])
            report = pd.DataFrame(training_progress_scores)
            report.to_csv(os.path.join(output_dir, "training_progress_scores.csv"), index=False)

            eval_spearman = results["eval_spearman"]
            if eval_spearman > best_eval_metric:
                best_eval_metric = eval_spearman
                logger.info(f"Save new best model, best_eval_metric: {best_eval_metric}")
                self.save_model(output_dir, model=self.bert, results=results)

        if 0 < max_steps < global_step:
            return global_step, training_progress_scores

    return global_step, training_progress_scores


if __name__ == "__main__":
    # 使用 mp.spawn 启动多进程
    world_size = torch.cuda.device_count()
    mp.spawn(train_loop, nprocs=world_size, args=(world_size,))
