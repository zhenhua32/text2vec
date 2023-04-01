# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Create CoSENT model for text matching task
"""

import math
import os

import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm, trange
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from text2vec.cosent_dataset import CosentTrainDataset, load_cosent_train_data, HFCosentTrainDataset
from text2vec.sentence_model import SentenceModel
from text2vec.text_matching_dataset import TextMatchingTestDataset, load_test_data, HFTextMatchingTestDataset
from text2vec.utils.stats_util import set_seed


class CosentModel(SentenceModel):
    def __init__(
            self,
            model_name_or_path: str = "hfl/chinese-macbert-base",
            encoder_type: str = "FIRST_LAST_AVG",
            max_seq_length: int = 128,
            device: str = None,
    ):
        """
        Initializes a CoSENT Model.
        模型主要先看这块

        Args:
            model_name_or_path: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            encoder_type: Enum of type EncoderType.
            max_seq_length: The maximum total input sequence length after tokenization.
            device: The device on which the model is allocated.
        """
        # 模型已经在这里加载了
        super().__init__(model_name_or_path, encoder_type, max_seq_length, device)

    def __str__(self):
        return f"<CoSENTModel: {self.model_name_or_path}, encoder_type: {self.encoder_type}, " \
               f"max_seq_length: {self.max_seq_length}>"

    def train_model(
            self,
            train_file: str = None,
            output_dir: str = None,
            eval_file: str = None,
            verbose: bool = True,
            batch_size: int = 32,
            num_epochs: int = 1,
            weight_decay: float = 0.01,
            seed: int = 42,
            warmup_ratio: float = 0.1,
            lr: float = 2e-5,
            eps: float = 1e-6,
            gradient_accumulation_steps: int = 1,
            max_grad_norm: float = 1.0,
            max_steps: int = -1,
            use_hf_dataset: bool = False,
            hf_dataset_name: str = "STS-B",
    ):
        """
        是用来训练模型的
        Trains the model on 'train_file'

        Args:
            train_file: Path to text file containing the text to _train the language model on.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            eval_file (optional): Path to eval file containing the text to _evaluate the language model on.
            verbose (optional): Print logger or not.
            batch_size (optional): Batch size for training.
            num_epochs (optional): Number of epochs for training.
            weight_decay (optional): Weight decay for optimization.
            seed (optional): Seed for initialization.
            warmup_ratio (optional): Warmup ratio for learning rate.
            lr (optional): Learning rate.
            eps (optional): Adam epsilon.
            gradient_accumulation_steps (optional): Number of updates steps to accumulate before performing a backward/update pass.
            max_grad_norm (optional): Max gradient norm.
            max_steps (optional): If > 0: set total number of training steps to perform. Override num_epochs.
            use_hf_dataset (optional): Whether to use the HF dataset.
            hf_dataset_name (optional): Name of the dataset to use for the HuggingFace datasets.
        Returns:
            global_step: Number of global steps trained
            training_details: full training progress scores
        """
        # 加载数据集
        if use_hf_dataset and hf_dataset_name:
            logger.info(
                f"Train_file will be ignored when use_hf_dataset is True, load HF dataset: {hf_dataset_name}")
            train_dataset = HFCosentTrainDataset(self.tokenizer, hf_dataset_name, max_len=self.max_seq_length)
            eval_dataset = HFTextMatchingTestDataset(self.tokenizer, hf_dataset_name, max_len=self.max_seq_length)
        elif train_file is not None:
            logger.info(
                f"Hf_dataset_name: {hf_dataset_name} will be ignored when use_hf_dataset is False, load train_file: {train_file}")
            train_dataset = CosentTrainDataset(self.tokenizer, load_cosent_train_data(train_file), self.max_seq_length)
            eval_dataset = TextMatchingTestDataset(self.tokenizer, load_test_data(eval_file), self.max_seq_length)
        else:
            raise ValueError("Error, train_file|use_hf_dataset must be specified")

        # 调用真正的训练过程
        global_step, training_details = self.train(
            train_dataset,
            output_dir,
            eval_dataset=eval_dataset,
            verbose=verbose,
            batch_size=batch_size,
            num_epochs=num_epochs,
            weight_decay=weight_decay,
            seed=seed,
            warmup_ratio=warmup_ratio,
            lr=lr,
            eps=eps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps
        )
        logger.info(f" Training model done. Saved to {output_dir}.")

        return global_step, training_details

    def calc_loss(self, y_true, y_pred):
        """
        https://kexue.fm/archives/8847
        矩阵计算batch内的cos loss
        y_true: (batch_size, )
        y_pred: (batch_size, hidden_size)
        """
        # 1. 取出真实的标签
        # shape change from (batch_size, ) to (batch_size // 2, )
        y_true = y_true[::2]  # tensor([1, 0, 1]) 真实的标签
        # 2. 对输出的句子向量进行l2归一化   后面只需要对应为相乘  就可以得到cos值了
        # norms shape: (batch_size, 1)
        norms = (y_pred ** 2).sum(axis=1, keepdims=True) ** 0.5
        # L2范数是一种衡量向量大小的方法，它定义为向量的各个元素的平方和的平方根。
        # y_pred shape: (batch_size, hidden_size)
        y_pred = y_pred / norms
        # 3. 奇偶向量相乘, 相似度矩阵除以温度系数0.05(等于*20)
        # shape chang list:
        # y_pred[::2] shape: (batch_size // 2, hidden_size) 原文中的第一列文本
        # y_pred[1::2] shape: (batch_size // 2, hidden_size) 原文中的第二列文本
        # torch.sum shape: (batch_size // 2, )
        y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * 20
        # 4. 取出负例-正例的差值
        # y_pred[:, None] shape: (batch_size // 2, 1), y_pred[None, :] shape: (1, batch_size // 2)
        # y_pred shape: (batch_size // 2, batch_size // 2)
        y_pred = y_pred[:, None] - y_pred[None, :]  # 这里是算出所有位置 两两之间余弦的差值
        # 矩阵中的第i行j列  表示的是第i个余弦值-第j个余弦值
        # y_true shape: (batch_size // 2, batch_size // 2)
        y_true = y_true[:, None] < y_true[None, :]  # 取出负例-正例的差值
        y_true = y_true.float()
        y_pred = y_pred - (1 - y_true) * 1e12
        # 变成只有一个维度的, y_pred shape: ( (batch_size // 2) ** 2, )
        y_pred = y_pred.view(-1)
        # 这里加0是因为e^0 = 1相当于在log中加了1
        # y_pred shape: ( (batch_size // 2) ** 2 + 1, )
        y_pred = torch.cat((torch.tensor([0]).float().to(self.device), y_pred), dim=0)
        # 返回一个标量, 返回给定维度dim中输入张量的每一行的总指数对数
        return torch.logsumexp(y_pred, dim=0)

    def train(
            self,
            train_dataset: Dataset,
            output_dir: str,
            eval_dataset: Dataset = None,
            verbose: bool = True,
            batch_size: int = 8,
            num_epochs: int = 1,
            weight_decay: float = 0.01,
            seed: int = 42,
            warmup_ratio: float = 0.1,
            lr: float = 2e-5,
            eps: float = 1e-6,
            gradient_accumulation_steps: int = 1,
            max_grad_norm: float = 1.0,
            max_steps: int = -1
    ):
        """
        真正的训练过程
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """
        os.makedirs(output_dir, exist_ok=True)
        logger.debug("Use device: {}".format(self.device))
        self.bert.to(self.device)
        set_seed(seed)

        # 为什么没有 shuffle=True
        train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
        total_steps = len(train_dataloader) * num_epochs
        # 参数分组
        param_optimizer = list(self.bert.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        # 预热步数, 优化器, 学习率调度器 初始化
        warmup_steps = math.ceil(total_steps * warmup_ratio)  # by default 10% of _train data for warm-up
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
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

        # 如果目录存在
        if self.model_name_or_path and os.path.exists(self.model_name_or_path):
            try:
                # set global_step to global_step of last saved checkpoint from model path
                checkpoint_suffix = self.model_name_or_path.split("/")[-1].split("-")
                # 就是取第二位的名字, 是训练步数
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                global_step = int(checkpoint_suffix)
                # 总步数 // (训练集大小 // 梯度累积步数)
                epochs_trained = global_step // (len(train_dataloader) // gradient_accumulation_steps)
                # 当前 epoch 训练的步数
                steps_trained_in_current_epoch = global_step % (len(train_dataloader) // gradient_accumulation_steps)
                logger.info("   Continuing training from checkpoint, will skip to saved global_step")
                logger.info("   Continuing training from epoch %d" % epochs_trained)
                logger.info("   Continuing training from global step %d" % global_step)
                logger.info("   Will skip the first %d steps in the current epoch" % steps_trained_in_current_epoch)
            except ValueError:
                logger.info("   Starting fine-tuning.")

        # 保存训练过程中的指标
        training_progress_scores = {
            "global_step": [],
            "train_loss": [],
            "eval_spearman": [],
            "eval_pearson": [],
        }
        for current_epoch in trange(int(num_epochs), desc="Epoch", disable=False, mininterval=0):
            self.bert.train()
            current_loss = 0
            # 跳过已经训练过的 epoch
            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            batch_iterator = tqdm(train_dataloader,
                                  desc=f"Running Epoch {epoch_number + 1} of {num_epochs}",
                                  disable=False,
                                  mininterval=0)
            for step, batch in enumerate(batch_iterator):
                # 跳过已经训练过的步数
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                # 的确只有一个输入
                inputs, labels = batch
                labels = labels.to(self.device)
                # inputs        [batch, 1, seq_len] -> [batch, seq_len]
                input_ids = inputs.get('input_ids').squeeze(1).to(self.device)
                attention_mask = inputs.get('attention_mask').squeeze(1).to(self.device)
                token_type_ids = inputs.get('token_type_ids').squeeze(1).to(self.device)
                output_embeddings = self.get_sentence_embeddings(input_ids, attention_mask, token_type_ids)
                # 计算损失
                # labels: (batch_size,)
                # output_embeddings: (batch_size, hidden_size)
                # TODO: 我没看懂这个维度是怎么缩减的
                loss = self.calc_loss(labels, output_embeddings)
                current_loss = loss.item()
                if verbose:
                    batch_iterator.set_description(
                        f"Epoch: {epoch_number + 1}/{num_epochs}, Batch:{step}/{len(train_dataloader)}, Loss: {current_loss:9.4f}")

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
            # 有格式的, 第二个参数是当前总步数, 第四个是当前 epoch
            output_dir_current = os.path.join(output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number))
            results = self.eval_model(eval_dataset, output_dir_current, verbose=verbose, batch_size=batch_size)
            self.save_model(output_dir_current, model=self.bert, results=results)
            training_progress_scores["global_step"].append(global_step)
            training_progress_scores["train_loss"].append(current_loss)
            for key in results:
                training_progress_scores[key].append(results[key])
            # 保存成 csv
            report = pd.DataFrame(training_progress_scores)
            report.to_csv(os.path.join(output_dir, "training_progress_scores.csv"), index=False)

            # 当前最佳是根据 eval_spearman 来评价的
            eval_spearman = results["eval_spearman"]
            if eval_spearman > best_eval_metric:
                best_eval_metric = eval_spearman
                logger.info(f"Save new best model, best_eval_metric: {best_eval_metric}")
                self.save_model(output_dir, model=self.bert, results=results)

            # 如果达到最大步数, 就停止训练. 即当 global_step > max_steps 时, 就停止训练
            if 0 < max_steps < global_step:
                return global_step, training_progress_scores

        return global_step, training_progress_scores
