# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Base sentence model function, add encode function.
Parts of this file is adapted from the sentence-transformers library at https://github.com/UKPLab/sentence-transformers.
"""
import os
from enum import Enum
from typing import List, Union, Optional
from tqdm.autonotebook import trange
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm, trange
from text2vec.utils.stats_util import compute_spearmanr, compute_pearsonr

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"


class EncoderType(Enum):
    """
    编码器类型
    """
    FIRST_LAST_AVG = 0
    LAST_AVG = 1
    CLS = 2
    POOLER = 3
    MEAN = 4

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return EncoderType[s]
        except KeyError:
            raise ValueError()


class SentenceModel:
    def __init__(
            self,
            model_name_or_path: str = "shibing624/text2vec-base-chinese",
            encoder_type: Union[str, EncoderType] = "MEAN",
            max_seq_length: int = 128,
            device: Optional[str] = None,
    ):
        """
        Initializes the base sentence model.

        :param model_name_or_path: The name of the model to load from the huggingface models library.
        :param encoder_type: The type of encoder to use, See the EncoderType enum for options:
            FIRST_LAST_AVG, LAST_AVG, CLS, POOLER(cls + dense), MEAN(mean of last_hidden_state)
        :param max_seq_length: The maximum sequence length.
        :param device: Device (like 'cuda' / 'cpu') that should be used for computation. If None, checks if GPU.

        bert model: https://huggingface.co/transformers/model_doc/bert.html?highlight=bert#transformers.BertModel.forward
        BERT return: <last_hidden_state>, <pooler_output> [hidden_states, attentions]
        Note that: in doc, it says <last_hidden_state> is better semantic summery than <pooler_output>.
        thus, we use <last_hidden_state>.
        """
        self.model_name_or_path = model_name_or_path
        # 有多种编码器可选
        encoder_type = EncoderType.from_string(encoder_type) if isinstance(encoder_type, str) else encoder_type
        if encoder_type not in list(EncoderType):
            raise ValueError(f"encoder_type must be in {list(EncoderType)}")
        self.encoder_type = encoder_type
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        logger.debug("Use device: {}".format(self.device))
        self.bert.to(self.device)
        self.results = {}  # Save training process evaluation result

    def __str__(self):
        return f"<SentenceModel: {self.model_name_or_path}, encoder_type: {self.encoder_type}, " \
               f"max_seq_length: {self.max_seq_length}>"

    def get_sentence_embeddings(self, input_ids, attention_mask, token_type_ids):
        """
        获取模型输出, 基于不同的编码器类型, 返回的 shape 都是 (batch_size, hidden_size)
        Returns the model output by encoder_type as embeddings.

        Utility function for self.bert() method.
        """
        model_output = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.encoder_type == EncoderType.FIRST_LAST_AVG:
            # Get the first and last hidden states, and average them to get the embeddings
            # hidden_states have 13 list, second is hidden_state
            # shape: (batch_size, sequence_length, hidden_size)
            first = model_output.hidden_states[1]
            last = model_output.hidden_states[-1]
            seq_length = first.size(1)  # Sequence length

            # shape change list:
            # first.transpose(1, 2) -> (batch_size, hidden_size, sequence_length)
            # torch.avg_pool1d -> (batch_size, hidden_size, 1)
            # 1 = (sequence_length + 2 * padding - kernel_size) / stride + 1 = 0 / kernel_size + 1
            # squeeze(-1) -> (batch_size, hidden_size)
            first_avg = torch.avg_pool1d(first.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # [batch, hid_size]
            last_avg = torch.avg_pool1d(last.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # [batch, hid_size]
            # shape change list:
            # first_avg.unsqueeze(1) -> (batch_size, 1, hidden_size)
            # torch.cat -> (batch_size, 2, hidden_size)
            # transpose(1, 2) -> (batch_size, hidden_size, 2)
            # torch.avg_pool1d -> (batch_size, hidden_size, 1)
            # squeeze(-1) -> (batch_size, hidden_size) 
            final_encoding = torch.avg_pool1d(
                torch.cat([first_avg.unsqueeze(1), last_avg.unsqueeze(1)], dim=1).transpose(1, 2),
                kernel_size=2).squeeze(-1)
            # final_encoding shape: (batch_size, hidden_size)
            return final_encoding

        if self.encoder_type == EncoderType.LAST_AVG:
            sequence_output = model_output.last_hidden_state  # [batch_size, max_len, hidden_size]
            seq_length = sequence_output.size(1)
            final_encoding = torch.avg_pool1d(sequence_output.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
            # final_encoding shape: (batch_size, hidden_size)
            return final_encoding

        if self.encoder_type == EncoderType.CLS:
            sequence_output = model_output.last_hidden_state
            # 这个就是取第一个维度的第一个元素, 也就是取第一个token的向量
            return sequence_output[:, 0]  # [batch, hid_size]

        if self.encoder_type == EncoderType.POOLER:
            return model_output.pooler_output  # [batch, hid_size]

        if self.encoder_type == EncoderType.MEAN:
            """
            Mean Pooling - Take attention mask into account for correct averaging
            """
            token_embeddings = model_output.last_hidden_state  # Contains all token embeddings
            # shape change list:
            # attention_mask.unsqueeze(-1) -> (batch_size, sequence_length, 1)
            # expand -> (batch_size, sequence_length, hidden_size)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            # shape change list:
            # token_embeddings * input_mask_expanded -> (batch_size, sequence_length, hidden_size)
            # sum -> (batch_size, hidden_size)
            # input_mask_expanded.sum(1) -> (batch_size, hidden_size)
            # torch.clamp -> (batch_size, hidden_size) # 保证最小值为 1e-9
            # / -> (batch_size, hidden_size)
            final_encoding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9)
            return final_encoding  # [batch, hid_size]

    def encode(
            self,
            sentences: Union[str, List[str]],
            batch_size: int = 64,
            show_progress_bar: bool = False,
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            device: str = None,
    ):
        """
        主函数, 基于pretrained model计算文本向量
        Returns the embeddings for a batch of sentences.

        :param sentences: str/list, Input sentences
        :param batch_size: int, Batch size
        :param show_progress_bar: bool, Whether to show a progress bar for the sentences
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        """
        self.bert.eval()
        if device is None:
            device = self.device
        # 优先级更高
        if convert_to_tensor:
            convert_to_numpy = False
        # 转换文本输入, 统一为 list
        input_is_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]
            input_is_string = True

        all_embeddings = []
        # 将句子按长度降序, 返回的是索引位置
        length_sorted_idx = np.argsort([-len(s) for s in sentences])
        # 重新排序后的句子, 长度降序
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        # 按批次处理
        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            # 当前批次的句子
            sentences_batch = sentences_sorted[start_index: start_index + batch_size]
            # 计算句子向量
            # Compute sentences embeddings
            with torch.no_grad():
                embeddings = self.get_sentence_embeddings(
                    # 填充到当前批次最大长度
                    **self.tokenizer(sentences_batch, max_length=self.max_seq_length,
                                     padding=True, truncation=True, return_tensors='pt').to(device)
                )
            embeddings = embeddings.detach()
            if convert_to_numpy:
                embeddings = embeddings.cpu()
            all_embeddings.extend(embeddings)
        # 按照原始顺序返回
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        # 如果输入是字符串, 则返回第一个向量, 让结构保持一致
        if input_is_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def eval_model(self, eval_dataset: Dataset, output_dir: str = None, verbose: bool = True, batch_size: int = 16):
        """
        Evaluates the model on eval_df. Saves results to args.output_dir
            result: Dictionary containing evaluation results.
        """
        result = self.evaluate(eval_dataset, output_dir, batch_size=batch_size)
        self.results.update(result)

        if verbose:
            logger.info(self.results)

        return result

    def evaluate(self, eval_dataset, output_dir: str = None, batch_size: int = 16):
        """
        Evaluates the model on eval_dataset.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """
        results = {}

        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
        self.bert.to(self.device)
        self.bert.eval()

        batch_labels = []
        batch_preds = []
        for batch in tqdm(eval_dataloader, disable=False, desc="Running Evaluation"):
            source, target, labels = batch
            labels = labels.to(self.device)
            batch_labels.extend(labels.cpu().numpy())
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(self.device)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(self.device)
            source_token_type_ids = source.get('token_type_ids').squeeze(1).to(self.device)

            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(self.device)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(self.device)
            target_token_type_ids = target.get('token_type_ids').squeeze(1).to(self.device)

            with torch.no_grad():
                source_embeddings = self.get_sentence_embeddings(source_input_ids, source_attention_mask,
                                                                 source_token_type_ids)
                target_embeddings = self.get_sentence_embeddings(target_input_ids, target_attention_mask,
                                                                 target_token_type_ids)
                preds = torch.cosine_similarity(source_embeddings, target_embeddings)
            batch_preds.extend(preds.cpu().numpy())

        spearman = compute_spearmanr(batch_labels, batch_preds)
        pearson = compute_pearsonr(batch_labels, batch_preds)
        logger.debug(f"labels: {batch_labels[:10]}")
        logger.debug(f"preds:  {batch_preds[:10]}")
        logger.debug(f"pearson: {pearson}, spearman: {spearman}")

        results["eval_spearman"] = spearman
        results["eval_pearson"] = pearson
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "eval_results.txt"), "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

        return results

    def save_model(self, output_dir, model, results=None):
        """
        Saves the model to output_dir.
        :param output_dir:
        :param model:
        :param results:
        :return:
        """
        logger.info(f"Saving model checkpoint to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))
