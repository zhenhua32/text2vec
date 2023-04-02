import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义一个类，继承自nn.Module
class CosentLoss(nn.Module):
    """
    谢谢你提供的信息。根据我的搜索结果，这个损失函数是一种基于余弦相似度的损失函数，用于自监督学习的任务，它的目的是让正样本对之间的余弦相似度大于负样本对之间的余弦相似度。它的变量和参数分别是：

    u_i, u_j, u_k, u_l 是四个向量，分别表示两个正样本对和两个负样本对。
    \Omega_{pos} 是正样本对的集合，\Omega_{neg} 是负样本对的集合。
    \lambda 是一个超参数，控制损失函数的敏感度。
    \cos 是余弦相似度函数。

    u_i 的 shape 应该是一个一维的张量，表示一个向量。它的长度取决于你的模型的输出维度，通常是一个固定的正整数。例如，如果你的模型输出是一个 128 维的向量，那么 u_i 的 shape 就应该是 [128]。
    """

    # 初始化方法，接收一个超参数lambda
    def __init__(self, lambda_):
        # 调用父类的初始化方法
        super(CosentLoss, self).__init__()
        # 将lambda作为一个属性保存
        self.lambda_ = lambda_

    # 前向传播方法，接收四个向量作为输入
    def forward(self, u_i, u_j, u_k, u_l):
        # 计算正样本对之间的余弦相似度
        cos_pos = F.cosine_similarity(u_i, u_j)
        # 计算负样本对之间的余弦相似度
        cos_neg = F.cosine_similarity(u_k, u_l)
        # 计算损失函数的值，使用torch.log1p和torch.exp避免数值溢出
        loss = torch.log1p(torch.exp(self.lambda_ * (cos_neg - cos_pos)))
        # 返回损失函数的值
        return loss
