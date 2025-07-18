UniG-Encoder 是一种通用的图数据编码器，主要用于将图结构数据（如社交网络、分子结构、知识图谱等）转换为低维向量表示（即图嵌入，Graph Embedding），并支持多种下游任务（如节点分类、链接预测、图分类）。
特点：通过模块化设计兼容多种图类型（如静态图、动态图、属性图）和任务需求，同时捕捉局部节点特征和全局图结构信息

基础结构：
-----------输入层----------
节点特征、边特征、图结构
----------编码层----------
图神经网络层（GNN）：使用GCN、GAT、GraphSAGE等聚合邻居信息（如通过GAT（图注意力网络）加权聚合重要邻居的特征）
全局池化
层次化编码
----------输出层----------
节点级输出/图级输出



import torch
from torch_geometric.nn import GATConv, global_mean_pool
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


# 4个用户，每个用户64维特征
x = torch.randn(4, 64)

# 边连接：COO格式 [[源节点], [目标节点]]
edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

batch = torch.tensor([0, 0, 1, 1])


class UniGEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim)  # 图注意力层
        self.conv2 = GATConv(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch=None):
        # 节点级编码
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)

        # 图级编码（如有batch信息,即有多个子图）
        if batch is not None:
            x = global_mean_pool(x, batch)  # 输出全图嵌入
        return x


# 使用示例
model = UniGEncoder(in_dim=64, hidden_dim=128, out_dim=256)
node_embeddings = model(x, edge_index) 
graph_embedding = model(x, edge_index, batch)
print(node_embeddings)
print(graph_embedding)

1.node_embeddings 会输出4个节点，每个节点256维向量 输出大部分值在[-1,1]符合激活函数特性。
梯度信息：grad_fn=<AddBackward0> 表示该张量由加法操作生成，支持反向传播
2.graph_embeddings 输出两张图，每张图256维向量 一张图的256维全局表示，由该图内所有节点嵌入聚合（如均值池化）生成。
梯度信息：grad_fn=<DivBackward0> 表示通过除法操作（如均值计算）生成


print("节点嵌入均值:", node_embeddings.mean(dim=0))  # 各维度均值
print("图嵌入方差:", graph_embedding.var(dim=0))     # 各维度方差


sim = cosine_similarity(node_embeddings[0], node_embeddings[1], dim=0)  # 用余弦相似度，比较节点0和1
print(sim)

# plt.subplot(2,1,2)
# plt.bar(range(256), node_mean.abs().numpy())
# plt.xlabel('Dimension')
# plt.ylabel('Mean Absolute Value')
# plt.title('Node Embedding Feature Importance')
# plt.show()

# PCA降维
# pca = PCA(n_components=2)
# emb_2d = pca.fit_transform(node_embeddings.detach().numpy())
# plt.subplot(2,1,1)
# plt.scatter(emb_2d[:, 0], emb_2d[:, 1], label=['Node 0', 'Node 1', 'Node 2', 'Node 3'])
# plt.legend()
# plt.show()


node_embeddings_np = node_embeddings.detach().numpy() # [4,256]

pca = PCA(n_components=2) # PCA降维
emb_2d = pca.fit_transform(node_embeddings_np)

plt.figure(figsize=(8, 6))

colors = ['red', 'blue', 'green', 'purple']
markers = ['o', 's', '^', 'D']  # 圆形、方形、三角形、菱形
labels = ['Node 0', 'Node 1', 'Node 2', 'Node 3']

for i in range(4):
    plt.scatter(emb_2d[i, 0], emb_2d[i, 1],
                c=colors[i],
                marker=markers[i],
                s=100,  # 点大小
                edgecolors='black',  # 边缘颜色
                linewidths=1,  # 边缘线宽
                label=labels[i])

plt.title('2D PCA Projection of Node Embeddings', fontsize=14)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)

# 显示方差解释率
explained_var = pca.explained_variance_ratio_
plt.text(0.05, 0.95,
         f'Explained Variance: PC1={explained_var[0]:.1%}, PC2={explained_var[1]:.1%}',
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8))

plt.legend(fontsize=10, loc='best')
plt.grid(True, linestyle='--', alpha=0.5)

for i in range(4):
    plt.text(emb_2d[i, 0]+0.02,  # x坐标微调
             emb_2d[i, 1]+0.02,  # y坐标微调
             f'Node {i}',
             fontsize=10,
             bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.show()

这里观察输出的图表，可以看出某些集聚的节点特征相似/某些孤立的节点特征独特
横轴PC1表现最大方差方向，解释数据差异的主要维度；纵轴PC2表现次大方差方向，与PC1正交的次要差异维度



# 节点到边的投影

import torch.nn as nn


class UniGEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        # 节点编码层
        self.node_encoder = GATConv(node_dim, node_dim)

        # 边投影层
        self.edge_proj = nn.Sequential(
            nn.Linear(2 * node_dim, edge_dim),
            nn.ReLU()
        )

    def forward(self, x, edge_index):
        # 1. 节点编码
        h_nodes = self.node_encoder(x, edge_index)  # [num_nodes, node_dim]

        # 2. 提取边对应的节点对
        src, dst = edge_index
        h_src = h_nodes[src]  # [num_edges, node_dim]
        h_dst = h_nodes[dst]  # [num_edges, node_dim]

        # 3. 生成边特征
        h_edges = self.edge_proj(torch.cat([h_src, h_dst], dim=-1))  # [num_edges, edge_dim]

        return h_nodes, h_edges
