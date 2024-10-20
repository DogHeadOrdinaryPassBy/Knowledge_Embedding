# **Knowledge-Embedding概述**
## 项目名称：知识图谱嵌入模型的训练与评估

本项目实现了知识图谱嵌入模型（TransE、TransH、TransR）的训练、负样本构建、模型评估等模块。通过对实体和关系进行嵌入训练，模型学习到知识图谱中的结构信息，并通过测试数据评估模型的性能。该项目主要包含以下四个部分：

-嵌入模型定义
-数据加载与负样本构建
-模型训练及可视化
-模型评估（Mean Rank 以及 hits@10）

## 环境依赖
环境依赖
该项目基于 Python 和 PyTorch，实现了知识图谱嵌入模型的训练与评估。运行本项目需要以下依赖库：

Python 3.x  
torch==1.x  
tqdm==4.x  
matplotlib==3.x  
csv

## 文件结构

├── dataset/                 &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;      _数据集文件夹_  
│   └── subgraph_kgp1.txt        &nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;  _知识图谱数据集_  
├── module/  
│   └── KgEmbedding.py       &nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;      _嵌入模型的定义与实现_  
├── save_weight/    
│   └──xxx.pkl               &nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;      _训练完毕的模型_      
│   └──xxx.pt              &nbsp;&nbsp;&nbsp;   &nbsp;&nbsp;&nbsp;       _训练完毕的权重文件_  
├── dataloader.py           &nbsp;&nbsp;&nbsp;   &nbsp;&nbsp;&nbsp;      _数据加载与负样本构建_  
├── train.py                &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;       _训练模型脚本_  
├── test.py                &nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;      _模型评估脚本_  
└── README.md             &nbsp;&nbsp;&nbsp;   &nbsp;&nbsp;&nbsp;       _项目说明文档_  

## 数据说明
dataset/subgraph_kgp1.txt 包含了知识图谱数据，格式为多个字段，每一行表示一个三元组（头实体、关系、尾实体），并附加了一些辅助信息。项目中只处理了中文语言的三元组。

# 运行步骤