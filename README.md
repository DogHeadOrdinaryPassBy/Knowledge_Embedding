# **Knowledge-Embedding概述**
## 项目名称：知识图谱嵌入模型的训练与评估

本项目实现了知识图谱嵌入模型（TransE、TransH、TransR）的训练、负样本构建、模型评估等模块。通过对实体和关系进行嵌入训练，模型学习到知识图谱中的结构信息，并通过测试数据评估模型的性能。该项目主要包含以下五个部分：

-嵌入模型定义  
-数据加载与负样本构建  
-模型训练及可视化  
-模型评估（Mean Rank 以及 hits@10）  
-实体预测和关系预测  

## 环境依赖
环境依赖
该项目基于 Python 和 PyTorch，实现了知识图谱嵌入模型的训练,训练效果可视化以及评估。运行本项目需要以下依赖库：

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
├── eval.py                &nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;      _实体和关系预测脚本_  
├── xxx_loss.png                &nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;      _损失图像_  
└── README.md             &nbsp;&nbsp;&nbsp;   &nbsp;&nbsp;&nbsp;       _项目说明文档_  

## 数据说明
dataset/subgraph_kgp1.txt 包含了知识图谱数据，格式为多个字段，每一行表示一个三元组（头实体、关系、尾实体），并附加了一些辅助信息。项目中只处理了中文语言的三元组。

## 使用说明
### 1. 数据加载与负样本生成
在 `dataloader.py` 中，实现了加载知识图谱数据集并构建负样本的逻辑。具体包括以下部分：

``get_data(path)``: 从给定的路径加载知识图谱数据，并生成实体和关系的字典映射。  

``build_neg(batch, data, entity_dic_size)``: 根据正样本随机替换头或尾实体，生成负样本。  

使用 `torch.utils.data.Dataset` 构建了一个可迭代的数据集，并通过 DataLoader 加载数据。

### 2. 知识图谱嵌入模型
在 `module/KgEmbedding.py` 中定义了嵌入模型 `DistanceModel`。支持 TransE、TransH、TransR 模型结构。模型实现了以下功能：  

``forward()``: 根据正负样本计算模型分数。  

``get_loss()``: 基于正负样本的分数计算损失函数，采用margin-based ranking loss。  

你可以根据需要选择不同的模型（如 TransE、TransH、TransR）。

### 3. 模型训练
`train.py` 包含了模型的训练逻辑：

-通过 `torch.optim.Adam` 优化器进行参数更新。  
-使用 `tqdm` 显示训练进度，记录每个 `epoch` 的损失值，并将结果保存为模型文件。  

训练完成后，会保存以下文件：

-训练好的模型参数  
-实体与关系的嵌入  
-损失曲线图  

### 4. 模型评估
`test.py` 实现了模型的评估逻辑。该部分主要计算模型的**MEAN RANK** 和 **HITS@10**：

-随机选择测试样本，计算每个测试样本的头实体评分和所有候选尾实体的评分。
-对评分进行排序，并计算测试样本中原始尾实体的排名，最终输出平均排名结果。  

### 5. 实体预测和关系预测
`eval.py` 实现了模型的实体预测和关系预测。

-通过指定路径加载训练完毕的模型和测试数据
-针对尾实体预测和关系预测进行评分，分别对每个测试样本计算所有候选实体和候选关系的得分。得分越小表示预测结果越符合期望。
-排序得分后，选取排名靠前的结果。
-将预测的实体和关系分别保存为 `entity_output` 和 `relation_output`。
-将预测结果覆盖到原始 JSON 文件中的 output 字段，并将更新后的文件保存为 `output_{path}.json`。

## 运行步骤
**下载依赖**   
在运行代码之前，请确保你已经安装了项目所需的依赖库。可以使用如下命令安装依赖：

`pip install torch tqdm matplotlib`  

**准备数据集**   
将 subgraph_kgp1.txt 数据集文件放置到 dataset/ 文件夹下，确保数据集路径正确。

**运行模型训练**  
在项目根目录下，运行 `train.py` 来训练模型：  

`python train.py`  

该脚本会输出训练过程中的损失，并将训练好的模型与嵌入参数保存到 save_weight/ 文件夹，将训练过程的损失函数保存到根目录。

**模型评估**   
在模型训练完成后，运行 `test.py` 进行评估：  

`python test.py`  

该脚本会输出评估过程中模型的**MEAN RANK** 和 **HITS@10**。

**实体预测和关系预测**    
同样是在模型训练完成后，运行 `eval.py`：  

`python eval.py`

该脚本会输出给定输入实体和关系的前五个最相关的预测结果并将其保存至指定路径
## 参考文献
若使用本项目中的代码或方法，请引用相关论文：

TransE: Bordes A., Usunier N., Garcia-Duran A., Weston J., Yakhnenko O. Translating Embeddings for Modeling Multi-relational Data. NeurIPS 2013.
TransH:
TransR: