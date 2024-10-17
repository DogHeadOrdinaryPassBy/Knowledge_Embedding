import torch
from tqdm import tqdm, trange  # 引入 tqdm 进度条模块
import random  # 用于随机生成索引

from dataloader import data, entity_dic, entity_dic_size, relation_dic_size, relation_dic
from module.KgEmbedding import get_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('D:/PyStudy/Knowledge_Embedding/save_weight/transR_2048_1.5_80.pkl').to(device)
test_size = 2500
num_test = 1


def evaluate_mean_rank(model, test_data, num_entities, device):
    model.eval()  # 进入评估模式
    total_rank = 0

    with torch.no_grad():  # 评估时不需要计算梯度
        for idx in trange(num_test, desc="Evaluating_all", ncols=100):
            eval_data = []
            for i in range(test_size):
                data = list(random.choice(test_data))  # 随机选择测试数据
                eval_data.append(data)

            eval_data = torch.tensor(eval_data).to(device)

            # 预分配评分矩阵，行数为test_size，列数为 num_entities + 1
            scores = torch.empty((test_size, num_entities + 1), device=device)

            # 计算头实体评分 (原始得分)
            score, _, _ = model(eval_data, eval_data)  # 计算评分
            scores[:, 0] = score  # 将原始的得分放入第 0 列

            # 计算所有候选尾实体的评分
            for candidate_t in trange(num_entities, desc="Evaluating", ncols=100):
                eval_data[:, -1] = candidate_t  # 替换尾实体为候选实体
                score, _, _ = model(eval_data, eval_data)  # 计算评分
                scores[:, candidate_t + 1] = score  # 存入第 candidate_t 列

            # 对所有得分进行排序，得分越小越好
            sorted_scores = torch.argsort(scores, dim=1)

            # 计算每一行中原始尾实体的排名
            index_sum = 0
            for row in sorted_scores:
                zero_indices = (row == 0).nonzero(as_tuple=True)[0] + 1  # 找到原始得分的排名
                index_sum += zero_indices.item()

            total_rank += index_sum

    mean_rank = total_rank / (test_size * num_test)
    return mean_rank


# 示例：假设你有一个训练好的模型和测试数据
test_data = data  # 你的测试数据，保持原始格式
num_entities = entity_dic_size  # 实体的数量

# 评估 Mean Rank
mean_rank_value = evaluate_mean_rank(model, test_data, num_entities, device)
print(f"Mean Rank: {mean_rank_value}")
