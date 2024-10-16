import torch
from tqdm import tqdm  # 引入 tqdm 进度条模块
import random  # 用于随机生成索引

from dataloader import data, entity_dic, entity_dic_size, relation_dic_size, relation_dic
from module.KgEmbedding import get_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('D:/PyStudy/Knowledge_Embedding/save_weight/transH_2048_1.0_80.pkl').to(device)


def evaluate_mean_rank(model, test_data, num_entities, device, sample_size=None):
    model.eval()  # 进入评估模式
    total_rank = 0
    total_samples = len(test_data)

    # 如果指定了 sample_size，进行随机索引采样，否则评估全部数据
    if sample_size is not None:
        sampled_indices = random.sample(range(total_samples), sample_size)
    else:
        sampled_indices = range(total_samples)

    with torch.no_grad():  # 评估时不需要计算梯度
        for idx in tqdm(sampled_indices, desc="Evaluating", ncols=100):
            h, r, t = test_data[idx]  # 通过索引获取对应的三元组
            # 计算所有候选实体的得分
            scores = []
            i = 0
            for candidate_t in range(num_entities):
                if i ==0:
                    candidate_t = t
                    data = torch.tensor([[h, r, candidate_t]]).to(device)
                    i += 1
                elif candidate_t != t :
                    data = torch.tensor([[h, r, candidate_t]]).to(device)
                else:
                    continue

                score, _, _ = model(data,data)  # 计算评分
                scores.append(score.item())

            # 对所有得分进行排序，得分越小越好
            sorted_scores = torch.argsort(torch.tensor(scores).to(device))

            # 找到正确实体 t 的排名
            rank = (sorted_scores == 0).nonzero(as_tuple=True)[0].item() + 1
            print('')
            print(rank)
            total_rank += rank

    mean_rank = total_rank / len(sampled_indices)
    return mean_rank

# 示例：假设你有一个训练好的模型和测试数据
test_data = data  # 你的测试数据，保持原始格式
num_entities = entity_dic_size  # 实体的数量
sample_size = 5  # 随机采样 5 个三元组进行评估

# 评估 Mean Rank
mean_rank_value = evaluate_mean_rank(model, test_data, num_entities, device, sample_size=sample_size)
print(f"Mean Rank: {mean_rank_value}")
