import torch
from tqdm import tqdm, trange
import random
from dataloader import data, entity_dic_size

path = './save_weight/transH_2048_1_200.pkl'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(path).to(device)
test_size = 1000
num_test = 1


def evaluate_mean_rank(model, test_data, num_entities, device):
    print(f'model:{path}')
    model.eval()
    total_rank = 0
    total_hit = 0
    total_MRR = 0
    with torch.no_grad():
        for j in trange(num_test, desc="Evaluating_all", ncols=100):
            eval_data = []
            for i in range(test_size):
                data = list(random.choice(test_data))
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
            hit = 0
            mrr = 0.0
            for row in sorted_scores:
                zero_indices = (row == 0).nonzero(as_tuple=True)[0] + 1  # 找到原始得分的排名
                index_sum += zero_indices.item()
                mrr += 1 / zero_indices.item()
                if zero_indices.item() <= 10:
                    hit += 1
            total_hit += hit
            total_rank += index_sum
            total_MRR += mrr


    mean_rank = total_rank / (test_size * num_test)
    hit10 = total_hit / (test_size * num_test)
    mrr = total_MRR / (test_size * num_test)
    return mean_rank, hit10, mrr



test_data = data
num_entities = entity_dic_size


mean_rank, hit10, MRR = evaluate_mean_rank(model, test_data, num_entities, device)
print(f"Mean Rank: {mean_rank},hit@10:{hit10},MRR:{MRR:.4f}")