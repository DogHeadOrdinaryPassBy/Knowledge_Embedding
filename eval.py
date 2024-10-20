import json
import os

import torch
from tqdm import tqdm, trange  # 引入 tqdm 进度条模块
from dataloader import entity_dic_size, entity_dic, relation_dic, relation_dic_size

path = './save_weight/transE_2048_3.5_200.pkl'
data_path = './dataset/subgraph_kgp1_valid.json'

with open(data_path, 'r', encoding='utf-8') as f:
    eval_data = json.load(f)

link_data = eval_data['link_prediction']
predict_link_data =[]
for i in link_data:
    predict_link_data.append([entity_dic[i['input'][0]], 0, entity_dic[i['input'][1]]])  # 0只做占位使用

tail_data = eval_data['entity_prediction']
predict_tail_data =[]
for i in tail_data:
    predict_tail_data.append([entity_dic[i['input'][0]], relation_dic[i['input'][1]], 0])  # 0只做占位使用

print(predict_link_data)
print('')
print(predict_tail_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(path).to(device)


def evaluate_mean_rank(model, test_data, num_entities, num_relations, device, function='tail_prediction'):
    print(f'model:{path}')
    model.eval()

    with torch.no_grad():
        test_data = torch.tensor(test_data ).to(device)

        # 计算所有候选关系/尾实体的评分
        if function == 'tail_prediction':
            # 预分配评分矩阵，行数为test_size，列数为 num_entities
            scores = torch.empty((len(predict_link_data), num_entities), device=device)
            for candidate_t in trange(num_entities, desc="Evaluating", ncols=100):
                test_data[:, -1] = candidate_t  # 替换尾实体为候选实体
                score, _, _ = model(test_data, test_data)  # 计算评分
                scores[:, candidate_t] = score  # 存入第 candidate_t 列

        if function == 'link_prediction':
            # 预分配评分矩阵，行数为test_size，列数为 num_relations
            scores = torch.empty((len(predict_link_data), num_relations), device=device)
            for candidate_r in trange(num_relations, desc="Evaluating", ncols=100):
                test_data[:, -2] = candidate_r  # 替换关系为候选关系
                score, _, _ = model(test_data, test_data)  # 计算评分
                scores[:, candidate_r] = score  # 存入第 candidate_r 列
        # 对所有得分进行排序，得分越小越好
        sorted_scores = torch.argsort(scores, dim=1)
        if function == 'tail_prediction':
            sorted_scores = sorted_scores[:, 1:6]  # 第一个是自己
        else:
            sorted_scores = sorted_scores[:, :5]

    return sorted_scores


test_data = predict_tail_data
num_entities = entity_dic_size
num_relations = relation_dic_size

index_to_entity = {v: k for k, v in entity_dic.items()}
index_to_relation = {v: k for k, v in relation_dic.items()}

sorted_entity = evaluate_mean_rank(model, test_data, num_entities, num_relations, device)
sorted_relation = evaluate_mean_rank(model, test_data, num_entities, num_relations, device, function='link_prediction')
entity_output=[]
relation_output=[]
sorted_entity = sorted_entity.tolist()
sorted_relation = sorted_relation.tolist()
print(sorted)
for row in sorted_entity:
    tem = []
    for j in row:
        tem.append(index_to_entity[j])
    entity_output.append(tem)

for row in sorted_relation:
    tem = []
    for j in row:
        tem.append(index_to_relation[j])
    relation_output.append(tem)


# 读取现有的 JSON 文件
with open(data_path, 'r', encoding='utf-8') as f:
    existing_data = json.load(f)

# 覆盖 output 字段
for i, item in enumerate(entity_output):
    existing_data["entity_prediction"][i]["output"] = item

for i, item in enumerate(relation_output):
    existing_data["link_prediction"][i]["output"] = item

# 将修改后的数据写回 JSON 文件
with open(f'output_{path[14:]}.json', 'w', encoding='utf-8') as f:
    json.dump(existing_data, f, ensure_ascii=False, indent=4)

print("output 字段已成功覆盖")


