import csv
import multiprocessing
import random
import torch
from torch.utils.data import DataLoader
import ipdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 2048
file_path = './dataset/subgraph_kgp1.txt'


def get_data(path):
    data = []
    entity_dic = {}
    relation_dic = {}
    entity_counter = 0
    relation_counter = 0
    with open(path, 'r', encoding='utf-8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        for row in reader:
            _, _, lan, _, weight, head, relation, tail, _, _, _, _ = row
            if lan == 'zh':
                item = {'head': head, 'relation': relation, 'tail': tail, 'weight': weight}

                # 处理实体词典
                if head not in entity_dic:
                    entity_dic[head] = entity_counter
                    entity_counter += 1
                if tail not in entity_dic:
                    entity_dic[tail] = entity_counter
                    entity_counter += 1
                # 处理关系词典
                if relation not in relation_dic:
                    relation_dic[relation] = relation_counter
                    relation_counter += 1

                data.append((entity_dic[head], relation_dic[relation], entity_dic[tail]))

    return data, entity_dic, relation_dic


def build_neg(batch, data, entity_dic_size):
    entity_set = range(0, entity_dic_size)
    neg_data = []
    data = set(data)
    for item in batch:
        valid_negative = False
        head = item[0]
        relation = item[1]
        tail = item[2]
        while not valid_negative:
            if random.random() < 0.3:
                # 替换头实体,尾实体保持不变
                neg_head = random.choice(entity_set)
                neg_tail = tail
                new_sample = (neg_head, relation, neg_tail)
            else:
                # 替换尾实体,头实体保持不变
                neg_head = head
                neg_tail = random.choice(entity_set)
                new_sample = (neg_head, relation, neg_tail)

            # 检查新生成的三元组是否在全局正样本集合中
            if new_sample not in data:
                valid_negative = True  # 合格的负样本
                neg_data.append((neg_head, relation, neg_tail))

    return neg_data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        knowledge = self.data[idx]

        return knowledge


def collate_fn(batch):
    pos = batch
    neg = build_neg(batch, data1, entity_dic_size)
    return {
            'pos': pos,
            'neg': neg
    }


data, entity_dic, relation_dic = get_data(file_path)
entity_dic_size = len(entity_dic)
relation_dic_size = len(relation_dic)
data1 = set(data)
dataset = Dataset(data)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, drop_last=False)

if __name__ == '__main__':
    for batch in data_loader:
        print(batch['pos'])
        print(batch['neg'])
        print(entity_dic_size)
        print(relation_dic_size)
        break

