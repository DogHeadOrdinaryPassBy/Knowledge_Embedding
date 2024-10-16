import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_score(head, relation, tail):
    return torch.norm(head + relation - tail, p=2, dim=1)


def project_H(head, norm):
    norm = F.normalize(norm, p=2, dim=-1)
    return head - norm * torch.sum(head * norm, dim=1, keepdim=True)


def project_R(head, project):
    return torch.mm(head, project)


class DistanceModel(nn.Module):
    def __init__(self, num_entities, num_relations, entity_embedding_dim, margin=1.0, function='transE', c=0.1
                 , relation_embedding_dim=None):
        super(DistanceModel, self).__init__()
        self.margin = margin
        self.embedding_dim = entity_embedding_dim
        self.relation_embedding_dim = entity_embedding_dim

        if function == 'transR' and relation_embedding_dim is not None:
            self.relation_embedding_dim = relation_embedding_dim

        self.function = function
        self.C = c

        # 嵌入矩阵
        self.entity_embedding_W = nn.Embedding(num_entities, self.embedding_dim)
        self.relation_embedding_W = nn.Embedding(num_relations, self.relation_embedding_dim)

        # 初始化
        nn.init.xavier_uniform_(self.entity_embedding_W.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding_W.weight.data)

        # 归一化
        self.entity_embedding_W.weight.data = nn.functional.normalize(self.entity_embedding_W.weight.data, p=2, dim=1)
        self.relation_embedding_W.weight.data = nn.functional.normalize(self.relation_embedding_W.weight.data, p=2,
                                                                        dim=1)

        if self.function == 'transH':
            self.relation_norm_embedding = nn.Embedding(num_relations, self.relation_embedding_dim)
            nn.init.xavier_uniform_(self.relation_norm_embedding.weight.data)

        if self.function == 'transR':
            self.project_W = nn.Parameter(torch.rand(self.embedding_dim, self.relation_embedding_dim))
            nn.init.xavier_uniform_(self.project_W)

    def get_loss(self, pos_score, neg_score, l=None):

        loss = torch.mean(F.relu(self.margin + pos_score - neg_score))
        if l is not None:
            loss = loss + l

        return loss

    def transe_function(self, pos_entity_head, pos_relations, pos_entity_tail, neg_entity_head, neg_relations,
                        neg_entity_tail):

        pos_score = get_score(pos_entity_head, pos_relations, pos_entity_tail)
        neg_score = get_score(neg_entity_head, neg_relations, neg_entity_tail)

        return pos_score, neg_score, None

    def transh_function(self, pos_entity_head, pos_relations, pos_entity_tail, pos_relations_norm, neg_entity_head,
                        neg_relations, neg_entity_tail, neg_relations_norm):

        pos_h_project = project_H(pos_entity_head, pos_relations_norm)
        pos_t_project = project_H(pos_entity_tail, pos_relations_norm)
        neg_h_project = project_H(neg_entity_head, neg_relations_norm)
        neg_t_project = project_H(neg_entity_tail, neg_relations_norm)

        pos_score = get_score(pos_h_project, pos_relations, pos_t_project)
        neg_head_score = get_score(neg_h_project, neg_relations, neg_t_project)

        L = torch.sum((torch.sum(pos_relations_norm * pos_relations, dim=1)) ** 2)

        return pos_score, neg_head_score, L

    def transr_function(self, pos_entity_head, pos_relations, pos_entity_tail, neg_entity_head, neg_relations,
                        neg_entity_tail):
        pos_head_project = project_R(pos_entity_head, self.project_W)
        pos_tail_project = project_R(pos_entity_tail, self.project_W)
        neg_head_project = project_R(neg_entity_head, self.project_W)
        neg_tail_project = project_R(neg_entity_tail, self.project_W)

        pos_score = get_score(pos_head_project, pos_relations, pos_tail_project)
        neg_head_score = get_score(neg_head_project, neg_relations, neg_tail_project)

        return pos_score, neg_head_score, None

    def forward(self, pos, neg):
        # input(pos or neg):batch_size * 3

        # 正样本
        pos_entity_head = self.entity_embedding_W(pos[:, 0])
        pos_relations = self.relation_embedding_W(pos[:, 1])
        pos_entity_tail = self.entity_embedding_W(pos[:, 2])

        # 负样本
        neg_entity_head = self.entity_embedding_W(neg[:, 0])
        neg_relations = self.relation_embedding_W(neg[:, 1])
        neg_entity_tail = self.entity_embedding_W(neg[:, 2])

        pos_score, neg_score, l = 0, 0, 0

        if self.function == 'transE':
            pos_score, neg_score, l = self.transe_function(pos_entity_head, pos_relations, pos_entity_tail,
                                                           neg_entity_head, neg_relations, neg_entity_tail)

        elif self.function == 'transH':
            self.relation_norm_embedding.weight.data = nn.functional.normalize(self.relation_norm_embedding.weight.data,
                                                                               p=2, dim=1)
            pos_relations_norm = self.relation_norm_embedding(pos[:, 1])
            neg_relations_norm = self.relation_norm_embedding(neg[:, 1])
            pos_score, neg_score, l = self.transh_function(pos_entity_head, pos_relations, pos_entity_tail,
                                                           pos_relations_norm, neg_entity_head, neg_relations,
                                                           neg_entity_tail, neg_relations_norm)

        elif self.function == 'transR':
            pos_score, neg_score, l = self.transr_function(pos_entity_head, pos_relations, pos_entity_tail,
                                                           neg_entity_head, neg_relations, neg_entity_tail)

        return pos_score, neg_score, l
