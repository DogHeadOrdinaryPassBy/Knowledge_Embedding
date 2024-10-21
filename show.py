import os
import numpy as np
import matplotlib.pyplot as plt
import torch

plt.switch_backend('agg')
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from dataloader import entity_dic,relation_dic

def visualize_data_3d_interactive_v1(data_dict, vocab_dict, save_path):

    save_path = os.path.splitext(save_path)[0]

    # 准备空列表来存储所有数据点和颜色
    all_data = []
    colors = []
    texts = []  # 用于存储文本标签
    pid_index = 0  # 用于为每个pid分配不同的颜色

    # 提取颜色映射
    color_palette = plt.cm.get_cmap('tab10', len(data_dict))

    # 将每个pid的数据点收集到一起，并为第一个特征添加标签
    for pid, features in data_dict.items():
        all_data.append(features)
        colors.extend([color_palette(pid_index)] * features.shape[0])

        # 使用词表为每个数据点打上标签
        vocab = vocab_dict.get(pid, {})
        reverse_vocab = {index: word for word, index in vocab.items()}  # 创建逆向映射字典
        feature_labels = [reverse_vocab.get(i, "") for i in range(features.shape[0])]
        texts.extend(feature_labels)

        pid_index += 1

    # 将所有数据合并成一个大矩阵
    all_data = np.vstack(all_data)

    # 使用PCA将数据降到3维
    print('PCA is processing ...')
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(all_data)

    # 创建一个3D散点图显示PCA结果
    fig_pca = go.Figure(data=[go.Scatter3d(
        x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
        mode='markers+text',
        text=texts,
        marker=dict(
            size=5,
            color=['rgb({}, {}, {})'.format(c[0] * 255, c[1] * 255, c[2] * 255) for c in colors],  # 将颜色转换为plotly格式
            opacity=0.8
        )
    )])
    fig_pca.update_layout(title='PCA Results', scene=dict(
        xaxis_title='Component 1',
        yaxis_title='Component 2',
        zaxis_title='Component 3'))
    fig_pca.write_html(save_path + '_pca.html')


if __name__ == '__main__':
    # test v1
    tensor1 = torch.load('transH_entity_embedding.pt').to('cpu')
    tensor2 = torch.load('transH_relation_embedding.pt').to('cpu')

    entity_vocab = entity_dic
    relation_vocab = relation_dic
    data_dict = {
        'entity': tensor1[:200, :],
        'relation': tensor2
    }

    vocab_dict = {
        'entity': entity_vocab,
        'relation': relation_vocab,
    }

    visualize_data_3d_interactive_v1(data_dict, vocab_dict, 'outputs/vis_hdim_vector_interactive_v1_2.html')
##此注释为修改submit内容用