def load_data(file_path):
    """
    加载 subgraph_kgp1.txt 中的三元组数据

    参数:
    - file_path: 数据文件的路径

    返回:
    - triples: (head, relation, tail) 的列表
    """
    triples = []

    with open(file_path, 'r') as file:
        for line in file:
            fields = line.strip().split('\t')

            if len(fields) < 8:  # 确保数据格式正确
                continue

            head = int(fields[1])  # 第2列为头实体
            relation = fields[6]  # 第7列为关系
            tail = fields[7]  # 第8列为尾实体（假设为尾实体）

            # 如果需要将 tail 也转换为实体ID，可以在此处理
            triples.append((head, relation, tail))

    return triples


# 使用示例
file_path = 'D:\PyStudy\Knowledge-embedding\dataset\subgraph_kgp1.txt'
triples = load_data(file_path)
print(triples[:5])  # 打印前5个三元组
