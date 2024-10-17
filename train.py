import torch
from matplotlib import pyplot as plt
from tqdm import trange
from module.KgEmbedding import DistanceModel
from dataloader import data_loader, entity_dic_size, relation_dic_size, batch_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

margin = 1.5
embedding_dim = 100
rel_embedding_dim = 128
learning_rate = 1e-3
epochs = 80
function = 'transR'

model = DistanceModel(entity_dic_size, relation_dic_size, embedding_dim, margin, function, c=0.1,
                      relation_embedding_dim=rel_embedding_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
model.zero_grad()

losses = []
with trange(epochs, desc='epoch', ncols=100) as pbar:
    for i, epoch in enumerate(pbar, start=1):
        epoch_loss = 0
        for j, batch in enumerate(data_loader, start=1):
            pos, neg= batch['pos'], batch['neg']

            pos, neg = torch.tensor(pos).to(device), torch.tensor(neg).to(device)

            optimizer.zero_grad()

            pos_score, neg_score, l = model(pos, neg)
            loss = model.get_loss(pos_score, neg_score, l)

            loss.backward()

            optimizer.step()

            pbar.set_description(f'epoch:{i}--' + f'batch:{j},' + f'batch_loss={loss.item():.6f}')
            epoch_loss += loss.item()
        print(f'epoch_total_loss={epoch_loss:.4f}')
        losses.append(epoch_loss)

torch.save(model, f'save_weight/{function}_{batch_size}_{margin}_{epochs}.pkl')
torch.save(model.entity_embedding_W.weight.data, f'save_weight/{function}_{batch_size}_{margin}_{epochs}_entity_embedding.pt')
torch.save(model.relation_embedding_W.weight.data, f'save_weight/{function}_{batch_size}_{margin}_{epochs}_relation_embedding.pt')
if function == 'transH':
    torch.save(model.relation_norm_embedding.weight.data, f'save_weight/{function}_{batch_size}_{margin}_{epochs}_norm_embedding.pt')
if function == 'transR':
    torch.save(model.project_W, f'save_weight/{function}_{batch_size}_{margin}_{epochs}_project_W.pt')

# 绘制损失函数图像
plt.plot(range(1, epochs + 1), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.grid()
plt.savefig(f'{function}_loss.png')
plt.show()
