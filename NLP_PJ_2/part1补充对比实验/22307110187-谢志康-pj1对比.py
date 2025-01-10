from torch.utils.data import Dataset

class CHisIECDataset(Dataset):
    label_label_id_mapping = {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "E-PER": 3,
        "S-PER": 4,
        "B-LOC": 5,
        "I-LOC": 6,
        "E-LOC": 7,
        "S-LOC": 8,
        "B-OFI": 9,
        "I-OFI": 10,
        "E-OFI": 11,
        "S-OFI": 12,
        "B-BOOK": 13,
        "I-BOOK": 14,
        "E-BOOK": 15,
        "S-BOOK": 16,
    }

    def __init__(self, path) -> None:
        super().__init__()
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            d = [[], []]
            while line := f.readline():
                line = line.strip()
                if line:
                    word, label = line.split()
                    d[0].append(word)
                    d[1].append(self.label_label_id_mapping[label])
                elif d[0]:
                    self.data.append(tuple(d))
                    d = [[], []]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

from torch.utils.data import DataLoader
from torch.nn.functional import one_hot

# 选择batch_size为32。16、64、128均尝试过，当选择16的时候太小了，迭代多轮epoch时训练过程极其不稳定，
# 具体表现为accuracy和f1-macro时而大幅下降且难以优化到能力上限。
# 64与32效果差不多，32略好一些，选为128应该是太大了，总体训练集都不是特别大，能力上限很低。
batch_size = 64

# 创建一个数据加载器（DataLoader）用于批量加载数据。
# 参数: dataset: 输入的数据集，通常是一个实现了__getitem__和__len__方法的对象。
#      shuffle: 是否在每个epoch中打乱数据顺序，默认值为True。
# 返回: DataLoader: 一个数据加载器对象，用于按批次加载数据。
def get_dataloader(dataset, shuffle=True):
    # 数据收集函数，用于将一批数据进行整理。
    def collect_fn(batch):
        t = batch[0][0]
        # 一个张量，包含这一批样本的标签，经过one-hot编码处理。
        l = one_hot(torch.tensor(batch[0][1], dtype=torch.int64), 17).float()
        return t, l

    return DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        collate_fn=collect_fn,
    )

# 读取训练集、验证集、测试集
train_set = CHisIECDataset("./train.txt")
dev_set = CHisIECDataset("./dev.txt")
test_set = CHisIECDataset("./test.txt")
# 训练集随机打乱
train_loader = get_dataloader(train_set)
val_loader = get_dataloader(dev_set, shuffle=False)
test_loader = get_dataloader(test_set, shuffle=False)

from torch import nn
from torchtext.vocab import Vectors
import torch

# 核心，自定义的网络结构，主要使用要求的bilstm网络（双层），后接多头自注意力机制
class MyAwesomeModel(nn.Module):
    def __init__(self, embed_dim=50, hidden_dim=128, num_heads=8, num_layers=2, dropout_rate=0.25) -> None:
        super().__init__()
        self.vectors = Vectors(
            name="gigaword_chn.all.a2b.uni.ite50.vec",
            cache=".",
        )
        # 核心encoder，双向LSTM网络，嵌入层维度转为隐藏层维度，使用两层
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True,num_layers=num_layers)
        # 层正则化
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        # 自注意力机制 (Multihead Attention)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=num_heads)
        # 线性层映射到分类维度
        self.classifier = nn.Linear(hidden_dim * 2, 17)
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: str):
        x = self.vectors.get_vecs_by_tokens(x)  # 获取输入字符串的词向量
        # 将输入扩展维度以符合LSTM的输入要求 # [batch_size, seq_len, hidden_dim * 2]
        x, _ = self.lstm(x.unsqueeze(0))
        x = self.layer_norm(x)
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output  # 残差连接
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.classifier(x[0])
        return x

from torch.optim import Adam
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import StepLR

model = MyAwesomeModel()
# 加入L2正则化可以让模型的权重不至于过于复杂，减轻过拟合的风险。在优化器中加入weight decay
# 但我在训练过程中发现accuracy和f1的值有上界，到瓶颈很难突破了，更像是一种欠拟合。
optimizer = Adam(model.parameters(), lr=0.0015)
# 添加学习率调度器 每6个epoch将学习率减为90%
scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

# 自定义损失函数：CrossEntropyLoss对类不均衡不太敏感。
# 使用Focal Loss来处理类间不平衡问题，有助于提升F1-macro分数，尤其是召回率较低的类
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 用于调整类别之间的权重，如果类别不均衡可以通过调整alpha（可调整的超参数）
        self.gamma = gamma  # 控制难易样本的权重（NER情景下用不到）
        self.ignore_index = -1  # 标记：无效label在上个类中标记为-1

    def forward(self, inputs, targets):
        valid_idx = targets != self.ignore_index  # 获取有效的标签，即不为ignore_index的部分
        inputs = inputs[valid_idx]  # 形状为(batch_size, num_classes)的模型输出
        targets = targets[valid_idx]  # 形状为 (batch_size) 的标签
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')  # 获取交叉熵损失
        pt = torch.exp(-ce_loss)  # 预测的概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  # 计算 focal loss
        return focal_loss.mean()

loss_fn = nn.CrossEntropyLoss()

# 早停法，epoch迭代训练loss无优化时提前停止（允许连续两次无优化）
class EarlyStopping:
    def __init__(self, patience=2):
        self.patience = patience  # 在验证损失未改善的情况下等待的epoch数
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0  # reset
        else:
            self.counter += 1
        return self.counter >= self.patience
early_stopping = EarlyStopping()

def train(loader):
    model.train()
    epoch_loss = []
    for x, label in loader:
        optimizer.zero_grad()
        pred = model(x)
        try:
            loss = loss_fn(pred, label)
        except:
            print(pred.shape, label.shape)
        epoch_loss.append(loss.item())
        loss.backward()
        # 梯度裁剪 防止梯度爆炸带来的不稳定性
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
        optimizer.step()
    return {"loss": sum(epoch_loss) / len(epoch_loss)}

def eval(loader):
    model.eval()
    pred = []
    target = []
    for x, y in loader:
        _pred = model(x).argmax(-1)
        pred += _pred.tolist()
        _target = y.argmax(-1)
        target += _target.tolist()
    return {
        "accuracy": accuracy_score(target, pred),
        "f1_macro": f1_score(target, pred, average="macro"),
    }


from tqdm import trange

for epoch in trange(200, desc="Epoch"):
    metrics = train(train_loader)
    scheduler.step()
    with torch.no_grad():
        metrics = {**eval(val_loader), **metrics}
    print(metrics)

print(eval(test_loader))