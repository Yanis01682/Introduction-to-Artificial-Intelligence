from gensim.models import KeyedVectors

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#加载二进制词向量文件
word2vec=KeyedVectors.load_word2vec_format(
    "Dataset/wiki_word2vec_50.bin",binary=True
)

def read_data(file_path):
    with open(file_path,"r",encoding="utf-8")as f:
        data=[]
        max_len=0
        for line in f:
            line=line.strip()
            parts=line.split()
            label=int(parts[0])
            words=parts[1:]
            #words=["死囚","爱","刽子手"]
            #把words中每一个词转为50维词向量
            #要把words转为词向量，这样的话word2vec["爱"]=...
            vectors=[]
            for word in words:
                if word in word2vec:
                    vectors.append(word2vec[word])
                else:
                    vectors.append([0.0]*50)
            #一行文本->label+words->vectors
            #这个时候，一行的words转成了一个句子长度*50的vector
            #下一步是把vectors和label转成tensor
            vectors=torch.tensor(vectors,dtype=torch.float)
            label=torch.tensor(label,dtype=torch.long)
            data.append((vectors,label))
            if vectors.shape[0]>max_len:
                max_len=vectors.shape[0]

    return data,max_len


    #tensor的结构是[句子长度,50]
    #label原来是一个一维的数，转tensor之后就是一个0维tensor
    #train_data=[
    #(vector1,label1),
    #(vector2,label2),
    #...
    #]
    #其中vector1的shape可能是[34,50]，接下来要padding，把词向量的行补成一致的
    #padding
def build_tensor(data,max_len):
    padded_vectors=[]
    padded_labels=[]
    for vectors,label in data:
        if vectors.shape[0]<max_len:
            pad=torch.zeros(max_len-vectors.shape[0],50)
            vectors=torch.cat([vectors,pad],dim=0)
            padded_vectors.append(vectors)
            padded_labels.append(label)
        else:
            vectors = vectors[:max_len]
            padded_vectors.append(vectors)
            padded_labels.append(label)

    #pad之后，padded_vectors里面的每一个元素都是[max_len,50]的tensor
    #stack一下，X是沿一个新的维度把这N个形状相同的二维张量叠起来
    #X.shape=[N,mex_len,50]
    #第 0 维：第几个句子
    #第 1 维：句子里的第几个词位置
    #第 2 维：这个词向量的 50 个数
    X=torch.stack(padded_vectors)
    y=torch.stack(padded_labels)
    #X是训练输入
    #y是训练标签
    X=X.unsqueeze(1)
    #X.shape=[N,1,max_len,50]
    #加一个通道维，把它看成单通道输出

    return X,y

train_data, train_max_len = read_data("Dataset/train.txt")
val_data, _ = read_data("Dataset/validation.txt")
test_data, _ = read_data("Dataset/test.txt")

X_train, y_train = build_tensor(train_data, train_max_len)
X_val, y_val = build_tensor(val_data, train_max_len)
X_test, y_test = build_tensor(test_data, train_max_len)

batch_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = X_train.to(device)
y_train = y_train.to(device)
X_val = X_val.to(device)
y_val = y_val.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=batch_size,
    shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=batch_size,
    shuffle=False,
)
test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=batch_size,
    shuffle=False,
)

class TextCNN(nn.Module):#所有神经网络的基类，我定义的神经网络要继承它
    def __init__(self):#初始化函数，也就是创建这个模型对象时，自动执行的函数
        super(TextCNN,self).__init__()#先把父类 nn.Module 该初始化的东西初始化一下
        num_filters = 100
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (2, 50)),
            nn.Conv2d(1, num_filters, (3, 50)),
            nn.Conv2d(1, num_filters, (4, 50)),
        ])
        self.dropout = nn.Dropout(0.5)
        #全连接层
        self.fc=nn.Linear(num_filters * 3,2)#输入300维特征，输出2维，对应二分类的两个类别
    #前向传播
    def forward(self,x):
        conv_outputs = []
        for conv in self.convs:
            conv_x = conv(x)
            conv_x = F.relu(conv_x)
            conv_x = conv_x.squeeze(3)
            conv_x = F.max_pool1d(conv_x, conv_x.size(2))
            conv_x = conv_x.squeeze(2)
            conv_outputs.append(conv_x)
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        #全连接层
        x = self.fc(x)
        return x
    
model=TextCNN().to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()#看模型给两个类别各打了多少分，看真实标签是哪一类，算出loss函数
optimizer = optim.Adam(model.parameters(), lr=0.001)#根据loss算出来的梯度，去更新模型参数


# 训练
min_val_loss=float("inf")
best_state=None#保存最好的模型参数
patience=3
wait=0

print(f"device={device}")

for epoch in range(50):
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for batch_x, batch_y in train_loader:
        output = model(batch_x)
        loss = criterion(output, batch_y)

        optimizer.zero_grad()#把旧梯度清空
        loss.backward()#反向传播，计算梯度
        optimizer.step()#沿这些梯度下降的方向修改

        train_loss_sum += loss.item() * batch_y.size(0)
        train_correct += (torch.argmax(output, dim=1) == batch_y).sum().item()
        train_total += batch_y.size(0)

    train_loss = train_loss_sum / train_total
    train_acc = train_correct / train_total

    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            val_output = model(batch_x)
            val_loss = criterion(val_output, batch_y)
            val_pred = torch.argmax(val_output, dim=1)

            val_loss_sum += val_loss.item() * batch_y.size(0)
            val_correct += (val_pred == batch_y).sum().item()
            val_total += batch_y.size(0)

    val_loss_avg = val_loss_sum / val_total
    val_acc = val_correct / val_total


    print(
        f"epoch {epoch+1}, "
        f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
        f"val_loss={val_loss_avg:.4f}, val_acc={val_acc:.4f}"
    )
    if val_loss_avg < min_val_loss:
        min_val_loss = val_loss_avg
        best_state = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1

    if wait >= patience:
        print("early stopping")
        break

if best_state is not None:
    model.load_state_dict(best_state)

model.eval()
test_correct = 0
test_total = 0

tp = 0
fp = 0
fn = 0

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        test_output = model(batch_x)
        test_pred = torch.argmax(test_output, dim=1)

        test_correct += (test_pred == batch_y).sum().item()
        test_total += batch_y.size(0)

        tp += ((test_pred == 1) & (batch_y == 1)).sum().item()
        fp += ((test_pred == 1) & (batch_y == 0)).sum().item()
        fn += ((test_pred == 0) & (batch_y == 1)).sum().item()

test_acc = test_correct / test_total
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"test_acc={test_acc:.4f}")
print(f"precision={precision:.4f}")
print(f"recall={recall:.4f}")
print(f"f1={f1:.4f}")
