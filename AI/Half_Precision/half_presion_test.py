import torch
from torch import nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import math
import os
import time

# Configration
enable_tf32 = True
dtype = torch.float16
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 2
batch_size = 64

# Defining classes
class PositionalEncoding(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, d_model, num_labels, vocab_size, nhead=4, dim_feedforward=2048, num_encoder_layers=2):
        super(Transformer, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model=d_model, vocab_size=vocab_size)

        en_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.enc = nn.TransformerEncoder(en_layer, num_encoder_layers)

        self.classifiction = nn.Linear(d_model, num_labels)
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, input_ids, mask=None):
        emb = self.embeddings(input_ids)
        out = self.pos(emb)

        out = self.enc(emb)
        out = out.mean(dim=1)

        out = self.classifiction(out)
        out = self.softmax(out)

        return out

def generate_mask(dataset):
    mask = dataset['attention_mask'].to(torch.float)
    mask = torch.bmm(mask.unsqueeze(2), mask.unsqueeze(1))
    return {'attention_mask_full': mask}

def tokenization(inputs):
    return tokenizer(inputs['text'], padding=True, truncation=True, max_length=1024)

def get_data(dataset_name='imdb'):
    dataset = load_dataset(dataset_name, split='train')
    dataset = dataset.map(tokenization, batched=True, num_proc=32)
    #dataset = dataset.map(generate_mask, batched=True, num_proc=32)
    dataset.set_format(type="torch", columns=['text', 'label', 'input_ids', 'token_type_ids', 'attention_mask'])

    dataset = dataset.train_test_split(test_size=0.05, shuffle=True)

    train_dataset = dataset['train']

    test_dataset = dataset['test']

    return train_dataset, test_dataset

# Loading pre-trained tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size

model = Transformer(d_model=1024, num_labels=2, vocab_size=vocab_size)

if dtype == torch.bfloat16:
    model_name = 'bf16'
    scaler = torch.cuda.amp.GradScaler()
elif dtype == torch.float16:
    model_name = 'fp16'
    scaler = torch.cuda.amp.GradScaler()
elif dtype == torch.float:
    model_name = 'fp32'
    scaler = None

if enable_tf32:
    torch.backends.cuda.matmul.allow_tf32 = True
    model_name = model_name + '_tf32'

if os.path.exists(model_name + '.pt'):
    model.load_state_dict(torch.load(model_name + '.pt'))

# Loading Dataset
train_dataset, test_dataset = get_data()
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=batch_size)

# Trainer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

model.to(device)
model.train()
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader):
        start = time.time()
        # get the inputs; data is a list of [inputs, labels]
        input_ids, labels = data['input_ids'].to(device), data['label'].to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        if scaler:
            with torch.amp.autocast(device_type=device, dtype=dtype):
                outputs = model(input_ids)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 2000 mini-batches
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for data in testloader:
                    input_ids = data['input_ids'].to(device)
                    predictions = model(input_ids).cpu()
                    labels = data['label']

                    correct += (predictions.argmax(axis=1) == labels).sum().item()
                    total += len(labels)

            end = time.time()
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 9:.3f}, Test Accuracy: {correct/total :.3f}, Speed: {end-start:.3f}s/it')
            running_loss = 0.0


            model.train()

# Saving Model
torch.save(model.state_dict(), model_name + '.pt')


print('Finished Training')