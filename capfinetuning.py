# args = {
#     'train_img_path' : '/content/images',
#     'path_description' :'/content/imgDes.txt',
#     'MAX_SEQ_LEN' :120,
#     'LR':5e-5,
#     'IMAGE_SIZE' :256,
#     'BATCH_SIZE' :10,
#     'PATH' :'/content/model.pth',
#     'EPOCHS' : 10,
#     train_bool = False,
#     prediction_image = '/content/images/1001896054.jpg',
# }


import warnings
import sys
warnings.filterwarnings('ignore')

import torch
import transformers
import torchvision
import torch.nn as nn
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

import cv2
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm.autonotebook import tqdm

from sklearn.model_selection import train_test_split

from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk import Text

import editdistance
from torchsummary import summary

seed = 42


# In[49]:

torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
from IPython.display import clear_output
from kobert_transformers import get_tokenizer


tokenizer = get_tokenizer()

MAX_SEQ_LEN = int(sys.argv[3])  #최대 글자 수 
LR = float(sys.argv[4]) # Learning Rate
EPOCHS = int(sys.argv[8])
IMAGE_SIZE = int(sys.argv[5]) #256x256
BATCH_SIZE = int(sys.argv[6])  
HIDDEN = 512 
ENC_LAYERS = 4 # transformer encoder layer 수
DEC_LAYERS = 4 # transformer decoder layer 수
N_HEADS = 8 
DROPOUT = 0.1
PATH = r'{sys.argv[7]}'
VOCAB_SIZE = tokenizer.vocab_size
clear_output()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Currently using "{device.upper()}" device')


# In[50]:



path = sys.argv[1]  #여기서 이미지 파일 조정 가능 
data = np.zeros((31, 2), dtype=np.object) #현재 158916개의 데이터셋
i = 0
for line in open(sys.argv[2], 'r', encoding='utf8'):
    data[i, :] = line.replace('\n', "").split('|')
    i += 1
    
df = pd.DataFrame(data=data[1:, :], columns=data[0, :])
df.sample(5)
df.head()

# In[51]:


train_transforms = T.Compose([
                              T.ToPILImage(),
                              T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                              T.RandomCrop(IMAGE_SIZE),
                              T.ColorJitter(brightness=(0.95, 1.05),
                                            contrast=(0.95, 1.05),
                                            saturation=(0.98, 1.02),
                                            hue=0.05),
                              T.RandomHorizontalFlip(p=0.1),
                              T.GaussianBlur(kernel_size=(1, 3), sigma=(0.1, 0.5)),
                              T.RandomAdjustSharpness(sharpness_factor=1.2, p=0.2),
                              T.RandomRotation(degrees=(-5, 5)),
                              T.ToTensor(),
                              T.Normalize(mean=[0.485, 0.456, 0.406], 
                                          std=[0.229, 0.224, 0.225])
])

valid_transforms = T.Compose([
                              T.ToPILImage(),
                              T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                              T.ToTensor(),
                              T.Normalize(mean=[0.485, 0.456, 0.406], 
                                          std=[0.229, 0.224, 0.225])
])
invTrans = T.Compose([T.Normalize(mean = [ 0., 0., 0. ],
                                      std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                          T.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                      std = [ 1., 1., 1. ]),
                          T.ToPILImage(),
                         ])


# In[52]:

root = sys.argv[1]
class CapDataset(Dataset):
    def __init__(self, df, root, tokenizer, transforms=None):
        self.df = df
        self.root = root
        self.tokenizer = tokenizer
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        row = self.df.iloc[ix].squeeze()
        id = row.image_name
        image_path = f'{self.root}/{id}'

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        caption = row.caption_text

        target = tokenizer(caption, 
                           return_token_type_ids=False, 
                           return_attention_mask=False, 
                           max_length=MAX_SEQ_LEN, 
                           padding="do_not_pad",
                           return_tensors="pt")
        
        target = target['input_ids'].squeeze()
        target = torch.LongTensor(target)
        return image, target, caption

    def collate_fn(self, batch):
        batch.sort(key=lambda x: len(x[1]), reverse=True)
        images, targets, captions = zip(*batch)
        images = torch.stack([self.transforms(image) for image in images], 0)
        lengths = [len(tar) for tar in targets]
        _targets = torch.zeros(len(captions), max(lengths)).long()
        for i, tar in enumerate(targets):
            end = lengths[i]
            _targets[i, :end] = tar[:end] 
        _targets = _targets.permute(1,0)
        return images.to(device), _targets.to(device), torch.tensor(lengths).long().to(device)


# In[53]:


train, test = train_test_split(df, test_size=0.01, shuffle=True, random_state=seed)
train, valid = train_test_split(train, test_size=0.1, shuffle=True, random_state=seed)
train.reset_index(drop=True, inplace=True)
valid.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)
print(f'Train size: {train.shape[0]}, valid size: {valid.shape[0]}, test size: {test.shape[0]}')


# In[54]:


train_dataset = CapDataset(train, root, tokenizer, train_transforms)
valid_dataset = CapDataset(valid, root, tokenizer, valid_transforms)
test_dataset = CapDataset(test, root, tokenizer, valid_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_dataset.collate_fn, drop_last=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=valid_dataset.collate_fn, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=test_dataset.collate_fn)


# In[55]:


class Encoder(nn.Module):
    def __init__(self, encoded_image_size=8, pretrained=True):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=pretrained)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        for param in self.resnet.parameters():
            param.requires_grad = False

        for child in list(self.resnet.children())[5:]:
            for param in child.parameters():
                param.requires_grad = True
                
        self.fc = nn.Sequential(nn.Conv2d(2048, 512, 1),
                                nn.LeakyReLU(0.01, inplace=True))
        #Encoder 정의 
        #Convolution layer와 RELU 활용        

    def forward(self, images):
        out = self.resnet(images)
        out = self.fc(out)
        out = self.adaptive_pool(out) # [B, C, H, W] pooling layer
        #out = out.permute(0, 3, 1, 2)
        out = out.flatten(2) # [B, C, HW] flattern layer 
        out = out.permute(2,0,1) # [HW, B, C] 
        return out
        
        # [HW, B, C]는 vector의 shape

class Encoder_50(nn.Module):
    def __init__(self, bb_name='resnet50', hidden=HIDDEN, pretrained=False):
        super(Encoder_50, self).__init__()
        self.backbone = torchvision.models.__getattribute__(bb_name)(pretrained=pretrained)
        self.backbone.fc = nn.Conv2d(2048, hidden, 1)
    
    def forward(self, src):
        x = self.backbone.conv1(src)

        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x) # [32, 2048, 8, 8] : [B,C,H,W]
            
        x = self.backbone.fc(x) # [32, 512, 8, 8] : [B,C,H,W]
        # x = x.permute(0, 3, 1, 2) # [64, 8, 512, 8] : [B,W,C,H]
        x = x.flatten(2) # [32, 512, 64] : [B,C,WH]
        #x = x.permute(1, 0, 2) # [64, 32, 512] : [W,B,CH]
        x = x.permute(2,0,1)
        return x
    
class TransformerModel(nn.Module):
  #TransformerModel 정의
    def __init__(self, outtoken, hidden, enc_layers=1, dec_layers=1, nhead=1, dropout=0.1, pretrained=False):
        super(TransformerModel, self).__init__()
        self.backbone = Encoder_50(pretrained=pretrained)
        
        self.pos_emb = nn.Embedding.from_pretrained(self.get_position_embedding_table(), freeze=True) ##
        self.pos_encoder = PositionalEncoding(hidden, dropout)
        self.decoder = nn.Embedding(outtoken, hidden, padding_idx=0)
        self.pos_decoder = PositionalEncoding(hidden, dropout)
        self.transformer = nn.Transformer(d_model=hidden, nhead=nhead, num_encoder_layers=enc_layers,
                                          num_decoder_layers=dec_layers, dim_feedforward=hidden * 4, dropout=dropout,
                                          activation='gelu')  # 'relu' or torch.nn.functional.leaky_relu

        self.fc_out = nn.Linear(hidden, outtoken)
        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None
        
    def get_position_embedding_table(self, num_pix=8, ch_pos=256): ##
        def cal_angle(position, hid_idx):
            x = position % num_pix
            y = position // num_pix
            x_enc = x / np.power(10000, hid_idx / ch_pos)
            y_enc = y / np.power(10000, hid_idx / ch_pos)
            return np.sin(x_enc), np.sin(y_enc)
        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_idx)[0] for hid_idx in range(ch_pos)] + [cal_angle(position, hid_idx)[1] for hid_idx in range(ch_pos)]

        embedding_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(int(num_pix**2))])
        return torch.FloatTensor(embedding_table).to(device)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)

    def forward(self, src, trg):
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(device) 
        x = self.backbone(src)
        
        batch_size = x.size(1) ##
        positions = x.size(0) ##
        
        src_pad_mask = self.make_len_mask(x[:, :, 0])
        #src = self.pos_encoder(x)
        pos_emb = self.pos_emb(torch.LongTensor([list(range(positions))]*batch_size).to(device)) # [32,64,512]
        src = x + pos_emb.permute(1,0,2) #
        
        trg_pad_mask = self.make_len_mask(trg)
        trg = self.decoder(trg)
        trg = self.pos_decoder(trg)

        output = self.transformer(src, trg, src_mask=self.src_mask, tgt_mask=self.trg_mask,
                                  memory_mask=self.memory_mask,
                                  src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask,
                                  memory_key_padding_mask=src_pad_mask) #  [HW,B,C]
        output = self.fc_out(output) # [L,B,C]

        return output
    
class PositionalEncoding(nn.Module):
  #Position Embading
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x) 


# In[56]:


def train_one_batch(model, data, optimizer, criterion):
    model.train()
    image, target, _ = data
    optimizer.zero_grad()
    output = model(image, target[:-1, :])
    
    if torch.any(torch.isnan(output)): 
        output = torch.nan_to_num(output)
        
    loss = criterion(output.view(-1, output.shape[-1]), torch.reshape(target[1:, :], (-1,)))
    loss.backward()
    optimizer.step()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0) 
    
    accuracy = (output.argmax(2).flatten() == target[1:, :].flatten()).float().mean().item()
    return loss.item(), accuracy

@torch.no_grad()
def validate_one_batch(model, data, criterion):
    model.eval()
    image, target, _ = data
    output = model(image, target[:-1, :])
    if torch.any(torch.isnan(output)):
        output = torch.nan_to_num(output)
    loss = criterion(output.view(-1, output.shape[-1]), torch.reshape(target[1:, :], (-1,)))
    accuracy = (output.argmax(2).flatten() == target[1:, :].flatten()).float().mean().item()
    return loss.item(), accuracy

@torch.no_grad()
def prediction(model, filepath='random', max_len=MAX_SEQ_LEN, tokenizer=tokenizer, plot=True):
    label = None
    if filepath == 'random':
        idx = np.random.randint(len(test))
        filepath = root + '/' + test.iloc[idx, 0]
        label = test.iloc[idx, -1]

    model.eval()
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = valid_transforms(img)
    src = torch.FloatTensor(image).unsqueeze(0).to(device)

    out_indexes = [101, ]

    for i in range(max_len):
                
        trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(device)
                
        output = model(src, trg_tensor)
        out_token = output.argmax(2)[-1].item()
        if out_token == 102:
            break
        out_indexes.append(out_token)

    preds = tokenizer.decode(out_indexes[1:])
    col_pled = preds.replace('[SEP]','')
    if plot:
        plt.figure(figsize=(6,4))
        plt.title("Predicted / Truth")
        plt.imshow(img)
        plt.tight_layout()
        plt.show()
        plt.pause(0.001)
        print(f'Prediction: {col_pled}, \nTruth: {label if label is not None else "NO label"}')
    
    return preds


# In[57]:


def bleu_score_fn(method_no: int = 4, ref_type='corpus'):
    smoothing_method = getattr(SmoothingFunction(), f'method{method_no}')
    
    def bleu_score_corpus(reference_corpus: list, candidate_corpus: list, n: int = 4):
        weights = [1 / n] * n
        return corpus_bleu(reference_corpus, candidate_corpus,
                           smoothing_function=smoothing_method, weights=weights)

    def bleu_score_sentence(reference_sentences: list, candidate_sentence: list, n: int = 4):
        weights = [1 / n] * n
        return sentence_bleu(reference_sentences, candidate_sentence,
                             smoothing_function=smoothing_method, weights=weights)
    if ref_type == 'corpus':
        return bleu_score_corpus
    elif ref_type == 'sentence':
        return bleu_score_sentence
    #이미지 캡셔닝 특성상 정답 sequnce와 비교하는 것이기 때문에 BLUE로 evaluation하는 경우가 많음.
@torch.no_grad()
def evaluate_model(dataloader, model, bleu_score_fn, tokenizer):
    running_bleu = [0.0] * 5
    model.eval()
    for batch_idx, batch in enumerate(tqdm(dataloader, leave=False)):
        images, captions, _ = batch
        outputs = []
        for j in range(images.size(0)):
            src = images[j, ...].unsqueeze(0)
            out_indexes = [101, ]
            for i in range(MAX_SEQ_LEN):
                trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(device)      
                output = model(src, trg_tensor)
                out_token = output.argmax(2)[-1].item()
                if out_token == 102:  # [SEP]
                    break
                out_indexes.append(out_token)
            preds = tokenizer.decode(out_indexes[1:]).replace('[PAD]', '').split()
            outputs.append(preds)
            
        captions = [tokenizer.decode(captions[:,i]).replace('[PAD]', '').split()[1:-1] for i in range(captions.size(1))]
        for i in (1, 2, 3, 4):
            running_bleu[i] += bleu_score_fn(reference_corpus=captions, candidate_corpus=outputs, n=i)
    for i in (1, 2, 3, 4):
        running_bleu[i] /= len(dataloader)
    return running_bleu


# In[58]:


model = TransformerModel(VOCAB_SIZE, hidden=HIDDEN, enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS,   
                         nhead=N_HEADS, dropout=DROPOUT, pretrained=True).to(device)  # pretrained False initially
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, min_lr=1e-6, factor=0.1)
corpus_bleu_score_fn = bleu_score_fn(4, 'corpus')


# In[ ]:


train_losses, valid_losses, train_accuracies, valid_accuracies = [], [], [], []
timeout = 360.  # minutes before stop
start_time = time.time()



if sys.argv[9] == 'True':
    for epoch in range(EPOCHS):
        print(f'{epoch+1}/{EPOCHS} epoch.')
        epoch_train_losses, epoch_valid_losses, epoch_train_accuracy, epoch_valid_accuracy = [], [], [], []
        tk0 = tqdm(train_dataloader, total=len(train_dataloader), leave=True)

        for i, batch in enumerate(tk0):
            loss, train_accuracy = train_one_batch(model, batch, optimizer, criterion)
            epoch_train_losses.append(loss)
            epoch_train_accuracy.append(train_accuracy)
            tk0.set_postfix(loss=loss)
            if (i+1) % 100 == 0:
                pred = prediction(model)

        train_epoch_loss = np.array(epoch_train_losses).mean()
        train_losses.append(train_epoch_loss)
        train_accuracy = np.array(epoch_train_accuracy).mean()
        train_accuracies.append(train_accuracy)

        tk1 = tqdm(valid_dataloader, total=len(valid_dataloader), leave=True)
        for _, batch in enumerate(tk1):
            loss, valid_accuracy = validate_one_batch(model, batch, criterion)
            epoch_valid_losses.append(loss)
            epoch_valid_accuracy.append(valid_accuracy)
            tk1.set_postfix(loss=loss)

        valid_epoch_loss = np.array(epoch_valid_losses).mean()
        valid_losses.append(valid_epoch_loss)
        valid_accuracy = np.array(epoch_valid_accuracy).mean()
        valid_accuracies.append(valid_accuracy)

        print(f'Epoch {epoch+1} summary:')
        print(f'Train loss: {train_epoch_loss:.4f}, validation loss: {valid_epoch_loss:.4f}')
        print(f'Train accuracy: {train_accuracy:.4f}, validation accuracy: {valid_accuracy:.4f}')
        print(f'Time per {epoch+1} epoch: {(time.time() - start_time)//60} minutes')

        scheduler.step(valid_epoch_loss)

        if (epoch+1) % 2 == 0:
            print(f'Prediction after {epoch+1} epoch.')
            pred = prediction(model)
            torch.save(model.state_dict(), sys.argv[7])

        if (epoch+1) % 5 == 0:
            test_bleu = evaluate_model(test_dataloader, model, corpus_bleu_score_fn, tokenizer)
            print(''.join([f'test_bleu{i}: {test_bleu[i]:.4f} ' for i in (1, 4)]))

        if (time.time() - start_time)//60 > timeout:
            print(f'Timeout stop.')
            torch.save(model.state_dict(), sys.argv[7])
            break


#train은 가능합니다. 시간이 오래 걸려 일단은 중간에 멈췄습니다.
#슈퍼컴퓨팅 자원을 쓰면 쉽게 train 가능합니다.
#나희씨가 만드신 dataset도 전처리하면 이 모델에서 적용가능합니다. (description은 영어로 번역된 상태이긴 합니다.)


elif sys.argv[9] == 'False':
    pred = prediction(model,filepath=str(sys.argv[10]))
