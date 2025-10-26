import os
from os.path import exists 
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax,pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torch.utils.data import DataLoader
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset


class SimpleMapDataset(Dataset):
        def __init__(self, iterable):
            self._data = list(iterable)

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            return self._data[idx]

def to_map_style_dataset(iterable):
        return SimpleMapDataset(iterable)

def build_vocab_from_iterator(iterator, specials=None, min_freq=1):
        from collections import Counter

        counter = Counter()
        for tokens in iterator:
            counter.update(tokens)

        stoi = {}
        if specials:
            for s in specials:
                if s not in stoi:
                    stoi[s] = len(stoi)

        for token, freq in counter.items():
            if freq >= min_freq and token not in stoi:
                stoi[token] = len(stoi)

        class Vocab:
            def __init__(self, stoi):
                self._stoi = dict(stoi)
                self._itos = {i: s for s, i in stoi.items()}

            def __len__(self):
                return len(self._stoi)

            def lookup_token(self, token, unk_token="<unk>"):
                return self._stoi.get(token, self._stoi.get(unk_token, 0))

            def __call__(self, tokens):
                return [self.lookup_token(t) for t in tokens]

            @property
            def get_stoi(self):
                return self._stoi

        return Vocab(stoi)


# --------------------------
# 第一部分 模型架构
# --------------------------

class EncoderDecoder(nn.Module):
    """标准的编码器-解码器架构。该架构以及许多其他模型的基础。"""
    def __init__(self, encoder,decoder,src_embed,tgt_embed,generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    
    def encode(self,src,src_mask):
        return self.encoder(self.src_embed(src),src_mask)
    
    def decode(self,memory,src_mask,tgt,tgt_mask):
        return self.decoder(self.tgt_embed(tgt),memory,src_mask,tgt_mask)
        
    def forward(self,src,tgt,src_mask,tgt_mask):
        """接收并处理掩蔽的 src 和目标序列"""
        return self.decode(self.encode(src,src_mask),src_mask,tgt,tgt_mask)
    
class Generator(nn.Module):
    """定义标准线性+softmax生成步骤。"""
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model,vocab)
        
    def forward(self,x):
        return log_softmax(self.proj(x),dim=-1)
    
# --------------------------
# 编码器和解码器堆栈
# --------------------------
"""
编码器 Encoder
编码器由一堆 N=6 个相同的层组成。
"""
def clones(module,N):
    """产生 N 个相同的层。"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    """构建一个层归一化模块。"""
    def __init__(self,features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))#归一化后的缩放参数 (Gain)
        self.b_2 = nn.Parameter(torch.zeros(features))#归一化后的偏移参数 (Bias)
        self.eps = eps
    
    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True)
        return self.a_2 *(x-mean)/(std + self.eps) + self.b_2    
        
    

class Encoder(nn.Module):
    """“核心编码器是 N 层的堆栈”"""
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        """“将输入（和掩码）依次传递到每一层。”"""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class SublayerConnection(nn.Module):
    """残差连接后跟层范数。注意，为了代码简单起见，范数是第一个而不是最后一个。"""
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        """“将残差连接应用于任何具有相同大小的子层。”"""
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    """“编码器由自注意力和前馈（定义如下）组成”"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size,dropout),2)
        self.size = size
    
    def forward(self, x, mask):
        x = self.sublayer[0](x,lambda x: self.self_attn(x,x,x,mask))
        return self.sublayer[1](x,self.feed_forward)

"""
解码器 Decoder
解码器由一堆 N=6 个相同的层组成。
"""
class Decoder(nn.Module):
    """带有掩蔽功能的通用 N 层解码器。"""
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x,memory,src_mask,tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    """“解码器由 self-attn、src-attn 和前馈（定义如下）组成”"""
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size,dropout),3)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x,lambda x : self.self_attn(x,x,x,tgt_mask))
        x = self.sublayer[1](x,lambda x : self.src_attn(x,m,m,src_mask))
        return self.sublayer[2](x,self.feed_forward)
    
    #修改自注意力子层，防止当前位置关注后续位置
def subsequent_mask(size):
    """掩盖后续位置。"""
    attn_shape = (1,size,size)
    subsequent_mask = torch.triu(torch.ones(attn_shape),diagonal=1).type(torch.uint8)#triu (上三角矩阵)
    return subsequent_mask == 0
    
def example_mask():
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
                    "Window": y,
                    "Masking": x,
                }
            )
            for y in range(20)
            for x in range(20)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect()
        .properties(height=250, width=250)
        .encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )




"""
注意力
"""
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query,key.transpose(-2,-1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==0,-1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn,value),p_attn

#一般使用8个注意力头，即h=8
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        #定义d_v总是与d_k相等
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model,d_model),4)
        self.attn = None
        self.dropout = nn.Dropout(p = dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask  = mask.unsqueeze(1)
        nbatches = query.size(0)
        #1) 从 d_model => h x d_k 批量执行所有线性投影 
        query,key, value = [
            lin(x).view(nbatches,-1,self.h,self.d_k).transpose(1,2)
            for lin,x in zip(self.linears,(query,key,value))
        ]
        
        #2）批量对所有投影向量施加注意力。
        x,self.attn = attention(
            query,key,value,mask=mask,dropout=self.dropout
        )
        # 3) 使用视图“连接”并应用最终的线性。
        x = (
            x.transpose(1,2)
            .contiguous()
            .view(nbatches,-1,self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)
"""
位置前馈网络
"""
class PositionwiseFeedForward(nn.Module):
    """FFN"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model,d_ff)
        self.w_2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

"""
嵌入和 Softmax
"""
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab,d_model)
        self.d_model = d_model
        
    def forward(self,x):
        return self.lut(x)*math.sqrt(self.d_model)

"""
位置编码
"""   
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

def example_positional():
    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [4, 5, 6, 7]
        ]
    )

    return (
        alt.Chart(data)
        .mark_line()
        .properties(width=800)
        .encode(x="position", y="embedding", color="dimension:N")
        .interactive()
    )

"""
完整模型
""" 
def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    """从超参数构建模型。"""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h,d_model)
    ff = PositionwiseFeedForward(d_model,d_ff,dropout)
    position = PositionalEncoding(d_model,dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model,c(attn),c(ff),dropout),N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab), 
    )
    #使用 Glorot / fan_avg 初始化参数。
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

"""
推理
""" 
def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(10):
        inference_test()


# --------------------------
# 第二部分 模型训练
# --------------------------
"""
批次和屏蔽
"""
class Batch:
    """用于在训练期间保存一批带有掩码的数据的对象。"""
    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:,:-1]
            self.tgt_y = tgt[:,1:]
            self.tgt_mask = self.make_std_mask(self.tgt,pad)
            self.ntokens = (self.tgt_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt,pad):
        """“创建一个掩码来隐藏填充和未来的单词。”"""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask

"""
训练循环
"""
class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed
    
def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src,batch.tgt,batch.src_mask,batch.tgt_mask
        )
        loss,loss_node = loss_compute(out,batch.tgt_y,batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none = True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()
            
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


"""
优化器
"""
def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

def example_learning_schedule():
    opts = [
        [512, 1, 4000],  # example 1
        [512, 1, 8000],  # example 2
        [256, 1, 4000],  # example 3
    ]

    dummy_model = torch.nn.Linear(1, 1)
    learning_rates = []

    # we have 3 examples in opts list.
    for idx, example in enumerate(opts):
        # run 20000 epoch for each example
        optimizer = torch.optim.Adam(
            dummy_model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        )
        lr_scheduler = LambdaLR(
            optimizer=optimizer, lr_lambda=lambda step: rate(step, *example)
        )
        tmp = []
        # take 20K dummy training steps, save the learning rate at each step
        for step in range(20000):
            tmp.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
        learning_rates.append(tmp)

    learning_rates = torch.tensor(learning_rates)

    # Enable altair to handle more than 5000 rows
    alt.data_transformers.disable_max_rows()

    opts_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Learning Rate": learning_rates[warmup_idx, :],
                    "model_size:warmup": ["512:4000", "512:8000", "256:4000"][
                        warmup_idx
                    ],
                    "step": range(20000),
                }
            )
            for warmup_idx in [0, 1, 2]
        ]
    )

    return (
        alt.Chart(opts_data)
        .mark_line()
        .properties(width=600)
        .encode(x="step", y="Learning Rate", color="model_size:warmup:N")
        .interactive()
    )

"""
正则化
"""
class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())



def loss(x, crit):
    d = x + 3 * 1
    # 将第一个元素从 0 更改为一个非常小的正数（例如 1e-9）
    epsilon = 1e-9
    predict = torch.FloatTensor([[epsilon, x / d, 1 / d, 1 / d, 1 / d]])
    # 更安全的做法是：在 log() 之前再加一次极小值保护。
    log_predict = torch.log(predict + 1e-12)
    return crit(log_predict, torch.LongTensor([1])).data

def penalization_visualization():
    crit = LabelSmoothing(5, 0, 0.1)
    loss_data = pd.DataFrame(
        {
            "Loss": [loss(x, crit) for x in range(1, 100)],
            "Steps": list(range(99)),
        }
    ).astype("float")
    print(loss_data)
    return (
        alt.Chart(loss_data)
        .mark_line()
        .properties(width=350)
        .encode(
            x="Steps",
            y="Loss",
        )
        .interactive()
    )


def show_example_mask(fn,args=[]):
    chart = fn(*args)
    if chart is None:
        print("example_mask returned None")
    else:
        out = os.path.join(os.path.dirname(__file__), "example_mask.html")
        try:
            chart.save(out)  # 保存为 HTML
            print("Chart saved to:", out)
            # 尝试在默认浏览器中打开
            import webbrowser
            webbrowser.open("file://" + os.path.abspath(out))
        except Exception as e:
            print("Failed to save/open chart:", e)
            # 备用：如果安装了 altair_viewer，可直接显示
            try:
                import altair_viewer
                altair_viewer.show(chart)
            except Exception as e2:
                print("Also failed to show via altair_viewer:", e2)




if __name__ == "__main__":
    show_example_mask(penalization_visualization)
    
    