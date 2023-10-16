import torch
import torch.nn as nn
import math

class InputEmbbedings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size =vocab_size 
        self.embeeding = nn.Embedding(vocab_size,d_model)
    
    def foward(self,x):
        return self.embeeding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self,d_model, seq_len, drop_out):
        super().__init__()
        self.d_model =d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(drop_out)

        #Create matrix 4 encoding
        pe = torch.zeros(seq_len,d_model)
        #Create a vector of shape
        position = torch.arrange(0,seq_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2)).float(0 * (-math.log(10000.0) /d_model))

        pe[:, 0::2]= torch.sin(position * div_term)
        pe[:, 1::2]= torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) # (1,seq_len, d_model)

        self.register_buffer('pe',pe)

    def forward(self, x):
        x = x+ (self.pe[:,:x.shape[1],:].requires_grad(False))
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self, eps = 10**-6) -> None:
        super().__init__()
        self.eps =eps
        self.alpha = nn.Parameter(torch.ones(1)) #multiple
        self.bias = nn.Parameter(torch.zeros(1))    # added

    def foward(self, x):
        mean = x.mean(dim= -1, keepdim= True) #get mean
        std = x.std(dim = -1, keepdim=True) # get standard dev
        return self.alpha *(x - mean) / (std + self.eps) + self.bias 

class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) ->  None:
        super().__init__()
        self.liner_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.liner_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.liner_1(x))))
    
class MultiheadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout =dropout 
        assert d_model % h == 0, "d_model is not divisiable by h" 

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # 
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None: 
            attention_score.masked_fill(mask ==0, -1e9)
        attention_score = attention_score.softmax(dim = -1) # (Batch, h, seq_len, seq_len) 

        if dropout is not None:
            attention_score = dropout(attention_score)
        return (attention_score @ value), attention_score

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key = self.w_k(k) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        value = self.w_q(v) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)

        ## (Batch, seq_len, d_model) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attetion_score = MultiheadAttention.attention(query, key, value, mask, self.dropout)

        # (batch, h, Seq_len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model) transform d_k , h to d_model
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)
    
class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiheadAttention, feed_forward_block: FeedForward, dropout: float) -> None:
        self.self_attention_block= self_attention_block
        self.feed_forward_block =feed_forward_block
        self.residual_connection = nn.ModuleList(ResidualConnection(dropout) for _ in range(2)) # make 2 input one to multiheadattention, one to normalization
    
    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module): 

    def __init__(self, self_attetion_block:MultiheadAttention, cross_attention_block:MultiheadAttention, feed_forward_block: FeedForward, dropout) -> None:
        super().__init__()
        self.self_attention_block =self_attetion_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block =feed_forward_block
        self.residual_connections = nn.Module(ResidualConnection(dropout) for _ in range(3))

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, lambda x: self.feed_forward_block)
        return x 
    
class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers =layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers: 
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.module):

    def __init__(self, d_model: int, vocab_size: int) -> None: 
        super().__init__()
        self.proj = nn.Linear(d_model: int, vocab_size: int)
    
    def forward(slef, x):
        # (Batch, Seq_len, d_model) -> (batch, seq_len,vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)
    
class Transformer( nn.Module):
    def __init__(self,encoder: Encoder, decoder: Decoder, src_embed: InputEmbbedings, tgt_embed: InputEmbbedings, /
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: property):
        
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.src_tgt = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src =  self .src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        #(batch, seq+len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.encoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
    def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int,
                          tgt_seq_len: int, d_model: int, N: int, h, int, dropout: float, d_ff: int ) -> Transformer:
        # creating embeding layer
        src_embed = InputEmbbedings(d_model, src_vocab_size)
        tgt_embed = InputEmbbedings(d_model, tgt_vocab_size)
        # creating positional encoding layer
        src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
        tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

        encoder_blocks = []
        for _ in range(N):
            eMHattention_block = MultiheadAttention(d_model,h, dropout)
            FeedForward_block =  FeedForward(d_model, d_ff, dropout)
            encoder_block = EncoderBlock(d_model, eMHattention_block, FeedForward_block, dropout)
            encoder_blocks.append(encoder_block)

        #create N decoder block 

        for _ in range(N):
            dMHattention



    


        
    









        
        


    

