import torch
import torch.nn as nn
import torch.nn.functional as F

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


class Encoder(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input):
        input = input.to(torch.float32)
        output = self.conv(input)
        return output  # (batch, hidden_size, seq_len)


class Attention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size), device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 3 * hidden_size), device=device, requires_grad=True))

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden):

        batch_size, hidden_size, _ = static_hidden.size()

        hidden = decoder_hidden.unsqueeze(2).expand_as(static_hidden)
        hidden = torch.cat((static_hidden, dynamic_hidden, hidden), 1)

        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)

        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        attns = F.softmax(attns, dim=2)  # (batch, seq_len)
        return attns


class Pointer(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(self, hidden_size, num_layers=1, dropout=0.2):
        super(Pointer, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Used to calculate probability of selecting next state
        self.v = nn.Parameter(torch.zeros((1, 4, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size),
                                          device=device, requires_grad=True))

        # Used to compute a representation of the current decoder output
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.encoder_attn = Attention(hidden_size)

        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden, last_hh):

        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)
        rnn_out = rnn_out.squeeze(1)

        # Always apply dropout on the RNN output
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            last_hh = self.drop_hh(last_hh) 

        # Given a summary of the output, find an  input context
        enc_attn = self.encoder_attn(static_hidden, dynamic_hidden, rnn_out) # [b,1,s]
        context = enc_attn.bmm(static_hidden.permute(0, 2, 1))  # [b,s,h]-[b,1,h]

        # Calculate the next output using Batch-matrix-multiply ops
        context = context.transpose(1, 2).expand_as(static_hidden) #[b,h,s]
        energy = torch.cat((static_hidden, context), dim=1)  # [b,2*h,s]

        v = self.v.expand(static_hidden.size(0), -1, -1) # [b,4,h]
        W = self.W.expand(static_hidden.size(0), -1, -1) # [b,h,2*h]
        
        # w:[b,h,2*h],energy:[b,2*h,s]=[b,h,s]
        # v:[b,4,h],相似度：[b,h,s] = [b,4,s]
        # 相似度：[b,s,h] v:[b,h,4] = [b,s,4]
        a = torch.tanh(torch.bmm(W, energy)).transpose(1, 2)
        b = v.transpose(1, 2)
        probs = torch.bmm(a, b).view(a.size(0),-1) # [b,s*4]
        probs =  probs[:,3:]
        
        return probs, last_hh

if __name__ == '__main__':
    raise Exception('Cannot be called from main')
