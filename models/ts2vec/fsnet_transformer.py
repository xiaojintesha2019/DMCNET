import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pdb

from itertools import chain


def normalize(W):
    W_norm = torch.norm(W)
    W_norm = torch.relu(W_norm - 1) + 1
    W = W / W_norm
    return W


class SamePadTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1, gamma=0.9):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        self.padding = self.receptive_field // 2
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]), requires_grad=True)
        self.dilation = dilation
        self.kernel_size = kernel_size

        self.grad_dim, self.shape = [], []
        self.dim = in_channels * out_channels
        self.in_channels = in_channels
        self.out_features = out_channels

        self.n_chunks = in_channels
        self.chunk_in_d = self.dim // self.n_chunks
        self.chunk_out_d = int(in_channels * kernel_size // self.n_chunks)

        self.grads = torch.Tensor(self.dim).fill_(0).cuda()
        self.f_grads = torch.Tensor(self.dim).fill_(0).cuda()
        nh = 64
        self.controller = nn.Sequential(nn.Linear(self.chunk_in_d, nh), nn.SiLU())
        self.calib_w = nn.Linear(nh, self.chunk_out_d)
        self.calib_b = nn.Linear(nh, out_channels // in_channels)
        self.calib_f = nn.Linear(nh, out_channels // in_channels)
        dim = self.n_chunks * (self.chunk_out_d + 2 * out_channels // in_channels)
        self.W = nn.Parameter(torch.empty(dim, 32), requires_grad=False)
        nn.init.xavier_uniform_(self.W.data)
        self.W.data = F.normalize(self.W.data)

        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        self.gamma = gamma
        self.f_gamma = 0.3
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.trigger = 0
        self.tau = 0.75

        # Transformer components
        self.self_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=8)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, 4 * in_channels),
            nn.ReLU(),
            nn.Linear(4 * in_channels, in_channels)
        )
        self.layer_norm1 = nn.LayerNorm(in_channels)
        self.layer_norm2 = nn.LayerNorm(in_channels)

    def ctrl_params(self):
        c_iter = chain(self.controller.parameters(), self.calib_w.parameters(),
                       self.calib_b.parameters(), self.calib_f.parameters())
        for p in c_iter:
            yield p

    def store_grad(self):
        grad = self.self_attn.in_proj_weight.grad.data.clone()
        grad = nn.functional.normalize(grad)
        grad = grad.view(-1)
        self.f_grads = self.f_gamma * self.f_grads + (1 - self.f_gamma) * grad
        if not self.training:
            e = self.cos(self.f_grads, self.grads)
            if e < -self.tau:
                self.trigger = 1
        self.grads = self.gamma * self.grads + (1 - self.gamma) * grad

    def fw_chunks(self):
        x = self.grads.view(self.n_chunks, -1)
        rep = self.controller(x)
        w = self.calib_w(rep)
        b = self.calib_b(rep)
        f = self.calib_f(rep)
        q = torch.cat([w.view(-1), b.view(-1), f.view(-1)])
        if not hasattr(self, 'q_ema'):
            setattr(self, 'q_ema', torch.zeros(*q.size()).float().cuda())
        else:
            self.q_ema = self.f_gamma * self.q_ema + (1 - self.f_gamma) * q
            q = self.q_ema
        if self.trigger == 1:
            dim = w.size(0)
            self.trigger = 0
            att = q @ self.W
            att = F.softmax(att / 0.5)
            v, idx = torch.topk(att, 2)
            ww = torch.index_select(self.W, 1, idx)
            idx = idx.unsqueeze(1).float()
            old_w = ww @ idx
            s_att = torch.zeros(att.size(0)).cuda()
            s_att[idx.squeeze().long()] = v.squeeze()
            W = old_w @ s_att.unsqueeze(0)
            mask = torch.ones(W.size()).cuda()
            mask[:, idx.squeeze().long()] = self.tau
            self.W.data = mask * self.W.data + (1 - mask) * W
            self.W.data = F.normalize(self.W.data)
            ll = torch.split(old_w, dim)
            nw, nb, nf = w.size(1), b.size(1), f.size(1)
            o_w, o_b, o_f = torch.cat(*[ll[:nw]]), torch.cat(*[ll[nw:nw + nb]]), torch.cat(*[ll[-nf:]])
            try:
                w = self.tau * w + (1 - self.tau) * o_w.view(w.size())
                b = self.tau * b + (1 - self.tau) * o_b.view(b.size())
                f = self.tau * f + (1 - self.tau) * o_f.view(f.size())
            except:
                pdb.set_trace()
        f = f.view(-1).unsqueeze(0).unsqueeze(2)
        return w.unsqueeze(0), b.view(-1), f

    def forward(self, x):
        w, b, f = self.fw_chunks()
        x = x.permute(2, 0, 1)  # (batch, channels, seq_len) -> (seq_len, batch, channels)
        attn_output, _ = self.self_attn(x, x, x)
        x = x + attn_output
        x = self.layer_norm1(x)
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.layer_norm2(x)
        x = x.permute(1, 2, 0)  # (seq_len, batch, channels) -> (batch, channels, seq_len)
        out= f * x
        print(x.size)
        return out

    def representation(self, x):
        x = x.permute(2, 0, 1)
        attn_output, _ = self.self_attn(x, x, x)
        x = x + attn_output
        x = self.layer_norm1(x)
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.layer_norm2(x)
        x = x.permute(1, 2, 0)
        return x

    def _forward(self, x):
        return self.forward(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False, gamma=0.9):
        super().__init__()
        self.conv1 = SamePadTransformer(in_channels, out_channels, kernel_size, dilation=dilation, gamma=gamma)
        self.conv2 = SamePadTransformer(out_channels, out_channels, kernel_size, dilation=dilation, gamma=gamma)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None

    def ctrl_params(self):
        c_iter = chain(self.conv1.controller.parameters(), self.conv1.calib_w.parameters(),
                       self.conv1.calib_b.parameters(), self.conv1.calib_f.parameters(),
                       self.conv2.controller.parameters(), self.conv2.calib_w.parameters(),
                       self.conv2.calib_b.parameters(), self.conv2.calib_f.parameters())

        return c_iter

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, gamma=0.9):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i - 1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2 ** i,
                final=(i == len(channels) - 1), gamma=gamma
            )
            for i in range(len(channels))
        ])

    def ctrl_params(self):
        ctrl = []
        for l in self.net:
            ctrl.append(l.ctrl_params())
        c = chain(*ctrl)
        for p in c:
            yield p

    def forward(self, x):
        return self.net(x)
