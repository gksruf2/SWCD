import torch
import torch.nn as nn
import torch.nn.functional as F

from stage2_cINN.modules.modules import BasicFullyConnectedNet, ActNorm


class ConditionalFlow(nn.Module):
    """Flat version. Feeds an embedding into the flow in every block"""
    def __init__(self, in_channels, embedding_dim, hidden_dim, hidden_depth,
                 n_flows, conditioning_option="none", activation='lrelu', control=False):
        super().__init__()
        self.in_channels = in_channels      # Decoder[z_dim] = 64
        self.cond_channels = embedding_dim  # Conditioning_Model[z_dim]+30 = 94
        self.mid_channels = hidden_dim
        self.num_blocks = hidden_depth
        self.n_flows = n_flows
        self.conditioning_option = conditioning_option  # default : none

        self.sub_layers = nn.ModuleList()
        if self.conditioning_option.lower() != "none":
            self.conditioning_layers = nn.ModuleList()
        for fl in range(self.n_flows):
            mode = 'cond' if (fl % 4 != 0 and control) else 'normal'
            self.sub_layers.append(ConditionalFlatDoubleCouplingFlowBlock(
                                   self.in_channels, self.cond_channels, self.mid_channels,
                                   self.num_blocks, activation=activation, mode=mode))
            if self.conditioning_option.lower() != "none":
                #n_flow(=20)만큼의 1x1 커널
                self.conditioning_layers.append(nn.Conv2d(self.cond_channels, self.cond_channels, 1))

    def forward(self, x, embedding, reverse=False):
        """
        x shape             : (BS, 64)
        embedding shape     : (BS, 94)
        x shape (reverse)           : (BS, 64=zdim)
        embedding shape (reverse)   : [(BS, rgb, 64, 64), (BS, (x,y,z))]
        """
        hconds = list()
        hcond = embedding[:, :, None, None] # hcond shape : (BS, 94, 1, 1)
        self.last_outs = []
        self.last_logdets = []
        for i in range(self.n_flows):   # 20개의 flow에 embedding을 넣는다.
            # conditioning_option != None일 때
            if self.conditioning_option.lower() == "parallel":
                hcond = self.conditioning_layers[i](embedding[:, :, None, None])
            elif self.conditioning_option.lower() == "sequential":
                hcond = self.conditioning_layers[i](hcond)
            # conditioning_option = None일 때
            hconds.append(hcond)
        if not reverse: 
            # T(z, x_0), 현재 x :=z, embedding := (x_0+cond)
            # x shape           : (BS, 64)
            # embedding shape   : (BS, 64)
            # embedding shape (endpoint)  : (BS, 94)
            logdet = 0.0
            for i in range(self.n_flows):
                """
                가우시안 형태의 latent vector z에
                invertable한 f (flow)를 계속해서 곱해주면
                복잡한 확률분포 v를 모델링 할 수 있다.
                """
                if len(x.shape) != 4:   # x.shape == 2일 때
                    #x shape : (BS, 64, 1, 1)
                    x = x.unsqueeze(-1).unsqueeze(-1)   # x.shape = 4
                """
                sub_layers 에서 여러 가지 처리
                h, ld = self.norm_layer(h)  # activation normalization
                h, ld = self.activation(h)  # lrelu
                h, ld = self.coupling(h, xcond) # Affine coupling layers
                # h shape : (BS, 64)
                h, ld = self.shuffle(h)
                """
                # hcond shape : (BS, 94, 1, 1)
                x, logdet_ = self.sub_layers[i](x, hconds[i])
                logdet = logdet + logdet_
                self.last_outs.append(x)
                self.last_logdets.append(logdet)
                # return shape : (BS, 64), scalar
            return x.unsqueeze(-1).unsqueeze(-1), logdet
        else:
            for i in reversed(range(self.n_flows)):
                if len(x.shape) != 4:   # x.shape == 2일 때
                    # (BS, 64) -> (BS, 64, 1, 1)
                    x = x.unsqueeze(-1).unsqueeze(-1)
                # affine coupling layers (reverse)
                x = self.sub_layers[i](x, hconds[i], reverse=True)
            return x

    def reverse(self, out, xcond):
        return self(out, xcond, reverse=True)


class ConditionalDoubleVectorCouplingBlock(nn.Module):
    # cond와 x_0를 결합
    # v = f1(f2( ... ( f20(z) ...)
    # 간단한 분포인 z에 flow를 곱하여 복잡한 분포를 만든다.
    # affine coupling : 자코비안 det을 계산하기 쉽고, 역함수 계산이 쉽다.
    def __init__(self, in_channels, cond_channels, hidden_dim, depth=2, mode='normal'):
        super(ConditionalDoubleVectorCouplingBlock, self).__init__()
        # in_channels : 64
        # cond_channels : 94
        # mode :
        # normal : dim = 96, else : dim = 94
        dim = in_channels//2+cond_channels if mode == 'normal' else cond_channels
        self.s = nn.ModuleList([
            BasicFullyConnectedNet(dim=dim, depth=depth,
                                   hidden_dim=hidden_dim, use_tanh=False, use_bn=False,
                                   out_dim=in_channels//2) for _ in range(2)])
        self.t = nn.ModuleList([
            BasicFullyConnectedNet(dim=dim, depth=depth,
                                   hidden_dim=hidden_dim, use_tanh=False, use_bn=False,
                                   out_dim=in_channels//2) for _ in range(2)])
        self.mode = mode

    def forward(self, x, xc, reverse=False):
        """
        affine coupling layers
        x  : latent vector  : (BS, 64, 1, 1), latent vector v, reverse일 때 z
        xc : Embeding       : (BS, 94, 1, 1)
        """
        assert len(x.shape) == 4
        assert len(xc.shape) == 4
        x = x.squeeze(-1).squeeze(-1)   # (BS, 64)
        xc = xc.squeeze(-1).squeeze(-1) # (BS, 94)
        if not reverse:
            logdet = 0
            for i in range(len(self.s)):    # 2개 BasicFullyConnectedNet
                idx_apply, idx_keep = 0, 1
                if i % 2 != 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                
                # x = (x1, x2), x1, x2 shape : (BS, 32)
                x = torch.chunk(x, 2, dim=1)
                # mode :
                # normal    -> conditioner_input = cat(x1, cond) shape  : (BS, 96)
                # else      -> conditioner_input = cond                 : (BS, 94)
                conditioner_input = torch.cat((x[idx_apply], xc), dim=1) if self.mode == 'normal' else xc

                # batch norm, lrelu, linear transform 등을 수행하여
                # conditional input을 x_0와 섞어 fully connected layer 통과
                # scale shape : (BS, 32)
                scale = self.s[i](conditioner_input)
    
                x_ = x[idx_keep] * scale.exp() + self.t[i](conditioner_input)
                
                # x shape : (BS, 64)
                x = torch.cat((x[idx_apply], x_), dim=1)
                logdet_ = torch.sum(scale, dim=1)
                logdet = logdet + logdet_
            return x, logdet
        else:
            idx_apply, idx_keep = 0, 1
            for i in reversed(range(len(self.s))):
                if i % 2 == 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                conditioner_input = torch.cat((x[idx_apply], xc), dim=1) if self.mode == 'normal' else xc
                x_ = (x[idx_keep] - self.t[i](conditioner_input)) * self.s[i](conditioner_input).neg().exp()
                x = torch.cat((x[idx_apply], x_), dim=1)
            return x[:,:,None,None]


class ConditionalFlatDoubleCouplingFlowBlock(nn.Module):
    def __init__(self, in_channels, cond_channels, hidden_dim, hidden_depth, activation="lrelu", mode='normal'):
        super().__init__()
        __possible_activations = {"lrelu": InvLeakyRelu, "none":IgnoreLeakyRelu}
        self.norm_layer = ActNorm(in_channels, logdet=True)
        self.coupling = ConditionalDoubleVectorCouplingBlock(in_channels, cond_channels, hidden_dim, hidden_depth, mode)
        self.activation = __possible_activations[activation]()
        self.shuffle = Shuffle(in_channels)

    def forward(self, x, xcond, reverse=False):
        """
        x       : latent vector v, reverse일 때 z
        xcond   : embedding[:, :, None, None]
        """
        if not reverse:
            h = x
            logdet = 0.0
            h, ld = self.norm_layer(h)
            logdet += ld
            h, ld = self.activation(h)
            logdet += ld
            h, ld = self.coupling(h, xcond)
            logdet += ld
            h, ld = self.shuffle(h)
            logdet += ld
            return h, logdet
        else:
            h = x
            h = self.shuffle(h, reverse=True)
            h = self.coupling(h, xcond, reverse=True)
            h = self.activation(h, reverse=True)
            h = self.norm_layer(h, reverse=True)
            return h

    def reverse(self, out, xcond):
        return self.forward(out, xcond, reverse=True)


class Shuffle(nn.Module):
    # affine coupling에서 변하지 않는 레이어들을 섞어줌.
    def __init__(self, in_channels):
        super(Shuffle, self).__init__()
        self.in_channels = in_channels
        idx = torch.randperm(in_channels)
        self.register_buffer('forward_shuffle_idx', nn.Parameter(idx, requires_grad=False))
        self.register_buffer('backward_shuffle_idx', nn.Parameter(torch.argsort(idx), requires_grad=False))

    def forward(self, x, reverse=False, conditioning=None):
        if not reverse:
            return x[:, self.forward_shuffle_idx, ...], 0
        else:
            return x[:, self.backward_shuffle_idx, ...]

class IgnoreLeakyRelu(nn.Module):
    """performs identity op."""
    def __init__(self):
        super().__init__()

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        h = input
        return h, 0.0

    def reverse(self, input):
        h = input
        return h


class InvLeakyRelu(nn.Module):
    def __init__(self, alpha=0.9):
        super().__init__()
        self.alpha = alpha

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        scaling = (input >= 0).to(input) + (input < 0).to(input)*self.alpha
        h = input*scaling
        return h, 0.0

    def reverse(self, input):
        scaling = (input >= 0).to(input) + (input < 0).to(input)*self.alpha
        h = input/scaling
        return h

