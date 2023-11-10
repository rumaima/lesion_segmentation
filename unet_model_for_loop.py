from torch.nn import Module, Sequential
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool1d, Dropout3d
from torch.nn import ReLU, Sigmoid
import torch
import time
from torch.autograd import Variable
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np
import itertools

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers=1, hidden_dim=1024):
        super(MLP, self).__init__()
        model = []
        model += [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers):
            model += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        model += [nn.Linear(hidden_dim, output_dim)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class UNet(Module):
    # __                            __
    #  1|__   ________________   __|1
    #     2|__  ____________  __|2
    #        3|__  ______  __|3
    #           4|__ __ __|4
    # The convolution operations on either side are residual subject to 1*1 Convolution for channel homogeneity

    def __init__(self, args, num_channels=1, feat_channels=[64, 128, 256, 512, 1024], residual='conv',domain=None,is_target=True, is_val=False):
        # residual: conv for residual input x through 1*1 conv across every layer for downsampling, None for removal of residuals

        super(UNet, self).__init__()
        
        self.args = args
        self.classifier = iVAE(self.args, backbone_net=None)
        self.z_dim = args.z_dim

        # Encoder downsamplers
        self.pool1 = MaxPool3d((2, 2, 2))
        self.pool2 = MaxPool3d((2, 2, 2))
        self.pool3 = MaxPool3d((2, 2, 2))
        self.pool4 = MaxPool3d((2, 2, 2))

        # Encoder convolutions

        self.conv_blk1 = Conv3D_Block(num_channels, feat_channels[0], residual=residual)
        self.conv_blk2 = Conv3D_Block(feat_channels[0], feat_channels[1], residual=residual)
        self.conv_blk3 = Conv3D_Block(feat_channels[1], feat_channels[2], residual=residual)
        self.conv_blk4 = Conv3D_Block(feat_channels[2], feat_channels[3], residual=residual)
        self.conv_blk5 = Conv3D_Block(feat_channels[3], feat_channels[4], residual=residual)
        # self.conv_blk6 = Conv3D_Block(feat_channels[4], feat_channels[4], residual=residual)


        # # Decoder convolutions
        # self.dec_conv_blk5 = Conv3D_Block(2 * feat_channels[3], feat_channels[3], residual=residual)
        self.dec_conv_blk4 = Conv3D_Block(2 * feat_channels[3], feat_channels[3], residual=residual)
        self.dec_conv_blk3 = Conv3D_Block(2 * feat_channels[2], feat_channels[2], residual=residual)
        self.dec_conv_blk2 = Conv3D_Block(2 * feat_channels[1], feat_channels[1], residual=residual)
        self.dec_conv_blk1 = Conv3D_Block(2 * feat_channels[0], feat_channels[0], residual=residual)

        # Decoder upsamplers
        # self.deconv_blk5 = Deconv3D_Block(feat_channels[4], feat_channels[4])
        self.deconv_blk4 = Deconv3D_Block(feat_channels[4], feat_channels[3])
        self.deconv_blk3 = Deconv3D_Block(feat_channels[3], feat_channels[2])
        self.deconv_blk2 = Deconv3D_Block(feat_channels[2], feat_channels[1])
        self.deconv_blk1 = Deconv3D_Block(feat_channels[1], feat_channels[0])


        # Final 1*1 Conv Segmentation map
        self.one_conv = Conv3d(feat_channels[0], num_channels, kernel_size=1, stride=1, padding=0, bias=True)

        # Activation function
        self.sigmoid = Sigmoid()

        # define all the encoder 

    def decode(self, z):
        out = self.classifier.decoder(z)
        return out

    def forward(self, input, flag_decode=False):
        if flag_decode:
            return self.decode(input)
        # Encoder part
        x, domain, is_target, is_val = input
        
        x1 = self.conv_blk1(x)

        x_low1 = self.pool1(x1)
        x2 = self.conv_blk2(x_low1)

        x_low2 = self.pool2(x2)
        x3 = self.conv_blk3(x_low2)

        x_low3 = self.pool3(x3)
        x4 = self.conv_blk4(x_low3)

        x_low4 = self.pool4(x4)
        base = self.conv_blk5(x_low4)
  
        updated_base, kl = self.classifier.disentangle(x=base, u=domain, z_dim = self.z_dim, track_bn=is_target)

        d4 = torch.cat([self.deconv_blk4(updated_base), x4], dim=1)
        d_high4 = self.dec_conv_blk4(d4)

        d3 = torch.cat([self.deconv_blk3(d_high4), x3], dim=1)
        d_high3 = self.dec_conv_blk3(d3)
        d_high3 = Dropout3d(p=0.5)(d_high3)

        d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
        d_high2 = self.dec_conv_blk2(d2)
        d_high2 = Dropout3d(p=0.5)(d_high2)
   
        d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
        d_high1 = self.dec_conv_blk1(d1)

        seg = self.one_conv(d_high1)
        seg_sigmoid = self.sigmoid(self.one_conv(d_high1))
        

        return seg_sigmoid, seg, kl
        # return seg_sigmoid, seg


class Conv3D_Block(Module):

    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=None):

        super(Conv3D_Block, self).__init__()

        self.conv1 = Sequential(
            Conv3d(inp_feat, out_feat, kernel_size=kernel,
                   stride=stride, padding=padding, bias=True),
            BatchNorm3d(out_feat),
            ReLU())

        self.conv2 = Sequential(
            Conv3d(out_feat, out_feat, kernel_size=kernel,
                   stride=stride, padding=padding, bias=True),
            BatchNorm3d(out_feat),
            ReLU())

        self.residual = residual

        if self.residual is not None:
            self.residual_upsampler = Conv3d(inp_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):

        res = x

        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)


class Deconv3D_Block(Module):

    def __init__(self, inp_feat, out_feat, kernel=3, stride=2, padding=1):
        super(Deconv3D_Block, self).__init__()

        self.deconv = Sequential(
            ConvTranspose3d(inp_feat, out_feat, kernel_size=(kernel, kernel, kernel),
                            stride=(stride, stride, stride), padding=(padding, padding, padding), output_padding=1, bias=True),
            ReLU())

    def forward(self, x):
        return self.deconv(x)


class ChannelPool3d(AvgPool1d):

    def __init__(self, kernel_size, stride, padding):
        super(ChannelPool3d, self).__init__(kernel_size, stride, padding)
        self.pool_1d = AvgPool1d(self.kernel_size, self.stride, self.padding, self.ceil_mode)

    def forward(self, inp):
        n, c, d, w, h = inp.size()
        inp = inp.view(n, c, d * w * h).permute(0, 2, 1)
        pooled = self.pool_1d(inp)
        c = int(c / self.kernel_size[0])
        return inp.view(n, c, d, w, h)


class iVAE(nn.Module):
    def __init__(self, args, backbone_net=None):
        super(iVAE, self).__init__()
        self.args = args
        self.backbone_net = backbone_net
        self.pool_layer = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1,1)), nn.Flatten())

        # latent space: [0:self.c_dim] [self.c_dim:self_z_dim]
        self.z_dim = args.z_dim
        self.s_dim = args.s_dim
        self.c_dim = self.z_dim - self.s_dim

        flow_dim = args.flow_dim
        flow_nlayer = args.flow_nlayer
        flow = args.flow
        dim = args.hidden_dim
        out_features = dim

        self.encoder = nn.Sequential(nn.Linear(out_features, dim),
                                     nn.BatchNorm1d(dim),
                                     nn.ReLU(), nn.Dropout())
        self.fc_mu = nn.Sequential(nn.Linear(dim, dim))
        self.fc_logvar = nn.Sequential(nn.Linear(dim, dim))

        self.decoder = nn.Sequential(nn.Linear(self.z_dim, dim),
                                     nn.BatchNorm1d(dim),
                                     nn.ReLU(),
                                     nn.Linear(dim, out_features))

        # if args.arch == 'resnet18':
        #     self.classifier = nn.Sequential(
        #                 nn.Linear(self.z_dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(),
        #                 nn.Linear(dim, args.num_classes)
        #     )
        # else:
        #     self.classifier = nn.Sequential(nn.Linear(self.z_dim, args.num_classes))

        self.flow_type = flow
        # self.u_embedding = nn.Embedding(10, 1024)
        self.u_embedding = nn.Embedding(4, 256)
        assert flow in ['ddsf', 'dsf', 'sf', 'nsf']
        if flow == 'sf':
            self.domain_flow = SigmoidFlow(flow_dim)
        elif flow == 'dsf':
            self.domain_flow = DenseSigmoidFlow(1, flow_dim, 1)
        elif flow == 'ddsf':
            self.domain_flow = DDSF(flow_nlayer, 1, flow_dim, 1)

        if self.flow_type in ['sf', 'dsf', 'ddsf']:
            domain_num_params = self.domain_flow.num_params * self.s_dim
            self.domain_mlp = MLP(256, domain_num_params)


        self.lambda_vae = args.lambda_vae

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def track_bn_stats(self, track):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.track_running_stats = track
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = track

    def extract_feature(self, x, u, track_bn=False):
        self.track_bn_stats(track_bn)
        x = self.backbone(x, track_bn)
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_logvar(h)
        if self.training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        tilde_z, _ = self.domain_influence(z, u) 
        return tilde_z

    def disentangle(self, x, u, z_dim, track_bn=False):

        self.track_bn_stats(track_bn)

        normal_dist = torch.distributions.MultivariateNormal(torch.zeros(z_dim).cuda(), torch.eye(z_dim).cuda(), validate_args=False)
       
        b,c,h,w,d = x.shape 
        #permute b,h,w,d,c
        x = x.permute(0,2,3,4,1) 
        x_updated = torch.zeros_like(x)
        for i in range(0,h):
            for j in range(0,w):
                for k in range(0,d):
                    x_ijk = x[:,i,j,k,:]

                    mu, log_var = self.fc_mu(x_ijk), self.fc_logvar(x_ijk)

                    if self.training or self.lambda_vae != 0:
                        # z = self.reparameterize(reshaped_mu, reshaped_lvar)
                        z = self.reparameterize(mu, log_var)
                    else:
                        z = mu
                    # de-influence u
                    tilde_z, logdet_u = self.domain_influence(z, u) # remove the domain influcence; back to Gaussian  
                    x_updated[:,i,j,k,:] = tilde_z

                    # finding the KL loss
                    q_dist = torch.distributions.Normal(mu, torch.exp(torch.clamp(log_var, min=-10) / 2))
                    log_qz = q_dist.log_prob(x_ijk)
                    log_pz = normal_dist.log_prob(tilde_z)+ logdet_u
                    kl = (log_qz.sum(dim=1) - log_pz).mean()
                    
    
        x_updated = x_updated.permute(0,4,1,2,3) 
        return x_updated, kl

    def encode(self, x, u, track_bn=False):

        self.track_bn_stats(track_bn)

        # sample z
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_logvar(h)

        if self.training or self.lambda_vae != 0:
            # z = self.reparameterize(reshaped_mu, reshaped_lvar)
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        # de-influence u
        tilde_z, logdet_u = self.domain_influence(z, u) # remove the domain influcence; back to Gaussian  
        # tilde_z.shape = [2, 256]          

        #decode tilde_z to change the dimension from (B, z_dim) to (B, dim)
        decoded_tilde_z = self.decode(tilde_z)
        # decoded_tilde_z.shape = [2, 1024]
        # get logits
        # logit = self.predict(tilde_z, track_bn=track_bn)

        return z, tilde_z, decoded_tilde_z, mu, log_var, logdet_u # tilde_z is for domain adversarial, tilde_tilde_z is for KL p, z is for reconstruction and KL q. 


    def domain_influence(self, z, u):

        if self.flow_type == 'nsf':
            zcont = z[:, :-self.s_dim]
            tilde_zs = self.domain_flow(z[:, -self.s_dim:], u)

        else:
            domain_embedding = self.u_embedding(u)  # B,h_dim
            B, _ = domain_embedding.size()
            dsparams = self.domain_mlp(domain_embedding)  # B, ndim
            dsparams = dsparams.view(B, self.s_dim, -1)
            zcont = z[:,:self.c_dim]
            tilde_zs, logdet = self.domain_flow(z[:,self.c_dim:], dsparams)

        tilde_z = torch.cat([zcont, tilde_zs], 1)

        return tilde_z, logdet

    def decode(self, z):
        out = self.decoder(z)
        return out

    def forward(self, x, u, track_bn=False):
        self.track_bn_stats(track_bn)
        x = self.backbone(x)
        _, _, _, _, _, logit = self.encode(x, u=u)
        if self.training:
            raise NotImplementedError
            return tilde_z, logit
        else:
            return logit

    def backbone(self, x, track_bn=False):
        self.track_bn_stats(track_bn)
        out = self.backbone_net(x)
        if len(out.size()) > 2:
            out = self.pool_layer(out)
        return out

    def predict(self, z, track_bn=False):
        self.track_bn_stats(track_bn)
        return self.classifier(z)

    def get_parameters(self, base_lr=1.0):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        base_params = itertools.chain(self.encoder.parameters(), self.fc_mu.parameters(),
                                      self.fc_logvar.parameters(), self.decoder.parameters(),
                                      self.classifier.parameters(), 
                                      self.u_embedding.parameters(),
                                      self.domain_flow.parameters(),
                                      self.domain_mlp.parameters(),
                                      )
        params = [
            {"params": self.backbone_net.parameters(), "lr": 0.1 * base_lr},
            {"params": base_params, "lr": 1.0 * base_lr},
        ]
        return params

class BaseFlow(nn.Module):

    def sample(self, n=1, context=None, **kwargs):
        dim = self.dim
        if isinstance(self.dim, int):
            dim = [dim, ]

        spl = Variable(torch.FloatTensor(n, *dim).normal_())
        lgd = Variable(torch.from_numpy(
            np.zeros(n).astype('float32')))
        if context is None:
            context = Variable(torch.from_numpy(
                np.ones((n, self.context_dim)).astype('float32')))

        if hasattr(self, 'gpu'):
            if self.gpu:
                spl = spl.cuda()
                lgd = lgd.cuda()
                context = context.gpu()

        return self.forward((spl, lgd, context))

    def cuda(self):
        self.gpu = True
        return super(BaseFlow, self).cuda()


delta = 1e-6
softplus_ = nn.Softplus()
softplus = lambda x: softplus_(x) + delta
sigmoid_ = nn.Sigmoid()
sigmoid = lambda x: sigmoid_(x) * (1 - delta) + 0.5 * delta
sigmoid2 = lambda x: sigmoid(x) * 2.0
logsigmoid = lambda x: -softplus(-x)
log = lambda x: torch.log(x * 1e2) - np.log(1e2)
logit = lambda x: log(x) - log(1 - x)


def softmax(x, dim=-1):
    e_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
    out = e_x / e_x.sum(dim=dim, keepdim=True)
    return out


sum1 = lambda x: x.sum(1)
sum_from_one = lambda x: sum_from_one(sum1(x)) if len(x.size()) > 2 else sum1(x)

class Sigmoid(nn.Module):
    def forward(self, x):
        return sigmoid(x)

class SigmoidFlow(BaseFlow):

    def __init__(self, num_ds_dim=4):
        super(SigmoidFlow, self).__init__()
        self.num_ds_dim = num_ds_dim
        self.num_params = 3 * num_ds_dim
        self.act_a = lambda x: softplus(x)
        self.act_b = lambda x: x
        self.act_w = lambda x: softmax(x, dim=2)

    def forward(self, x, dsparams, mollify=0.0, delta=delta):
        ndim = self.num_ds_dim
        a_ = self.act_a(dsparams[:, :, 0 * ndim:1 * ndim])
        b_ = self.act_b(dsparams[:, :, 1 * ndim:2 * ndim])
        w = self.act_w(dsparams[:, :, 2 * ndim:3 * ndim])

        a = a_ * (1 - mollify) + 1.0 * mollify
        b = b_ * (1 - mollify) + 0.0 * mollify

        pre_sigm = a * x[:, :, None] + b
        sigm = torch.sigmoid(pre_sigm)
        x_pre = torch.sum(w * sigm, dim=2)
        x_pre_clipped = x_pre * (1 - delta) + delta * 0.5
        x_ = log(x_pre_clipped) - log(1 - x_pre_clipped)
        xnew = x_

        return xnew.squeeze(-1)

class DenseSigmoidFlow(BaseFlow):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super(DenseSigmoidFlow, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.act_a = lambda x: softplus_(x)
        self.act_b = lambda x: x
        self.act_w = lambda x: softmax(x, dim=3)
        self.act_u = lambda x: softmax(x, dim=3)

        self.u_ = Parameter(torch.Tensor(hidden_dim, in_dim))
        self.w_ = Parameter(torch.Tensor(out_dim, hidden_dim))
        self.num_params = 3 * hidden_dim + in_dim
        self.reset_parameters()

    def reset_parameters(self):
        self.u_.data.uniform_(-0.001, 0.001)
        self.w_.data.uniform_(-0.001, 0.001)

    def forward(self, x, dsparams, logdet=None):
        if len(x.size()) == 2:
            x = x.unsqueeze(-1)
        inv = np.log(np.exp(1 - delta) - 1)
        ndim = self.hidden_dim
        pre_u = self.u_[None, None, :, :] + dsparams[:, :, -self.in_dim:][:, :, None, :]        #pre_u.shape = 1: [16, 4, 16, 1], 2:[16, 4, 16, 16]
        pre_w = self.w_[None, None, :, :] + dsparams[:, :, 2 * ndim:3 * ndim][:, :, None, :]    #pre_w.shape = 1: [16, 4, 16, 16], 2: [16, 4, 1, 16]
        a = self.act_a(dsparams[:, :, 0 * ndim:1 * ndim] + inv)
        b = self.act_b(dsparams[:, :, 1 * ndim:2 * ndim])
        w = self.act_w(pre_w)
        u = self.act_u(pre_u)

        pre_sigm = torch.sum(u * a[:, :, :, None] * x[:, :, None, :], 3) + b   # pre_sigm.shape = 1: [16,4,16], 2: [16, 4, 16]
        sigm = torch.sigmoid(pre_sigm)                                         # sigm.shape = 1: [16,4,16], 2: [16,4,16]
        x_pre = torch.sum(w * sigm[:, :, None, :], dim=3)                      # x_pre.shape = 1: [16,4,16], 2: [16, 4, 1]

        x_pre_clipped = x_pre * (1 - delta) + delta * 0.5                      # x_pre_clipped.shape = 1: [16,4,16], 2: [16, 4, 1]
        x_ = log(x_pre_clipped) - log(1 - x_pre_clipped)                       # x_.shape = 1: [16,4,16], 2: [16, 4, 1]
        xnew = x_                                                              # xnew.shape =  1: [16,4,16], 2: [16, 4, 1]

        logj = F.log_softmax(pre_w, dim=3) + \
            logsigmoid(pre_sigm[:,:,None,:]) + \
            logsigmoid(-pre_sigm[:,:,None,:]) + log(a[:,:,None,:])             
        # n, d, d2, dh
        
        logj = logj[:,:,:,:,None] + F.log_softmax(pre_u, dim=3)[:,:,None,:,:]
        # n, d, d2, dh, d1
        
        logj = log_sum_exp(logj,3).sum(3)                                    # logj.shape = 1: [16, 4, 16, 1], 2: [16, 4, 1, 16]]
        # n, d, d2, d1
        
        logdet_ = logj + np.log(1-delta) - \
            (log(x_pre_clipped) + log(-x_pre_clipped+1))[:,:,:,None]
        
        if logdet is None:
            logdet = logdet_.new_zeros(logdet_.shape[0], logdet_.shape[1], 1, 1)
        
        logdet = log_sum_exp(
            logdet_[:,:,:,:,None] + logdet[:,:,None,:,:], 3
        ).sum(3)
        # n, d, d2, d1, d0 -> n, d, d2, d0                              # logdet.shape = 1: [16, 4, 16, 1], 2: [16, 4, 1, 1]
        return xnew.squeeze(-1), logdet 
        

    def extra_repr(self):
        return 'input_dim={in_dim}, output_dim={out_dim}'.format(**self.__dict__)

class DDSF(nn.Module):
    def __init__(self, n_layers, in_dim, hidden_dim, out_dim):
        super(DDSF, self).__init__()
        blocks = [DenseSigmoidFlow(in_dim, hidden_dim, hidden_dim)]
        for _ in range(n_layers - 2):
            blocks += [DenseSigmoidFlow(hidden_dim, hidden_dim, hidden_dim)]
        blocks += [DenseSigmoidFlow(hidden_dim, hidden_dim, out_dim)]
        self.num_params = 0
        for block in blocks:
            self.num_params += block.num_params
        self.model = nn.ModuleList(blocks)

    def forward(self, x, dsparams):
        start = 0
        _logdet = None
        x = x.view(-1,x.shape[-1])
        
        for block in self.model:
            block_dsparams = dsparams[:, :, start:start + block.num_params]
            x, _logdet = block(x, block_dsparams, logdet=_logdet)
            start = start + block.num_params

        logdet = _logdet[:,:,0,0].sum(1)
        
        return x, logdet


def oper(array,oper,axis=-1,keepdims=False):
    a_oper = oper(array)
    if keepdims:
        shape = []
        for j,s in enumerate(array.size()):
            shape.append(s)
        shape[axis] = -1
        a_oper = a_oper.view(*shape)
    return a_oper

def log_sum_exp(A, axis=-1, sum_op=torch.sum):    
    maximum = lambda x: x.max(axis)[0]    
    A_max = oper(A,maximum, axis, True)
    summation = lambda x: sum_op(torch.exp(x-A_max), axis)
    B = torch.log(oper(A,summation,axis,True)) + A_max    
    return B


