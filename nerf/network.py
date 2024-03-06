import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_bg="hashgrid",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 bound=1,
                 device="cpu",
                 num_levels=16,
                 lipshitz_color=False,
                 lipshitz_sigma=False,
                 freq_reg_mask_color=False,
                 freq_reg_mask_sigma=False,
                        **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        self.lipshitz_color = lipshitz_color
        self.lipshitz_sigma = lipshitz_sigma
        self.freq_reg_mask_color = freq_reg_mask_color
        self.freq_reg_mask_sigma = freq_reg_mask_sigma

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound, device=device, num_levels=num_levels)

        enc_net = [nn.Linear(self.in_dim, self.in_dim, bias=False, device=device)]
        self.enc_net = nn.ModuleList(enc_net)

        if not self.lipshitz_sigma:
            sigma_net = []
            for l in range(num_layers):
                if l == 0:
                    in_dim = self.in_dim
                else:
                    in_dim = hidden_dim

                if l == num_layers - 1:
                    out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
                else:
                    out_dim = hidden_dim

                sigma_net.append(nn.Linear(in_dim, out_dim, bias=False, device=device))

            self.sigma_net = nn.ModuleList(sigma_net)
        else:
            self.sigma_net = LipshitzMLP(self.in_dim, [hidden_dim, 1 + self.geo_feat_dim], last_layer_linear=True)


        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)

        if not self.lipshitz_color:
            color_net =  []
            for l in range(num_layers_color):
                if l == 0:
                    in_dim = self.in_dim_dir + self.geo_feat_dim
                else:
                    in_dim = hidden_dim_color

                if l == num_layers_color - 1:
                    out_dim = 3 # 3 rgb
                else:
                    out_dim = hidden_dim_color

                color_net.append(nn.Linear(in_dim, out_dim, bias=False, device=device))

            self.color_net = nn.ModuleList(color_net)
        else:
            self.color_net = LipshitzMLP(self.in_dim_dir + self.geo_feat_dim, [hidden_dim_color, hidden_dim_color, 3], last_layer_linear=True)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg        
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19, desired_resolution=2048) # much smaller hashgrid 
            
            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg
                
                if l == num_layers_bg - 1:
                    out_dim = 3 # 3 rgb
                else:
                    out_dim = hidden_dim_bg
                
                bg_net.append(nn.Linear(in_dim, out_dim, bias=False, device=device))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None


    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = self.encoder(x, bound=self.bound)

        h = x

        if self.lipshitz_sigma:
            h = self.sigma_net(h)
        else:
            for l in range(self.num_layers):
                h = self.sigma_net[l](h)
                if l != self.num_layers - 1:
                    h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)

        if self.lipshitz_color:
            h = self.color_net(h)
        else:
            for l in range(self.num_layers_color):
                h = self.color_net[l](h)
                if l != self.num_layers_color - 1:
                    h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return sigma, color

    def density(self, x, perturb_feat=None, current_iter=None, total_iter=None, start_ptr=1):
        # x: [N, 3], in [-bound, bound]

        x = self.encoder(x, bound=self.bound)

        # apply an initial linear dense layer and activation function
        #x = self.enc_net[0](x)
        #x = F.relu(x, inplace=True)

        if self.freq_reg_mask_sigma and current_iter is not None and total_iter is not None:
            mask = self.get_freq_mask(x.shape, current_iter, total_iter, start_ptr)
            x = x * mask

        h = x
        if self.lipshitz_sigma:
            h = self.sigma_net(h)
        else:
            for l in range(self.num_layers):
                h = self.sigma_net[l](h)
                if l != self.num_layers - 1:
                    h = F.relu(h, inplace=True)

        # Adv perturb on features
        if perturb_feat is not None:
            perturb_feat = torch.reshape(perturb_feat, [-1, perturb_feat.shape[-1]])
            h[:perturb_feat.shape[0]] = h[:perturb_feat.shape[0]] + perturb_feat

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]



        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    def background(self, x, d):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x) # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, current_iter=None, total_iter=None, start_ptr=1, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = self.encoder_dir(d)
        if self.freq_reg_mask_color and current_iter is not None and total_iter is not None:
            mask2 = self.get_freq_mask(d.shape, current_iter, total_iter, start_ptr)
            d = d * mask2
        h = torch.cat([d, geo_feat], dim=-1)

        if self.lipshitz_color:
            h = self.color_net(h)
        else:
            for l in range(self.num_layers_color):
                h = self.color_net[l](h)
                if l != self.num_layers_color - 1:
                    h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs        

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr}, 
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params

    def get_freq_mask(self, embedded_shape, current_iter, total_iter, start_ptr):
        if len(embedded_shape) == 2:
            embedded_shape = embedded_shape[1]
        if current_iter > total_iter:
            return torch.ones(embedded_shape).cuda().float()
        else:
            alpha_t = np.zeros(embedded_shape)
            ptr = embedded_shape / 3 * current_iter / total_iter + start_ptr #era 1 3 5 7 ( -,+,++,+++)
            ptr = ptr if ptr < embedded_shape / 3 else embedded_shape / 3
            int_ptr = int(ptr)
            alpha_t[: int_ptr * 3] = 1.0
            alpha_t[int_ptr * 3: int_ptr * 3 + 3] = (ptr - int_ptr)
            return torch.from_numpy(np.clip(np.array(alpha_t), 1e-8, 1 - 1e-8)).cuda().float()


# from https://arxiv.org/pdf/2202.08345.pdf
class LipshitzMLP(torch.nn.Module):

    def __init__(self, in_channels, nr_out_channels_per_layer, last_layer_linear):
        super(LipshitzMLP, self).__init__()

        self.last_layer_linear = last_layer_linear

        self.layers = torch.nn.ModuleList()
        # self.layers=[]
        for i in range(len(nr_out_channels_per_layer)):
            cur_out_channels = nr_out_channels_per_layer[i]
            self.layers.append(torch.nn.Linear(in_channels, cur_out_channels))
            in_channels = cur_out_channels
        apply_weight_init_fn(self, leaky_relu_init, negative_slope=0.0)
        if last_layer_linear:
            leaky_relu_init(self.layers[-1], negative_slope=1.0)

        # we make each weight separately because we want to add the normalize to it
        self.weights_per_layer = torch.nn.ParameterList()
        self.biases_per_layer = torch.nn.ParameterList()
        for i in range(len(self.layers)):
            self.weights_per_layer.append(self.layers[i].weight)
            self.biases_per_layer.append(self.layers[i].bias)

        self.lipshitz_bound_per_layer = torch.nn.ParameterList()
        for i in range(len(self.layers)):
            max_w = torch.max(torch.sum(torch.abs(self.weights_per_layer[i]), dim=1))
            # we actually make the initial value quite large because we don't want at the beggining to hinder the rgb model in any way. A large c means that the scale will be 1
            c = torch.nn.Parameter(torch.ones((1)) * max_w * 2)
            self.lipshitz_bound_per_layer.append(c)

        self.weights_initialized = True  # so that apply_weight_init_fn doesnt initialize anything

    def normalization(self, w, softplus_ci):
        absrowsum = torch.sum(torch.abs(w), dim=1)
        # scale = torch.minimum(torch.tensor(1.0), softplus_ci/absrowsum)
        # this is faster than the previous line since we don't constantly recreate a torch.tensor(1.0)
        scale = softplus_ci / absrowsum
        scale = torch.clamp(scale, max=1.0)
        return w * scale[:, None]

    def lipshitz_bound_full(self):
        lipshitz_full = 1
        for i in range(len(self.layers)):
            lipshitz_full = lipshitz_full * torch.nn.functional.softplus(self.lipshitz_bound_per_layer[i])

        return lipshitz_full

    def forward(self, x):

        # x=self.mlp(x)

        for i in range(len(self.layers)):
            weight = self.weights_per_layer[i]
            bias = self.biases_per_layer[i]

            weight = self.normalization(weight, torch.nn.functional.softplus(self.lipshitz_bound_per_layer[i]))

            x = torch.nn.functional.linear(x, weight, bias)

            is_last_layer = i == (len(self.layers) - 1)

            if is_last_layer and self.last_layer_linear:
                pass
            else:
                x = torch.nn.functional.gelu(x)

        return x


def leaky_relu_init(m, negative_slope=0.2):
    gain = np.sqrt(2.0 / (1.0 + negative_slope ** 2))

    if isinstance(m, torch.nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // 2
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // 4
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] // 8
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * np.sqrt(2.0 / (n1 + n2))
    else:
        return

    m.weight.data.uniform_(-std * np.sqrt(3.0), std * np.sqrt(3.0))
    if m.bias is not None:
        m.bias.data.zero_()

    if isinstance(m, torch.nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    # m.weights_initialized=True


def apply_weight_init_fn(m, fn, negative_slope=1.0):
    should_initialize_weight = True
    if not hasattr(m, "weights_initialized"):  # if we don't have this then we need to intiialzie
        # fn(m, is_linear, scale)
        should_initialize_weight = True
    elif m.weights_initialized == False:  # if we have it but it's set to false
        # fn(m, is_linear, scale)
        should_initialize_weight = True
    else:
        print("skipping weight init on ", m)
        should_initialize_weight = False

    if should_initialize_weight:
        # fn(m, is_linear, scale)
        fn(m, negative_slope)
        # m.weights_initialized=True
        for module in m.children():
            apply_weight_init_fn(module, fn, negative_slope)

