import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class GNNBaseModel(nn.Module):
    def __init__(self, config, d_model, device):
        super().__init__()
        self.config = config
        self.device = device
        self.d_model = d_model
        self.num_classes = config["num_classes"]
        self.strides = [tuple(config["strides"])] * config["num_layers"]
        self.z_dim = config["z_dim"]
        self.gf_dim = config["gf_dim"]
        self.df_dim = config["df_dim"]
        self.height_d, self.width_d = self.get_dimentions_factors(self.strides)
        self.c_h = int(1 / self.height_d)
        self.c_w = int(self.d_model / self.width_d)
        self.kernel = (config["kernel_height"], config["kernel_width"])
        self.kernel = self.get_kernel(self.c_h, self.c_w, self.kernel)
        self.padding_dim = self.get_padding_dim(self.kernel, config["dilation_rate"])

    def get_dimentions_factors(self, strides):
        """
        Method calculates how much height and width will be increased/decreased basen on given stride schedule
        Args:
            strides_schedule: A list of stride tuples(heigh, width)

        Returns:
        Two numbers that tells how many times height and width will be increased/decreased
        """
        width_d = 1
        height_d = 1
        for s in strides:
            width_d = width_d * s[1]
            height_d = height_d * s[0]
        return height_d, width_d
    
    def get_kernel(self, c_h, c_w, kernel):
        """
        Calculates the kernel size given the input. Kernel size is changed only if the input dimentions are smaller
        than kernel

        Args:
        x: The input vector.
        kernel:  The height and width of the convolution kernel filter

        Returns:
        The height and width of new convolution kernel filter
        """
        height = kernel[0]
        width = kernel[1]
        if c_h < height:
            height = c_h
        elif c_w < width:
            width = c_w

        return (height, width)
    
    def get_padding_dim(self, kernel, dilations):
        """
        Calculates required padding for given axis
        Args:
            kernel: A tuple of kernel height and width
            dilations: A tuple of dilation height and width
            axis: 0 - height, 1 width

        Returns:
            An array that contains a length of padding at the begging and at the end
        """
        extra_padding_height = (kernel[0] - 1) * (dilations)
        extra_padding_width = (kernel[1] - 1) * (dilations)

        return [extra_padding_width // 2, extra_padding_width - (extra_padding_width // 2), 
            extra_padding_height // 2, extra_padding_height - (extra_padding_height // 2)]

    def init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def hw_flatten(self, x):
        return torch.reshape(x, [x.size(dim=0), x.size(dim=-1), -1])
                
    def forward(self, input):
        pass


class Generator(GNNBaseModel):
    def __init__(self, config, d_model, device):
        super().__init__(config, d_model, device)
        self.hidden_dim = self.gf_dim * (2 ** (len(self.strides)-1))
        self.input_dim = self.hidden_dim
        self.reshape_dim = [-1, self.hidden_dim, self.c_h, self.c_w]
        self.snlinear = spectral_norm(nn.Linear(self.z_dim, self.c_h * self.c_w * self.hidden_dim))
        self.pad = torch.nn.ReflectionPad2d(self.padding_dim)
        for layer_id in range(len(self.strides)):
            if layer_id == len(self.strides) - 2:
                self.attention1 =  spectral_norm(nn.Conv2d(self.hidden_dim, int(math.sqrt(self.hidden_dim)), 1))
                self.attention2 = spectral_norm(nn.Conv2d(self.hidden_dim, int(math.sqrt(self.hidden_dim)), 1))
                self.attention3 = spectral_norm(nn.Conv2d(self.hidden_dim, self.hidden_dim, 1))
                self.softmax = nn.Softmax(dim=-1)
            exec(f'self.upsampling{layer_id} = nn.UpsamplingNearest2d(scale_factor = self.strides[layer_id])')
            exec(f'self.conv1{layer_id} = spectral_norm(nn.Conv2d(self.input_dim, self.hidden_dim, self.kernel, padding="valid"))')
            exec(f'self.conv2{layer_id} = spectral_norm(nn.Conv2d(self.hidden_dim, self.hidden_dim, self.kernel, padding="valid"))')
            self.input_dim = self.hidden_dim
            self.hidden_dim = self.hidden_dim // self.strides[layer_id][1]
        self.bnorm = nn.BatchNorm2d(self.input_dim)
        self.final_conv = spectral_norm(nn.Conv2d(self.input_dim, self.num_classes, (1,1)))
        
        self.apply(self.init_weights)
    
    def forward(self, input):
        z = self.snlinear(input)
        z = torch.reshape(z, self.reshape_dim)
        
        for layer_id in range(len(self.strides)):
            z_0 = z
            z = F.leaky_relu(z)
            z = eval(f'self.upsampling{layer_id}(z)')
            z = self.pad(z)
            z = eval(f'self.conv1{layer_id}(z)')
            z = F.leaky_relu(z)
            z = self.pad(z)
            z = eval(f'self.conv2{layer_id}(z)')
            z_0 = eval(f'self.upsampling{layer_id}(z_0)')
            z_0 = self.pad(z_0)
            z_0 = eval(f'self.conv1{layer_id}(z_0)')
            z = z_0 + z

            if layer_id == len(self.strides) - 2:
                s = torch.matmul(self.hw_flatten(self.attention2(z)), torch.transpose(self.hw_flatten(self.attention1(z)), 2, 1))
                beta = self.softmax(s)
                o = torch.matmul(beta, self.hw_flatten(self.attention3(z)))
                attention_multiplier = torch.zeros(1, 1).to(self.device)
                o = torch.reshape(o, z.size())
                z = attention_multiplier * o + z

        z = F.leaky_relu(self.bnorm(z))
        z = torch.tanh(self.final_conv(z))
        return z


class Discriminator(GNNBaseModel):
    def __init__(self, config, d_model, device):
        super().__init__(config, d_model, device)
        self.hidden_dim = self.df_dim
        self.input_dim = self.num_classes
        self.snlinear = spectral_norm(nn.Linear(self.c_w, 1))
        self.relu = nn.ReLU()
        self.pad = torch.nn.ReflectionPad2d(self.padding_dim)
        for layer_id in range(len(self.strides)):
            self.hidden_dim = self.hidden_dim * self.strides[layer_id][0]
            if layer_id == 1:
                self.attention1 =  spectral_norm(nn.Conv2d(self.hidden_dim, int(math.sqrt(self.hidden_dim)), 1))
                self.attention2 = spectral_norm(nn.Conv2d(self.hidden_dim, int(math.sqrt(self.hidden_dim)), 1))
                self.attention3 = spectral_norm(nn.Conv2d(self.hidden_dim, self.hidden_dim, 1))
                self.softmax = nn.Softmax(dim=-1)
            exec(f'self.conv1{layer_id} = spectral_norm(nn.Conv2d(self.input_dim, self.input_dim, self.kernel, padding="valid"))')
            exec(f'self.conv2{layer_id} = spectral_norm(nn.Conv2d(self.input_dim, self.hidden_dim, self.kernel, padding="valid"))')
            exec(f'self.avg_pool{layer_id} = nn.AvgPool2d(self.strides[layer_id])')
            self.input_dim = self.hidden_dim
        
        self.apply(self.init_weights)

    def minibatch_stddev_layer(self, x, group_size=4):
        """
            Original version from ProGAN
        Args:
        x: Input tensor
        group_size:  The number of groups (Default value = 4)

        Returns:
            A standard deviation of chosen number groups. This result is repeated until the shape is matching input
            shape for concatication
        """
        group_size = min(group_size, x.size(dim=0)) 
        s = x.size()  
        y = torch.reshape(x, (group_size, -1, s[1], s[2], s[3]))  
        y = y.to(torch.float32) 
        y = y - torch.mean(y, dim=0, keepdim=True) 
        y = torch.mean(torch.square(y), dim=0)
        y = torch.sqrt(y + 1e-8)
        for axis in [1,2,3]:
            y = torch.mean(y, dim=int(axis), keepdim=True) 
        y = y.to(x.dtype)  
        y = torch.tile(y, [group_size, 1, s[2], s[3]])  
        return torch.cat((x, y), dim=1) 
    
    def forward(self, input):
        h = input
        for layer_id in range(len(self.strides)):
            
            if layer_id == 1:
                s = torch.matmul(self.hw_flatten(self.attention2(h)), torch.transpose(self.hw_flatten(self.attention1(h)), 2, 1))
                beta = self.softmax(s)
                o = torch.matmul(beta, self.hw_flatten(self.attention3(h)))
                attention_multiplier = torch.zeros(1, 1).to(self.device)
                o = torch.reshape(o, h.size())
                h = attention_multiplier * o + h
            
            input_channels = h.size(dim=-1)
            h_0 = h
            h = self.pad(h)
            h = eval(f'self.conv1{layer_id}(h)')
            h = F.leaky_relu(h)
            h = self.pad(h)
            h = eval(f'self.conv2{layer_id}(h)')
            h = eval(f'self.avg_pool{layer_id}(h)')
            if self.strides[layer_id][0] > 1 or self.strides[layer_id][1] > 1 or input_channels != h.size(dim=1):
                h_0 = self.pad(h_0)
                h_0 = eval(f'self.conv2{layer_id}(h_0)')
                h_0 = eval(f'self.avg_pool{layer_id}(h_0)')
            h = h_0 + h
        
        h = F.relu(h)
        h = self.minibatch_stddev_layer(h)
        h = torch.sum(h, dim=(1, 2))
        h = self.snlinear(h)
        return h

