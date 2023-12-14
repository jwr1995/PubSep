
""" Implementation of a popular speech separation model.
"""
import torch
import torch.nn as nn
import speechbrain as sb
import torch.nn.functional as F
from math import log2

from speechbrain.processing.signal_processing import overlap_and_add
from speechbrain.lobes.models.conv_tasnet import GlobalLayerNorm, ChannelwiseLayerNorm, Chomp1d, choose_norm
from speechbrain.nnet.CNN import Conv1d

# from fast_transformers.attention import linear_attention, attention_layer
# from fast_transformers.masking import FullMask, LengthMask

EPS = 1e-8

class WDTemporalBlocksSequential(sb.nnet.containers.Sequential):
    """
    A wrapper for the temporal-block layer to replicate it

    Arguments
    ---------
    input_shape : tuple
        Expected shape of the input.
    H : int
        The number of intermediate channels.
    P : int
        The kernel size in the convolutions.
    R : int
        The number of times to replicate the multilayer Temporal Blocks.
    X : int
        The number of layers of Temporal Blocks with different dilations.
    norm type : str
        The type of normalization, in ['gLN', 'cLN'].
    causal : bool
        To use causal or non-causal convolutions, in [True, False].

    Example
    -------
    >>> x = torch.randn(14, 100, 10)
    >>> H, P, R, X = 10, 5, 2, 3
    >>> WDTemporalBlocks = WDTemporalBlocksSequential(
    ...     x.shape, H, P, R, X, 'gLN', False
    ... )
    >>> y = WDTemporalBlocks(x)
    >>> y.shape
    torch.Size([14, 100, 10])
    """

    def __init__(
        self, 
        input_shape, 
        H, 
        P, 
        R, 
        X, 
        norm_type, 
        causal,
        se_kernel_size=20,
        bias=True,
        pool="global",
        attention_type="se",
        num_heads=4
        ):
        super().__init__(input_shape=input_shape)
        for r in range(R):
            for x in range(X):
                dilation = 2 ** x
                self.append(
                    WDTemporalBlock,
                    out_channels=H,
                    kernel_size=P,
                    stride=1,
                    padding="same",
                    dilation=dilation,
                    norm_type=norm_type,
                    causal=causal,
                    layer_name=f"temporalblock_{r}_{x}",
                    se_kernel_size=se_kernel_size,
                    bias=bias,
                    pool=pool,
                    attention_type=attention_type,
                    num_heads=num_heads
                )
    
    # def get_output_shape(self):
    #     """Returns expected shape of the output.

    #     Computed by passing dummy input constructed with the
    #     ``self.input_shape`` attribute.
    #     """
    #     self.store_intermediates = False
    #     with torch.no_grad():
    #         dummy_input = torch.zeros(self.input_shape)
    #         dummy_output = self(dummy_input)
    #     if isinstance(dummy_output,tuple):
    #         return dummy_output[0].shape
    #     else:
    #         return dummy_output.shape
        
    def set_store_intermediates(self, store_intermediates=True):
        self.store_intermediates = store_intermediates

    def forward(self, x):
        i_dict = {}
        for name, layer in self.items():
            
            if "store_intermediates" in self.__dict__.keys():
                if self.store_intermediates:
                    layer.set_store_intermediates(self.store_intermediates)
                    x, intermediate = layer(x)
                    i_dict[name] = intermediate
            else:
                x = layer(x)
            if isinstance(x, tuple):
                x = x[0]
            
        if "store_intermediates" in self.__dict__.keys():
                if self.store_intermediates:
                    return x, i_dict
        else:
            return x

class MaskNet(nn.Module):
    """
    Arguments
    ---------
    N : int
        Number of filters in autoencoder.
    B : int
        Number of channels in bottleneck 1 × 1-conv block.
    H : int
        Number of channels in convolutional blocks.
    P : int
        Kernel size in convolutional blocks.
    X : int
        Number of convolutional blocks in each repeat.
    R : int
        Number of repeats.
    C : int
        Number of speakers.
    norm_type : str
        One of BN, gLN, cLN.
    causal : bool
        Causal or non-causal.
    mask_nonlinear : str
        Use which non-linear function to generate mask, in ['softmax', 'relu'].

    Example:
    ---------
    >>> N, B, H, P, X, R, C = 11, 12, 2, 5, 3, 1, 2
    >>> MaskNet = MaskNet(N, B, H, P, X, R, C)
    >>> mixture_w = torch.randn(10, 11, 100)
    >>> est_mask = MaskNet(mixture_w)
    >>> est_mask.shape
    torch.Size([2, 10, 11, 100])
    """

    def __init__(
        self,
        N,
        B,
        H,
        P,
        X,
        R,
        C,
        norm_type="gLN",
        causal=False,
        mask_nonlinear="relu",
        se_kernel_size=20,
        bias=True,
        pool="global",
        attention_type="se",
        num_heads=4,
        store_intermediates=False
    ):
        super(MaskNet, self).__init__()

        # Hyper-parameter
        self.C = C
        self.mask_nonlinear = mask_nonlinear

        # Components
        # [M, K, N] -> [M, K, N]
        self.layer_norm = ChannelwiseLayerNorm(N)

        # [M, K, N] -> [M, K, B]
        self.bottleneck_conv1x1 = sb.nnet.CNN.Conv1d(
            in_channels=N, out_channels=B, kernel_size=1, bias=False,
        )

        # [M, K, B] -> [M, K, B]
        in_shape = (None, None, B)
        self.temporal_conv_net = WDTemporalBlocksSequential(
            in_shape, 
            H, 
            P, 
            R, 
            X, 
            norm_type, 
            causal,
            se_kernel_size=20,
            bias=True,
            pool=pool,
            attention_type=attention_type,
            num_heads=num_heads
        )

        # [M, K, B] -> [M, K, C*N]
        self.mask_conv1x1 = sb.nnet.CNN.Conv1d(
            in_channels=B, out_channels=C * N, kernel_size=1, bias=False
        )
    def set_store_intermediates(self, store_intermediates=True):
        self.store_intermediates = store_intermediates

    def forward(self, mixture_w):
        """Keep this API same with TasNet.

        Arguments
        ---------
        mixture_w : Tensor
            Tensor shape is [M, K, N], M is batch size.

        Returns
        -------
        est_mask : Tensor
            Tensor shape is [M, K, C, N].
        """

        mixture_w = mixture_w.permute(0, 2, 1)
        M, K, N = mixture_w.size()
        y = self.layer_norm(mixture_w)
        y = self.bottleneck_conv1x1(y)
        if "store_intermediates" in self.__dict__.keys():
                if self.store_intermediates:
                    self.temporal_conv_net.set_store_intermediates(self.store_intermediates)
        y = self.temporal_conv_net(y)
        if isinstance(y, tuple):
            i_dict = y[1]
            y = y[0]

        score = self.mask_conv1x1(y)

        # score = self.network(mixture_w)  # [M, K, N] -> [M, K, C*N]
        score = score.contiguous().reshape(
            M, K, self.C, N
        )  # [M, K, C*N] -> [M, K, C, N]

        # [M, K, C, N] -> [C, M, N, K]
        score = score.permute(2, 0, 3, 1)

        if self.mask_nonlinear == "softmax":
            est_mask = F.softmax(score, dim=2)
        elif self.mask_nonlinear == "relu":
            est_mask = F.relu(score)
        else:
            raise ValueError("Unsupported mask non-linear function")

        if "store_intermediates" in self.__dict__.keys():
            if self.store_intermediates:
                return est_mask, i_dict
            else:
                return est_mask
        else:
            return est_mask

class WDTemporalBlock(torch.nn.Module):
    """The conv1d compound layers used in Masknet.

    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input.
    out_channels : int
        The number of intermediate channels.
    kernel_size : int
        The kernel size in the convolutions.
    stride : int
        Convolution stride in convolutional layers.
    padding : str
        The type of padding in the convolutional layers,
        (same, valid, causal). If "valid", no padding is performed.
    dilation : int
        Amount of dilation in convolutional layers.
    norm type : str
        The type of normalization, in ['gLN', 'cLN'].
    causal : bool
        To use causal or non-causal convolutions, in [True, False].

    Example:
    ---------
    >>> x = torch.randn(14, 100, 10)
    >>> WDTemporalBlock = WDTemporalBlock(x.shape, 10, 11, 1, 'same', 1)
    >>> y = WDTemporalBlock(x)
    >>> y.shape
    torch.Size([14, 100, 10])
    """

    def __init__(
        self,
        input_shape,
        out_channels,
        kernel_size,
        stride,
        dilation,
        padding="same",
        norm_type="gLN",
        causal=False,
        se_kernel_size=20,
        bias=True,
        pool="global",
        attention_type="se",
        num_heads=4,
        store_intermediates=False,
    ):
        super().__init__()
        M, K, B = input_shape # batch x time x features

        self.layers = sb.nnet.containers.Sequential(input_shape=input_shape)
        # print(input_shape,out_channels)
        # [M, K, B] -> [M, K, H]
        self.layers.append(
            sb.nnet.CNN.Conv1d,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
            layer_name="conv",
        )
        self.layers.append(nn.PReLU(), layer_name="act")
        self.layers.append(
            choose_norm(norm_type, out_channels), layer_name="norm"
        )

        # [M, K, H] -> [M, K, B]
        self.layers.append(
            WDDepthwiseSeparableConv,
            out_channels=B,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            norm_type=norm_type,
            causal=causal,
            se_kernel_size=se_kernel_size,
            bias=bias,
            pool=pool,
            layer_name="DSconv",       
            attention_type=attention_type,
            num_heads=num_heads     
        )

    def set_store_intermediates(self, store_intermediates=True):
        self.store_intermediates = store_intermediates

    def forward(self, x, intermediates=False):
        """
        Arguments
        ---------
        x : Tensor
            Tensor shape is [batch size, sequence length, input channels].

        Returns
        -------
        x : Tensor
            Tensor shape is [M, K, B].
        """
        residual = x
        i_dict = {}
        for name, layer in self.layers.items():

            if type(layer)== WDDepthwiseSeparableConv and "store_intermediates" in self.__dict__.keys():
                if self.store_intermediates:
                    layer.set_store_intermediates(self.store_intermediates)
                    x, intermediate = layer(x)
                    i_dict[name] = intermediate
            else:
                try:
                    x = layer(x)
                except Exception as e:
                    raise e
        return x + residual, i_dict

class CumAvgPool1d(nn.Module):

    def __init__(
        self
    ):
        super().__init__()

    def forward(x):
        y = torch.zeros(x.shape)
        y[:,:,0] = x[:,:,0]
        for i in range(1,y.shape[-1]):
            y[:,:,i] = torch.mean(x[:,:,:i+1], dim=-1)

        return y

class SqueezeExciteAttention(nn.Module):

    def __init__(
        self,
        kernel_size=20,
        input_d=512,
        excite_d=4,
        output_d=2,
        pool="global" # ["global","sequential","cumulative"]
    ):
        super().__init__()

        if pool=="global" or kernel_size == None:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
        elif pool=="sequential":
            self.avg_pool = nn.AvgPool1d(
                kernel_size = kernel_size,
                stride=1,
                padding=kernel_size//2,
                ceil_mode=True,
            )
        else:
            self.avg_pool = CumAvgPool1d()

        self.linear_1 = nn.Linear(
            input_d,
            excite_d
        )

        self.linear_2 = nn.Linear(
            excite_d,
            output_d
        )
    
    def forward(self, x):
        # print(x.shape)
        
        L = x.shape[0]
        x = self.avg_pool(x) # batch x features x length#
        # assert (x.shape[1] == self.linear_1.weight.shape[-1]),"Mismatch " + str(x.shape[1])+"!="+str(self.linear_1.weight.shape[-1])
        
        x = F.relu(self.linear_1(x.moveaxis(-2,-1))) # batch x length x features

        x = F.relu(self.linear_2(x))

        # print(x.shape)
        assert x.shape[0] == L, f"L={L} but x has dimension {x.shape[1]}"

        return F.softmax(x,dim=-1) # batch x length x features

class WDDepthwiseConvolution(nn.Module):

    def __init__(
        self,
        kernel_size,
        input_shape=None,
        stride=1,
        dilation=1,
        attention_type="se", # or "la"
        se_kernel_size=20,
        padding="same",
        bias=True,
        pool="global",
        num_heads=4,
        store_intermediates=False,
        causal=False
    ):
        super().__init__()
        num_convs = int(log2(dilation))+1
        bz, time, chn = input_shape
        if attention_type=="se":
            self.se_block = SqueezeExciteAttention(
                kernel_size=se_kernel_size,
                input_d=chn,
                excite_d=4,
                output_d=num_convs,
                pool=pool,
            )
        else:
            self.attention = LinearSelfAttention(
                chn, 
                d_out=num_convs,
                num_heads=num_heads, 
                causal=causal,
            )
            self.se_block = None

        self.d_convlist = torch.nn.ModuleList(
            [
                Conv1d(
                    chn,
                    kernel_size,
                    input_shape=input_shape,
                    stride=stride,
                    dilation=2**(X),
                    padding=padding,
                    groups=chn,
                    bias=bias,
                ) for X in range(num_convs) # X in {0,..,num_convs-1}
            ]
        )
        
    def set_store_intermediates(self, store_intermediates=True):
        self.store_intermediates = store_intermediates

    def forward(self, x):
        # B x N x L
        if type(self.se_block) == SqueezeExciteAttention:
            attn = self.se_block(x.moveaxis(-1,-2))[:,:x.shape[-2],:]  # batch x length / 1 x feats
        else:
            attn = self.attention(x)
        
        dwcs = [dc(x) for dc in self.d_convlist]

        _sum = torch.zeros(dwcs[0].shape,device=attn.device)

        for a, dc in zip(attn.moveaxis(-1,0),dwcs):
            _sum += a.unsqueeze(-1) * dc

        if not "store_intermediates" in self.__dict__.keys():
            return _sum
        else:
            if self.store_intermediates:
                return _sum, attn.detach().cpu().numpy()

class WDDepthwiseSeparableConv(sb.nnet.containers.Sequential):
    """Building block for the Temporal Blocks of Masknet in ConvTasNet.

    Arguments
    ---------
    input_shape : tuple
        Expected shape of the input.
    out_channels : int
        Number of output channels.
    kernel_size : int
        The kernel size in the convolutions.
    stride : int
        Convolution stride in convolutional layers.
    padding : str
        The type of padding in the convolutional layers,
        (same, valid, causal). If "valid", no padding is performed.
    dilation : int
        Amount of dilation in convolutional layers.
    norm type : str
        The type of normalization, in ['gLN', 'cLN'].
    causal : bool
        To use causal or non-causal convolutions, in [True, False].

    Example
    -------
    >>> x = torch.randn(14, 100, 10)
    >>> DSconv =WDDepthwiseSeparableConv(x.shape, 10, 11, 1, 'same', 1)
    >>> y = DSconv(x)
    >>> y.shape
    torch.Size([14, 100, 10])

    """

    def __init__(
        self,
        input_shape,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        norm_type="gLN",
        causal=False,
        se_kernel_size=20,
        padding="same",
        bias=True,
        pool="global",
        attention_type="se",
        num_heads=4,
        store_intermediates=False
    ):
        super().__init__(input_shape=input_shape)

        batchsize, time, in_channels = input_shape

        # Depthwise [M, K, H] -> [M, K, H]
        self.append(
            WDDepthwiseConvolution,
            kernel_size=kernel_size,
            # input_shape=input_shape,
            stride=stride,
            dilation=dilation,
            se_kernel_size=se_kernel_size,
            padding=padding,
            bias=bias,
            pool=pool,
            layer_name="conv_0",
            attention_type=attention_type,
            num_heads=num_heads,
            causal=causal
        )

        if causal:
            self.append(Chomp1d(padding), layer_name="chomp")

        self.append(nn.PReLU(), layer_name="act")
        self.append(choose_norm(norm_type, in_channels), layer_name="act")

        # Pointwise [M, K, H] -> [M, K, B]
        self.append(
            sb.nnet.CNN.Conv1d,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
            layer_name="conv_1",
        )
        
    def set_store_intermediates(self, store_intermediates=True):
        self.store_intermediates = store_intermediates 

    def forward(self, x):
        """Applies layers in sequence, passing only the first element of tuples.

        Arguments
        ---------
        x : torch.Tensor
            The input tensor to run through the network.
        """
        i_dict = {}

        for name, layer in super().items():
            if type(layer) == WDDepthwiseConvolution and"store_intermediates" in self.__dict__.keys():
                if self.store_intermediates:
                    layer.set_store_intermediates(self.store_intermediates)
                    x, intermediate = layer(x)
                    i_dict[name] = intermediate
                else:
                    try:
                        x = layer(x)
                    except Exception as e:
                        raise e
            else:
                try:
                    x = layer(x)
                except Exception as e:
                    raise e
            if isinstance(x, tuple):
                    x = x[0]
        if "store_intermediates" in self.__dict__.keys():
                if self.store_intermediates:
                    return x, i_dict
        else: 
            return x

# class LinearSelfAttention(nn.Module):
#     def __init__(
#         self, 
#         d_model, 
#         d_out=2,
#         num_heads=4, 
#         device='cuda',
#         causal=False,
#         ):
       
#         super(LinearSelfAttention, self).__init__()

#         self.linear_r = nn.Linear(d_model, d_out*num_heads)

#         self.linear_attn = attention_layer.AttentionLayer(
#             attention = linear_attention.LinearAttention(d_out),
#             d_model=d_out*num_heads,
#             n_heads=num_heads,
#             )
        
#         self.linear_o = nn.Linear(d_out*num_heads,d_out)

#         self.causal = causal
    
#     def forward(self,x):
#         """
#         Input shape = B x L x N
#         """
#         x = F.relu(self.linear_r(x))

#         B, L, N = x.shape

#         attn_mask = FullMask(
#             mask=None, 
#             N=L, 
#             M=N, 
#             device=x.device
#             )
#         lens_mask = LengthMask(
#             lengths=torch.ones((B,))*L,
#             device=x.device
#         )

#         x = F.relu(self.linear_attn(
#             queries = x,
#             keys = x,
#             values = x,
#             attn_mask = attn_mask,
#             query_lengths = lens_mask,
#             key_lengths = lens_mask
#             ))   # B, L, N

#         x = F.softmax(self.linear_o(x),dim=-1)

#         return x
        
if __name__ == '__main__':
    batch_size, N, L = 4, 512, 3321
    P=3

    x = torch.rand((batch_size, N, L))

    N=N
    B=N//4
    H=N
    P=3
    X=3
    R=4
    C=2

    ddc = MaskNet(
        N=N,
        B=B,
        H=H,
        P=P,
        X=X,
        R=R,
        C=C,
        norm_type="gLN",
        causal=False,
        mask_nonlinear="relu",
        se_kernel_size=64,
        bias=True,
        pool="sequential",
        attention_type="se"
    )

    print(x.shape)
    ddc.set_store_intermediates(True)
    x,i = ddc(x)
    # print(i)
    print(x.shape,i.keys())
