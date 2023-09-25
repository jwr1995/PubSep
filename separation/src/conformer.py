import torch
from torch import nn
# from flash_attn.flash_attention import FlashMHA

from typing import Callable

from typing import Optional, Tuple

"""
Conformer model from "Conformer: Convolution-augmented Transformer for Speech Recognition"
Much of the code is copied from PyTorch implementation to fix a padding issue: https://pytorch.org/audio/main/generated/torchaudio.models.Conformer.html
Licence: https://github.com/pytorch/pytorch/blob/master/LICENSE
"""
__all__ = ["Conformer"]

def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) >= lengths.unsqueeze(1)
    return padding_mask


class _ConvolutionModule(torch.nn.Module):
    r"""Conformer convolution module.

    Args:
        input_dim (int): input dimension.
        num_channels (int): number of depthwise convolution layer input channels.
        depthwise_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        bias (bool, optional): indicates whether to add bias term to each convolution layer. (Default: ``False``)
        use_group_norm (bool, optional): use GroupNorm rather than BatchNorm. (Default: ``False``)
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        depthwise_kernel_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_group_norm: bool = False,
    ) -> None:
        super().__init__()

        self.layer_norm = torch.nn.LayerNorm(input_dim)
        self.sequential = torch.nn.Sequential(
            torch.nn.Conv1d(
                input_dim,
                2 * num_channels,
                1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            torch.nn.GLU(dim=1),
            torch.nn.Conv1d(
                num_channels,
                num_channels,
                depthwise_kernel_size,
                stride=1,
                padding="same", 
                groups=num_channels,
                bias=bias,
            ),
            torch.nn.GroupNorm(num_groups=1, num_channels=num_channels)
            if use_group_norm
            else torch.nn.BatchNorm1d(num_channels),
            torch.nn.SiLU(),
            torch.nn.Conv1d(
                num_channels,
                input_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, D)`.

        Returns:
            torch.Tensor: output, with shape `(B, T, D)`.
        """
        x = self.layer_norm(input)
        x = x.transpose(1, 2)
        x = self.sequential(x)
        return x.transpose(1, 2)


class _FeedForwardModule(torch.nn.Module):
    r"""Positionwise feed forward layer.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        dropout (float, optional): dropout probability. (Default: 0.0)
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim, bias=True),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, input_dim, bias=True),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(*, D)`.

        Returns:
            torch.Tensor: output, with shape `(*, D)`.
        """
        return self.sequential(input)


class ConformerLayer(torch.nn.Module):
    r"""Conformer layer that constitutes Conformer.

    Args:
        input_dim (int): input dimension.
        ffn_dim (int): hidden layer dimension of feedforward network.
        num_attention_heads (int): number of attention heads.
        depthwise_conv_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)
    """

    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_attention_heads: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
        use_flash: bool = False,
    ) -> None:
        super().__init__()

        self.ffn1 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)

        self.self_attn_layer_norm = torch.nn.LayerNorm(input_dim)
        if not use_flash:
            self.self_attn = torch.nn.MultiheadAttention(input_dim, num_attention_heads, dropout=dropout)
        # else:
        #     self.self_attn = FlashMHA(input_dim, num_attention_heads, attention_dropout=dropout)

        self.self_attn_dropout = torch.nn.Dropout(dropout)

        self.conv_module = _ConvolutionModule(
            input_dim=input_dim,
            num_channels=input_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            bias=True,
            use_group_norm=use_group_norm,
        )

        self.ffn2 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.final_layer_norm = torch.nn.LayerNorm(input_dim)
        self.convolution_first = convolution_first

    def _apply_convolution(self, input: torch.Tensor) -> torch.Tensor:
        residual = input
        input = input.transpose(0, 1)
        input = self.conv_module(input)
        input = input.transpose(0, 1)
        input = residual + input
        return input

    def forward(self, input: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): input, with shape `(T, B, D)`.
            key_padding_mask (torch.Tensor or None): key padding mask to use in self attention layer.

        Returns:
            torch.Tensor: output, with shape `(T, B, D)`.
        """
        residual = input
        x = self.ffn1(input)
        x = x * 0.5 + residual
        # print(x)
        if self.convolution_first:
            x = self._apply_convolution(x)
        # print(x)
        residual = x
        x = self.self_attn_layer_norm(x)
        # print(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.self_attn_dropout(x)
        x = x + residual

        if not self.convolution_first:
            x = self._apply_convolution(x)

        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual

        x = self.final_layer_norm(x)
        return x


class Conformer(torch.nn.Module):
    r"""Conformer architecture introduced in
    *Conformer: Convolution-augmented Transformer for Speech Recognition*
    :cite:`gulati2020conformer`.

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads in each Conformer layer.
        ffn_dim (int): hidden layer dimension of feedforward networks.
        num_layers (int): number of Conformer layers to instantiate.
        depthwise_conv_kernel_size (int): kernel size of each Conformer layer's depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)

    Examples:
        >>> conformer = Conformer(
        >>>     input_dim=80,
        >>>     num_heads=4,
        >>>     ffn_dim=128,
        >>>     num_layers=4,
        >>>     depthwise_conv_kernel_size=31,
        >>> )
        >>> lengths = torch.randint(1, 400, (10,))  # (batch,)
        >>> input = torch.rand(10, int(lengths.max()), input_dim)  # (batch, num_frames, input_dim)
        >>> output = conformer(input, lengths)
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
    ):
        super().__init__()

        self.conformer_layers = torch.nn.ModuleList(
            [
                ConformerLayer(
                    input_dim,
                    ffn_dim,
                    num_heads,
                    depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, input_dim)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.

        Returns:
            (torch.Tensor, torch.Tensor)
                torch.Tensor
                    output frames, with shape `(B, T, input_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output frames.
        """
        encoder_padding_mask = _lengths_to_padding_mask(lengths)

        x = input.transpose(0, 1)
        for layer in self.conformer_layers:
            x = layer(x, encoder_padding_mask)
        return x.transpose(0, 1), lengths


class Bottleneck(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(input_dim)
        self.conv = nn.Conv1d(input_dim, bottleneck_dim, 1)
        self.prelu = nn.PReLU(bottleneck_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x.moveaxis(1,2))
        x = self.conv(x.moveaxis(1,2))
        x = self.prelu(x)
        return x

class Projection(nn.Module):
    def __init__(self, bottleneck_dim: int, out_dim: int, num_sources: int, mask_nonlinear: Callable):
        super().__init__()
        self.ln = nn.LayerNorm(bottleneck_dim)
        self.conv = nn.Conv1d(bottleneck_dim, num_sources*out_dim, 1)
        self.mask_nonlinear = mask_nonlinear
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = self.conv(x.moveaxis(1,2))
        x = self.mask_nonlinear(x)
        return x

class SubSamplingLayers(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, num_layers=2) -> None:        
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        assert stride == kernel_size/2, "50% overlap expected for stride"
        self.stride=stride
        self.kernel_size=kernel_size
        self.layers = nn.Sequential(
            *([nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),nn.PReLU(out_channels),nn.LayerNorm(out_channels)]*num_layers)
        )
    
    def forward(self, x):
        residuals = []
        for layer in self.layers:
            if isinstance(layer, nn.Conv1d):
                residuals.append(x)
                x=layer(x)
            elif isinstance(layer, nn.PReLU):
                x=layer(x)
            elif isinstance(layer, nn.LayerNorm):
                x=layer(x.moveaxis(1,2)).moveaxis(1,2)
        return x, residuals

class SuperSamplingLayers(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, num_layers=2, output_seq_len=0) -> None:        
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        assert stride == kernel_size/2, "50% overlap expected for stride"
        self.stride=stride
        self.kernel_size=kernel_size
        self.layers = nn.Sequential(
            *([nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False, output_padding=1),nn.PReLU(out_channels),nn.LayerNorm(out_channels)]*num_layers)
        )
    
    def forward(self, x, residuals):
        j=1
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.ConvTranspose1d):
                x=layer(x.moveaxis(1,2))[:,:,:residuals[-j].shape[-1]]+residuals[-j]
                j+=1
            elif isinstance(layer, nn.PReLU):
                x=layer(x)#.moveaxis(1,2)
            elif isinstance(layer, nn.LayerNorm):
                x=layer(x.moveaxis(1,2))#.moveaxis(1,2)

        # x=self.layers(x.moveaxis(1,2)).moveaxis(1,2)
        return x

class MaskNet(Conformer):
    def __init__(self,
        input_dim: int,
        bottleneck_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
        num_sources: int = 2,
        mask_nonlinear: Callable = nn.functional.relu,
        subsampling_layers: int = 2,
    ):
        super().__init__(
            input_dim=bottleneck_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            use_group_norm=use_group_norm,
            convolution_first=convolution_first,
        )
        self.bottleneck = Bottleneck(input_dim, bottleneck_dim)
        self.subsampling = SubSamplingLayers(bottleneck_dim, bottleneck_dim, num_layers=subsampling_layers)
        self.supersampling = SuperSamplingLayers(bottleneck_dim, bottleneck_dim, num_layers=subsampling_layers)
        self.projection = Projection(bottleneck_dim, input_dim, num_sources, mask_nonlinear)
        self.num_sources = num_sources
    
    def forward(self, x: torch.Tensor, lens: torch.Tensor=None) -> torch.Tensor:
        bs, ch, l  = x.shape
        x = self.bottleneck(x)
        x, sub_res = self.subsampling(x)
        if lens is None:
            lens = torch.ones((x.shape[0]), dtype=torch.long).to(x.device) * x.shape[2]
        else:
            lens = (torch.ceil(lens*x.shape[-1])).long()
        y, lens = super().forward(x.moveaxis(1,2), lens)
        x = y+x.moveaxis(1,2)
        x = self.supersampling(x, residuals=sub_res)
        x = self.projection(x) # batch_size, num_sources*input_dim, length        
        x = torch.stack(torch.split(x, x.shape[1]//self.num_sources, dim=1))
        x = x[:, :, :, :l]
        assert x.shape == (self.num_sources, bs, ch, l)
        return x

if __name__ == '__main__':
    import torch
    from torchsummary import summary
    from thop import profile

    # model = MaskNet(257, 256, 4, 256, 6, 31)
    in_channels = 256

    bottleneck_dim = 1024
    model = MaskNet(
        input_dim=in_channels,
        bottleneck_dim=bottleneck_dim,
        num_heads=4,
        ffn_dim=bottleneck_dim,
        num_layers=16,
        depthwise_conv_kernel_size=64,
        dropout=0.1,
        use_group_norm=True,
        convolution_first=True,
        num_sources=2,
        mask_nonlinear=nn.functional.relu,
    ) 
    batch_size = 1
    length = 1000

    # lens = torch.randint(low=1, high=length, size=(batch_size,))
    # print(lens)
    x = torch.randn(1, in_channels, length) 
    y = model(x)
    # print(y.shape)
    # summary(model, (in_channels, 128))
    macs, size = (profile(model, inputs=(x,)))
    print(macs/1e9, "G, ", size/1e6, "M")