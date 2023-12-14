import torch
import speechbrain as sb
from thop.vision.basic_hooks import count_convNd, count_normalization
from thop.vision.calc_func import l_prod
from dc1d.nn import gLN, DeformConv1d

def sb_calculate_conv2d_flops(input_size: list, output_size: list, kernel_size: list, groups: int, bias: bool = False):
    # n, out_c, oh, ow = output_size
    # n, in_c, ih, iw = input_size
    # out_c, in_c, kh, kw = kernel_size
    in_c = input_size[1]
    g = groups
    return l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])

def calculate_conv2d_flops(input_size: list, output_size: list, kernel_size: list, groups: int, bias: bool = False):
    # n, out_c, oh, ow = output_size
    # n, in_c, ih, iw = input_size
    # out_c, in_c, kh, kw = kernel_size
    in_c = input_size[1]
    g = groups
    return l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])

def sb_count_convNd(m: sb.nnet.CNN.Conv1d, x, y: torch.Tensor):
    x = x[0]

    kernel_ops = torch.zeros(m.conv.weight.size()[2:]).numel()  # Kw x Kh
    bias_ops = 1 if m.conv.bias is not None else 0

    m.total_ops += calculate_conv2d_flops(
        input_size = list(x.shape),
        output_size = list(y.shape),
        kernel_size = list(m.conv.weight.shape),
        groups = m.conv.groups,
        bias = m.conv.bias
    )

def count_deformconvNd(m: DeformConv1d, x, y: torch.Tensor):
    x = x[0]
    # print(x, x.shape,l_prod(x.shape))
    # exit()

    kernel_ops = torch.zeros(m.conv.weight.size()[2:]).numel()  # Kw x Kh
    bias_ops = 1 if m.conv.bias is not None else 0

    m.total_ops += calculate_conv2d_flops(
        input_size = list(x.shape),
        output_size = list(y.shape),
        kernel_size = list(m.conv.weight.shape),
        groups = m.conv.groups,
        bias = m.conv.bias
    ) + 2 * l_prod(x.shape) * kernel_ops

def count_mha(m: torch.nn.MultiheadAttention, x, y: torch.Tensor):
    try:
        q = x[0]
        k = x[1]
        v = x[2]
    except Exception as e:
        q = y[0]
        k = y[0]
        v = y[0]

    
    E_d = m.embed_dim
    m.total_ops += float(E_d * (2 * m.kdim + m.vdim)) * q.shape[1] # linear layers
    if m.batch_first:
        L = q.shape[1]
        S = v.shape[1]
    else:
        L = q.shape[0]
        S = v.shape[0]
      
    # print(L, S, q.shape, v.shape)
    
    E_q = m.kdim
    E_k = m.kdim
    E_v = m.vdim
    m.total_ops +=  float(L *  E_k * S) # Q.K^T
    m.total_ops +=  float(L *  E_v * S) # V.V^T
    m.total_ops +=  float(E_d ** 2 * S) 

def sb_count_mha(m: sb.nnet.attention.MultiheadAttention, x, y: torch.Tensor):
    q = x[0]
    k = x[1]
    v = x[2]
    E_d = m.att.embed_dim
    m.total_ops += float(E_d * (2 * m.att.kdim + m.att.vdim))*q.shape[0]
    
    L = q.shape[0]
    S = v.shape[0]


    E_q = m.att.kdim
    E_k = m.att.kdim
    E_v = m.att.vdim
    
    m.total_ops +=  float(L *  E_k**2 * S )
    m.total_ops +=  float(L *  E_v**2 * S)
    m.total_ops +=  float(E_d ** 2 * S)

sb_ops_dict={
    gLN: count_normalization,
    DeformConv1d: count_deformconvNd,
    sb.lobes.models.conv_tasnet.ChannelwiseLayerNorm: count_normalization,
    sb.lobes.models.conv_tasnet.GlobalLayerNorm: count_normalization,
    sb.lobes.models.dual_path.Decoder: count_convNd,
    # sb.nnet.CNN.Conv1d: sb_count_convNd,
    torch.nn.MultiheadAttention: count_mha,
    sb.nnet.attention.MultiheadAttention: sb_count_mha,
}
