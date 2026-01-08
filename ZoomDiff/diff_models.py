import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_torch_trans(heads=8, layers=1, channels=64):
    '''
    函数的作用是构建一个 Transformer 编码器，用于处理序列数据。
    参数包括头数 (heads)、层数 (layers) 和通道数 (channels)。
    '''
    #首先创建一个 Transformer 编码器层 (nn.TransformerEncoderLayer)
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    
    #然后创建一个 Transformer 编码器 (nn.TransformerEncoder)，并返回该编码器的实例
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    '''
    定义并Kaiming初始化一个1D卷积层
    '''
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class Conv1dSameShape(nn.Module):
    def __init__(self, channels):
        super(Conv1dSameShape, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        # x shape: (B, L, C)
        # Permute to shape (B, C, L) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        # Permute back to shape (B, L, C)
        x = x.permute(0, 2, 1)
        return x

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        '''
        num_steps:扩散步数, projection_dim:投影维度
        '''
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )#注册为模块的缓冲区（buffer）
        
        # 创建两个线性投影层，用于将嵌入映射到指定的维度
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        '''
        创建嵌入表
        '''
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_CSDI(nn.Module):
    def __init__(self, config, device, inputdim=1):
        super().__init__()
        self.channels = config["channels"]
        self.device = device

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1) # kernal Size 由 1 改为了 3
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    device=self.device,
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_static, cond_poi, side_info, current_step):
        B, K, L_, H_, _ = x.shape
        x = x.unsqueeze(1) # B, 1, K, L_, H_, H_
        x = x.reshape(B, 1, K * L_ * H_ * H_)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L_, H_, H_)

        diffusion_emb = self.diffusion_embedding(current_step) # (B, 128)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_static, cond_poi, side_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L_ * H_ * H_)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L_)
        x = x.reshape(B, K, L_, H_, H_)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, device):
        super().__init__() #调用父类的构造方法
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.side_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.cond_static_projection = Conv1d_with_init(4, 2 * channels, 1)
        self.cond_poi_projection = Conv1d_with_init(21, 2 * channels, 1)
        self.prior_projection = Conv1d_with_init(1, 2 * channels, 1)
        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.device = device        

    def forward_time(self, y, base_shape):
        B, channel, K, L_, H_, _ = base_shape
        if L_ == 1:
            return y
        y = y.reshape(B, channel, K, L_, H_, H_).permute(0, 2, 4, 5, 1, 3).reshape(B * K * H_ * H_, channel, L_)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, H_, H_, channel, L_).permute(0, 4, 1, 5, 2, 3) # (B, channel, K, L_, H_, H_)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L_, H_, _ = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L_, H_, H_).permute(0, 3, 4, 5, 1, 2).reshape(B * L_ * H_ * H_, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L_, H_, H_, channel, K).permute(0, 4, 5, 2, 3) # (B, channel, K, L_, H_, H_)
        return y
    
    def forward(self, x, cond_static, cond_poi, side_info, diffusion_emb):
        B, channel, K, L_, H_, _ = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L_ * H_ * H_)

        # 对扩散嵌入进行线性投影，并加上输入数据
        diffusion_emb_after = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb_after # 可参见Figure 6

        # 在时间和特征方向上进行变换
        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)=(B,C,L),下面就按K=1来做
        y = y.reshape(B, channel, K * L_ * H_ * H_)
        y = self.mid_projection(y)  # (B,2*channel,K*L),1D卷积

        # 处理时序与特征条件信息，并结合到处理后的数据中
        _, side_dim, _, _, _, _ = side_info.shape
        side_info = side_info.reshape(B, side_dim, K * L_ * H_ * H_)
        side_info = self.side_projection(side_info)  # (B,2*channel,K*L*H*H)
        y = y + side_info # 可参见Figure 6,如何添加入side_info

        # 处理环境条件信息，并结合到处理后的数据中
        _, Ds, _, _ = cond_static.shape
        cond_static = cond_static.unsqueeze(2).expand(-1, -1, L_, -1, -1) # (B,Ds,L_,H_,H_)
        cond_static = cond_static.reshape(B, Ds, L_ * H_ * H_)
        cond_static = self.cond_static_projection(cond_static)  # (B,2*channel,L_*H_*H_)
        y = y + cond_static

        # 处理POI条件信息，并结合到处理后的数据中
        _, Dp, _, _ = cond_poi.shape
        cond_poi = cond_poi.unsqueeze(2).expand(-1, -1, L_, -1, -1) # (B,Dp,L_,H_,H_)
        cond_poi = cond_poi.reshape(B, Dp, L_ * H_ * H_)
        cond_poi = self.cond_poi_projection(cond_poi)  # (B,2*channel,L_*H_*H_)
        y = y + cond_poi

        # 执行门控操作，使用 sigmoid 函数作为门控，tanh 函数作为过滤器
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        # 分离残差项和用于跳跃连接的项
        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        
        # 返回残差块的输出，包括残差项和用于跳跃连接的项
        return (x + residual) / math.sqrt(2.0), skip