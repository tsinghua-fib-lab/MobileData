import numpy as np
import torch
import torch.nn as nn
from diff_models import diff_CSDI, Conv1d_with_init

def nonlinear_step_cutting(num_steps, num_segments, power=1):
    # 1. 在 [0, 1] 上生成非均匀点，使后段更长
    x = np.linspace(0, 1, num_segments + 1)
    y = x ** power  # 越往后，值增加越快（间距变大）

    # 2. 映射到 [0, num_steps] 区间
    steps = (y * num_steps).round().astype(int)

    # 3. 从大到小排列
    descending_steps = sorted(set(steps), reverse=True)

    return list(reversed([num_steps - _  for _ in descending_steps]))

class ConvMappingModule(nn.Module):
    def __init__(self, d_model, K, out_channel, activation=nn.ReLU()):
        super(ConvMappingModule, self).__init__()
        self.conv = nn.Conv2d(in_channels=d_model, out_channels=out_channel, kernel_size=(K, 1), stride=1)
        self.activation = activation

    def forward(self, x):
        """
        x: Tensor, shape = (B, d_model, K, L)
        Returns:
        output: Tensor, shape = (B, out_channel, L)
        """
        x = self.conv(x)  # Shape: (B, out_channel, 1, L)
        x = self.activation(x)  # Apply activation
        return x.squeeze(2)  # Remove the size-1 dimension at dim=2, shape -> (B, out_channel, L)

class CSDI_base(nn.Module):
    def __init__(self, target_dim, args):
        super().__init__()
        self.device = args.device
        self.target_dim = target_dim
        self.config = args.config
        self.args = args
        self.mse = nn.MSELoss()

        # 从配置中获取时空嵌入维度 (timeemb)，该维度用于生成时间嵌入。
        self.emb_time_dim = self.config["model"]["timeemb"]
        self.emb_spatial_dim = self.config["model"]["spatialemb"]
        # 从配置中获取特征嵌入维度 (feature_emb)，该维度用于生成特征嵌入。
        self.feature_dim = self.config["model"]["feature_emb"]
        # 从配置中获取特征嵌入维度 (condemb)，该维度用于生成特征嵌入。
        self.emb_cond_dim = self.config["model"]["cond_emb"]
        # 从配置中获取是否为无条件模型 (is_unconditional)，即模型是否仅基于噪声进行扩散，而不考虑条件观测。
        self.is_unconditional = self.config["model"]["is_unconditional"]
        self.sim = self.config["model"]["sim"]

        # 计算总的side嵌入维度
        self.side_dim = self.emb_time_dim + self.emb_spatial_dim + self.feature_dim
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.feature_dim
        )

        self.feature_pattern_projection = ConvMappingModule(d_model=32, K=258, out_channel=64).to(self.device)

        # 配置扩散模型
        config_diff = self.config["diffusion"]
        config_diff["side_dim"] = self.side_dim

        self.diffmodel = diff_CSDI(config_diff, self.device, inputdim=1)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]

        # 配置多尺度扩散过程
        self.multi_scale = config_diff["multi_scale"]
        self.time_scale = self.multi_scale["time_scale"] # 时间分级数
        self.spatial_scale = self.multi_scale["spatial_scale"] # 空间分级数
        self.spatial_scale_coarse = self.multi_scale["spatial_scale_coarse"] # 空间粗度
        self.mini_steps_time = self.num_steps // (self.time_scale) # 时间步长
        self.mini_steps_spatial = self.num_steps // (self.spatial_scale) # 空间步长

        self.max_scale = max(self.time_scale, self.spatial_scale)
        self.step_list = torch.tensor(nonlinear_step_cutting(self.num_steps, self.max_scale)).to(self.device)
        self.time_step_list = torch.concat((self.step_list[:self.time_scale], self.step_list[-1:]), dim=0)
        self.spatial_step_list = torch.concat((self.step_list[:self.spatial_scale], self.step_list[-1:]), dim=0)

        # 合并时间步和空间步，并标记为ts、t、s
        combined_step_values = sorted(set(self.time_step_list.cpu().numpy()) | set(self.spatial_step_list.cpu().numpy()), reverse=True)
        combined_step_values = torch.tensor(combined_step_values).to(self.device)
        combined_step_labels = []

        t_scale_labels, s_scale_labels = [], []
        i_t, i_s = self.time_scale + 1, self.spatial_scale + 1
        for v in combined_step_values:
            in_time = v in self.time_step_list
            in_spatial = v in self.spatial_step_list
            t_scale_labels.append(i_t)
            s_scale_labels.append(i_s)
            if in_time and in_spatial:
                combined_step_labels.append('ts')
                i_t -= 1
                i_s -= 1
            elif in_time:
                combined_step_labels.append('t')
                i_t -= 1
            else:
                combined_step_labels.append('s')
                i_s -= 1

        self.combined_step_list = [combined_step_values, combined_step_labels, t_scale_labels, s_scale_labels]

        # 计算alpha和beta
        self.beta, self.alpha_hat, self.alpha, self.alpha_torch = [], [], [], []
        for i in range(len(self.combined_step_list[0]) - 1):
            if config_diff["schedule"] == "quad":
                beta = np.linspace(
                    config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.combined_step_list[0][i] - self.combined_step_list[0][i + 1] + 1
                ) ** 2
            elif config_diff["schedule"] == "linear":
                beta = np.linspace(
                    config_diff["beta_start"], config_diff["beta_end"], self.combined_step_list[0][i] - self.combined_step_list[0][i + 1] + 1
                )
        
            alpha_hat = 1 - beta
            alpha = np.cumprod(alpha_hat) # 累乘函数
            alpha_torch = torch.tensor(alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

            self.beta.append(beta)
            self.alpha_hat.append(alpha_hat)
            self.alpha.append(alpha)
            self.alpha_torch.append(alpha_torch)

        # 空间位置编码
        self.spatial_embedding = [nn.Parameter(torch.randn(self.emb_time_dim, self.args.H, self.args.H)).to(self.device) for _ in range(self.max_scale)]
        # self.spatial_embedding = nn.ParameterList(
        #     [
        #         nn.Parameter(torch.randn(self.emb_time_dim, self.args.H, self.args.H))
        #         for _ in range(self.max_scale)
        #     ]
        # )

        # 噪声融合机制
        self.predicted_noise_projection = Conv1d_with_init(in_channels=1, out_channels=128, kernel_size=1).to(self.device)
        self.noise_prior_projection = Conv1d_with_init(in_channels=1, out_channels=128, kernel_size=1).to(self.device)
        self.infused_noise_projection = Conv1d_with_init(in_channels=128, out_channels=1, kernel_size=1).to(self.device)

    def time_embedding(self, pos, d_model=128):
        '''
        self, pos, d_model=128
        生成时间嵌入，通过对时间进行编码而得到向量表示
        返回生成的时间嵌入矩阵 pe,其中每一行代表一个时间步的嵌入向量,形状为 (B, L, d_model)
        '''
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def get_feature_pattern(self, observed_cond, out_channel, d_model):
        B, K, L = observed_cond.shape # (B, K, L)
        
        # 定义pattern提取的Transformer编码器
        time_layer_cond = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=8, dim_feedforward=64, activation="gelu"
            ),
            num_layers = 1,
        ).to(self.device)
        feature_layer_cond = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=8, dim_feedforward=64, activation="gelu"
            ),
            num_layers = 1,
        ).to(self.device)

        # 在时间维提取pattern
        observed_cond = observed_cond.unsqueeze(1).expand(-1, d_model, -1, -1) #用映射代替expand
        observed_cond = observed_cond.reshape(B, d_model, K, L).permute(0, 2, 1, 3).reshape(B * K, d_model, L)
        time_pattern_cond = time_layer_cond(observed_cond.permute(2, 0, 1)).permute(1, 2, 0)
        time_pattern_cond = time_pattern_cond.reshape(B, K, d_model, L).permute(0, 2, 1, 3) # (B, d_model, K, L)

        # 在特征维提取pattern
        time_pattern_cond = time_pattern_cond.permute(0, 3, 1, 2).reshape(B * L, d_model, K)
        feature_pattern_cond = feature_layer_cond(time_pattern_cond.permute(2, 0, 1)).permute(1, 2, 0)
        feature_pattern_cond = feature_pattern_cond.reshape(B, L, d_model, K).permute(0, 2, 3, 1) # (B, d_model, K, L)

        # 进行维度映射，整理为标准形式(B, out_channel, L)
        if self.feature_pattern_projection == None:
            self.feature_pattern_projection = ConvMappingModule(d_model=d_model, K=K, out_channel=out_channel).to(self.device)
        
        pattern_cond = self.feature_pattern_projection(feature_pattern_cond)

        return pattern_cond

    def get_side_info(self, observed_tp, observed_data):
        B, K, L, H, _ = observed_data.shape  # [B, K, L, H, H]

        # Time Embedding
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # [B, L, emb]
        time_embed = time_embed.unsqueeze(2).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, K, -1, H, H)  # [B, L, K, emb, H, H]

        # Feature Embedding
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # [K, emb]
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(B, L, K, -1, H, H)  # [B, L, K, emb, H, H]

        # Spatial-Temporal Position Encoding
        # spatial_embedding: [H, H, emb]
        spatial_embed = [self.spatial_embedding[i].unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, L, K, -1, -1, -1) for i in range(self.max_scale)]  # [B, L, K, emb, H, H]
        # Modulate spatial embedding with time embedding
        # spatiotemporal_embed = spatial_embed * time_embed  # [B, L, K, emb, H, H]

        # Concatenate side_info
        side_info = [torch.cat([time_embed, spatial_embed[i], feature_embed], dim=3).permute(0, 3, 2, 1, 4, 5) for i in range(self.max_scale)]  # concat on emb dim
        

        # # Transform to expected output format (B, *, K, L, H, H)
        # side_info = side_info.permute(0, 3, 2, 1, 4, 5)  # [B, total_emb, K, L, H, H]

        return side_info

    def calc_loss_valid(
        self, observed_data, observed_cond_static, observed_cond_poi, side_info, is_train
    ):
        '''
        计算在模型验证过程中的损失, 取多个时间步并计算均值
        '''
        loss_sum = 0
        for n in range(self.num_steps):  # calculate loss for all n
            loss = self.calc_loss(
                observed_data, observed_cond_static, observed_cond_poi, side_info, is_train, set_n=n
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, observed_cond_static, observed_cond_poi, side_info, is_train, set_n=-1
    ):
        '''
        计算模型在给定时间步 n 下的损失
        '''
        B, K, L, H, _ = observed_data.shape
        if is_train != 1:  # for validation
            n = (torch.ones(B) * set_n).long().to(self.device)
        else: # for train
            # 获取随机时间步
            n = torch.randint(0, self.num_steps, [B]).to(self.device)

        indices_for_scales = []
        noise = [None for _ in range(B)]
        data_noised = [None for _ in range(B)]
        noise_prior = [None for _ in range(B)]
        previous_data_noised = [None for _ in range(B)]
        multi_scale_side_info = [None for _ in range(B)]

        for scale_level in range(len(self.combined_step_list[0]) - 1):
            lower = self.combined_step_list[0][scale_level + 1]
            upper = self.combined_step_list[0][scale_level]

            # 先生成掩码：True 表示 n 在区间内
            mask = (n >= lower.item()) & (n < upper.item())  # [B] bool

            # 再用 nonzero 获得索引：
            indices = mask.nonzero(as_tuple=False).squeeze(1)  # [N_in_interval]
            indices_for_scales.append(indices)

            if len(indices) != 0:
                i_t = self.combined_step_list[2][scale_level + 1] # 时间对应的 i
                i_s = self.combined_step_list[3][scale_level + 1] # 空间对应的 i

                n_scale_level = n[indices]

                # current_alpha_scale_level = self.alpha_torch[n_scale_level] / self.alpha_torch[lower.item() - 1] if lower.item() > 0 else self.alpha_torch[n_scale_level]
                current_alpha_scale_level = self.alpha_torch[scale_level][n_scale_level - lower + 1]

                # 时空粒度分解
                current_observed_data_scale_level = self.spatial_division(observed_data[indices], i_s)
                current_observed_data_scale_level = self.temporal_division(current_observed_data_scale_level, i_t)

                current_side_info_scale_level = side_info[scale_level][indices]
                # current_side_info_scale_level = self.spatial_division(current_side_info_scale_level.reshape(len(indices), -1, L, H, H), i_s).reshape(len(indices), -1, K, L, H, H)
                # current_side_info_scale_level = self.temporal_division(current_side_info_scale_level.reshape(len(indices), -1, L, H, H), i_t).reshape(len(indices), -1, K, L, H, H)

                noise_scale_level = torch.randn_like(current_observed_data_scale_level)
                data_noised_scale_level = (current_alpha_scale_level.unsqueeze(-1).unsqueeze(-1) ** 0.5) * current_observed_data_scale_level + (1.0 - current_alpha_scale_level.unsqueeze(-1).unsqueeze(-1)) ** 0.5 * noise_scale_level

                if scale_level > 0:
                    i_t_previous = self.combined_step_list[2][scale_level]
                    i_s_previous = self.combined_step_list[3][scale_level]

                    previous_data_noised_scale_level = self.spatial_division(observed_data[indices], i_s_previous)
                    previous_data_noised_scale_level = self.temporal_division(previous_data_noised_scale_level, i_t_previous)
                    noise_prior_scale_level = (data_noised_scale_level - (current_alpha_scale_level.unsqueeze(-1).unsqueeze(-1) ** 0.5) * previous_data_noised_scale_level) / (1.0 - current_alpha_scale_level.unsqueeze(-1).unsqueeze(-1)) ** 0.5
                    # noise_prior_scale_level = ((current_alpha_scale_level.unsqueeze(-1).unsqueeze(-1) ** 0.5) * previous_data_noised_scale_level) / (1.0 - current_alpha_scale_level.unsqueeze(-1).unsqueeze(-1)) ** 0.5
                else:
                    previous_data_noised_scale_level = torch.randn_like(observed_data[indices])
                    noise_prior_scale_level = torch.zeros_like(observed_data[indices])
            
                for i in range(len(indices)):
                    noise[indices[i]] = noise_scale_level[i]
                    data_noised[indices[i]] = data_noised_scale_level[i]
                    noise_prior[indices[i]] = noise_prior_scale_level[i]
                    previous_data_noised[indices[i]] = previous_data_noised_scale_level[i]
                    multi_scale_side_info[indices[i]] = current_side_info_scale_level[i]

        noise = torch.stack(noise)
        data_noised = torch.stack(data_noised)
        noise_prior = torch.stack(noise_prior)
        previous_data_noised = torch.stack(previous_data_noised)
        multi_scale_side_info = torch.stack(multi_scale_side_info)

        # predicted = self.diffmodel(data_noised, previous_data_noised, observed_cond_static, observed_cond_poi, multi_scale_side_info, n)
        predicted = self.diffmodel(data_noised, observed_cond_static, observed_cond_poi, multi_scale_side_info, n)

        predicted_noise_emb = self.predicted_noise_projection(predicted.reshape(B, K, -1))
        prior_noise_emb = self.noise_prior_projection(noise_prior.reshape(B, K, -1))
        noise_infused = self.sim * predicted + (1 - self.sim) * self.infused_noise_projection(predicted_noise_emb + prior_noise_emb).reshape(B, K, L, H, H)

        # noise_infused = self.sim * predicted + (1 - self.sim) * noise_prior  # (B,K,L,H,H)

        loss = self.mse(noise, noise_infused)
                
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data):
        if self.is_unconditional == True: # 非条件生成
            total_input = noisy_data  # (B,1,K,L)
        else:
            cond_obs = observed_data.unsqueeze(1)
            noisy_target = noisy_data.unsqueeze(1) #在没有观测到的位置引入噪声
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, shape_base, observed_cond_static, observed_cond_poi, side_info, n_samples):

        B, K, L, H, _ = shape_base

        imputed_samples = torch.zeros(B, n_samples, K, L, H, H).to(self.device)
        sample_scale_save = torch.zeros(B, n_samples, len(self.combined_step_list[0]) - 1, K, L, H, H).to(self.device)

        for i in range(n_samples):

            current_sample = torch.randn(shape_base).to(self.device)
            sample_scale = []
            current_sample_scale_save = []

            scale_level = 0
            for n in range(self.num_steps - 1, -1, -1):

                if n < self.combined_step_list[0][scale_level + 1]:
                    scale_level += 1
                    current_sample = torch.randn(shape_base).to(self.device)
                
                # 所处级别的step上下界
                lower = self.combined_step_list[0][scale_level + 1]
                upper = self.combined_step_list[0][scale_level]
                
                # else:
                diff_input = current_sample
                
                i_t = self.combined_step_list[2][scale_level + 1] # 时间对应的 i
                i_s = self.combined_step_list[3][scale_level + 1] # 空间对应的 i
                alpha_n = self.alpha_torch[scale_level][n - lower.item() + 1]

                # 时空粒度分解
                side_info_scale_level = side_info[scale_level]
                # side_info_scale_level = self.spatial_division(side_info_scale_level.reshape(B, -1, L, H, H), i_s).reshape(B, -1, K, L, H, H)
                # side_info_scale_level = self.temporal_division(side_info_scale_level.reshape(B, -1, L, H, H), i_t).reshape(B, -1, K, L, H, H)

                if scale_level > 0:
                    i_t_previous = self.combined_step_list[2][scale_level]
                    i_s_previous = self.combined_step_list[3][scale_level]                    

                    previous_data_noised_scale_level = self.spatial_division(sample_scale[-1], i_s_previous)
                    previous_data_noised_scale_level = self.temporal_division(previous_data_noised_scale_level, i_t_previous)

                    noise_prior = (diff_input - (alpha_n ** 0.5) * previous_data_noised_scale_level) / (1.0 - alpha_n) ** 0.5
                    # noise_prior = ((alpha_n ** 0.5) * previous_data_noised_scale_level) / (1.0 - alpha_n) ** 0.5

                else:
                    previous_data_noised_scale_level = torch.randn_like(diff_input)
                    noise_prior = torch.zeros_like(diff_input)
                
                # predicted = self.diffmodel(diff_input, previous_data_noised_scale_level, observed_cond_static, observed_cond_poi, side_info_scale_level, torch.tensor([n]).to(self.device))
                predicted = self.diffmodel(diff_input, observed_cond_static, observed_cond_poi, side_info_scale_level, torch.tensor([n]).to(self.device))

                predicted_noise_emb = self.predicted_noise_projection(predicted.reshape(B, K, -1))
                prior_noise_emb = self.noise_prior_projection(noise_prior.reshape(B, K, -1))
                noise_infused = self.sim * predicted + (1 - self.sim) * self.infused_noise_projection(predicted_noise_emb + prior_noise_emb).reshape(B, K, L, H, H)
                # noise_infused = self.sim * predicted + (1 - self.sim) * noise_prior  # (B,K,L,H,H)

                coeff1 = 1 / self.alpha_hat[scale_level][n - lower.item() + 1] ** 0.5
                coeff2 = (1 - self.alpha_hat[scale_level][n - lower.item() + 1]) / (1 - alpha_n) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * noise_infused)

                if n > 0:
                    noise = torch.randn_like(current_sample)
                    # alpha_n_1 = self.alpha_torch[n - 1] / self.alpha_torch[self.combined_step_list[0][scale_level + 1] - 1] if self.combined_step_list[0][scale_level + 1] > 0 else self.alpha_torch[n - 1]
                    alpha_n_1 = self.alpha_torch[scale_level][n - lower.item()]
                    sigma = (
                        (1.0 - alpha_n_1) / (1.0 - alpha_n) * self.beta[scale_level][n - lower.item() + 1]
                    ) ** 0.5
                    current_sample += sigma * noise
                
                if n == self.combined_step_list[0][scale_level + 1]:
                    sample_scale.append(current_sample)
                    current_sample_scale_save.append(current_sample.detach().unsqueeze(1))

            current_sample_scale_save = torch.concat(current_sample_scale_save, dim=1)
            imputed_samples[:, i] = current_sample.detach()
            sample_scale_save[:, i] = current_sample_scale_save
        return imputed_samples, sample_scale_save

    def forward(self, batch, is_train=1):
        (
            datatype,
            observed_data,
            observed_cond_static,
            observed_cond_poi,
            observed_loc,
            observed_tp,
            idex_test
        ) = self.process_data(batch)

        side_info = self.get_side_info(observed_tp, observed_data)

        if is_train == 1:
            return self.calc_loss(observed_data, observed_cond_static, observed_cond_poi, side_info, is_train=is_train)
        
        else:
            return self.calc_loss_valid(observed_data, observed_cond_static, observed_cond_poi, side_info, is_train=is_train)

    def evaluate(self, batch, n_samples):
        (
            datatype,
            observed_data,
            observed_cond_static,
            observed_cond_poi,
            observed_loc,
            observed_tp,
            idex_test
        ) = self.process_data(batch)

        with torch.no_grad():
            side_info = self.get_side_info(observed_tp, observed_data)

            shape_base = observed_data.shape
            samples, samples_scale = self.impute(shape_base, observed_cond_static, observed_cond_poi, side_info, n_samples)
        
            return datatype, samples, samples_scale, observed_data, observed_tp, observed_loc

    def spatial_division(self, X, i_s):
        if len(X.shape) == 4:
            B, K, H, _ = X.shape

            # 空间粒度拆解
            s_division_level = 2 ** (self.spatial_scale_coarse - i_s) # 对边长几等分
            if s_division_level != H:
                # s_granularity = int(16 // (s_division_level * (2 ** (5 - spatial_scale))))
                X = X.reshape(B, K, s_division_level, int(H // s_division_level), H)
                X = X.mean(dim=3, keepdim=True).expand(-1, -1, -1, int(H // s_division_level), -1).reshape(B, K, H, H)
                X = X.reshape(B, K, H, s_division_level, int(H // s_division_level))
                X = X.mean(dim=4, keepdim=True).expand(-1, -1, -1, -1, int(H // s_division_level)).reshape(B, K, H, H)
            
        elif len(X.shape) == 5:
            B, K, L, H, _ = X.shape

            # 空间粒度拆解
            s_division_level = 2 ** (self.spatial_scale_coarse - i_s) # 对边长几等分
            if s_division_level != H:
                # s_granularity = int(16 // (s_division_level * (2 ** (5 - spatial_scale))))
                X = X.reshape(B, K, L, s_division_level, int(H // s_division_level), H)
                X = X.mean(dim=4, keepdim=True).expand(-1, -1, -1, -1, int(H // s_division_level), -1).reshape(B, K, L, H, H)
                X = X.reshape(B, K, L, H, s_division_level, int(H // s_division_level))
                X = X.mean(dim=5, keepdim=True).expand(-1, -1, -1, -1, -1, int(H // s_division_level)).reshape(B, K, L, H, H)

        return X

    def temporal_division(self, observed_data, i_t):
        B, K, L, H, _ = observed_data.shape

        # 时间粒度拆解
        if i_t == 5: # (B, K, L, H, H)
            X = observed_data.mean(dim=2, keepdim=True).expand(-1, -1, L, -1, -1).reshape(B, K, L, H, H) # (B, K, 1, H, H)
        elif i_t == 4:
            X = observed_data.reshape(B, K, 7, int(L // 7), H, H) # (B, K, 7, 24, H, H)
            X = X.mean(dim=3, keepdim=True).expand(-1, -1, -1, int(L // 7), -1, -1).reshape(B, K, L, H, H) # (B, K, L, H, H)
        else:
            X = observed_data.reshape(B, K, 42 * (2 ** (3 - i_t)), int(L // (42 * (2 ** (3 - i_t)))), H, H) # (B, K, 42, 4, H, H)
            X = X.mean(dim=3, keepdim=True).expand(-1, -1, -1, int(L // (42 * (2 ** (3 - i_t)))), -1, -1).reshape(B, K, L, H, H) # (B, K, L, H, H)
        
        return X

class CSDI_Value(CSDI_base):
    def __init__(self, args, target_dim=1):
        super(CSDI_Value, self).__init__(target_dim, args)

    def process_data(self, batch):
        datatype = batch[0]
        
        observed_data = batch[1]["observed_series"].to(self.device).float()
        observed_cond_static = batch[1]["observed_cond_static"].to(self.device).float()
        observed_cond_poi = batch[1]["observed_cond_poi"].to(self.device).float()
        observed_loc = batch[1]["observed_loc"].to(self.device).float()
        observed_tp = batch[1]["timepoints"].to(self.device).float()
        idex = batch[1]["idex"].to(self.device).int()

        # observed_data = observed_data.permute(0, 2, 1)

        return (
            datatype,
            observed_data,
            observed_cond_static,
            observed_cond_poi,
            observed_loc,
            observed_tp,
            idex
        )