"""
气象数据时空预测模型模块

本模块包含两种用于ERA5气象数据预测的深度学习模型：
1. ConvLSTMForecast - 基于卷积LSTM的序列预测模型
2. SpatioTemporalTransformer - 基于时空Transformer的先进预测模型

作者：Climate-D-S项目
版本：1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """
    卷积LSTM单元 - 结合CNN的空间特征提取和LSTM的时间序列建模能力
    
    参数:
        input_dim: 输入通道数
        hidden_dim: 隐藏层通道数
        kernel_size: 卷积核大小 (默认: 3)
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2  # 保持空间维度不变的填充
        # 输入+隐藏状态拼接后通过卷积生成4个门控信号
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)
        self.hidden_dim = hidden_dim

    def forward(self, x, h_prev, c_prev):
        """
        前向传播
        
        参数:
            x: 当前时间步输入 (B, C, H, W)
            h_prev: 上一时间步隐藏状态 (B, Hc, H, W)
            c_prev: 上一时间步细胞状态 (B, Hc, H, W)
        
        返回:
            h: 当前隐藏状态
            c: 当前细胞状态
        """
        # x: (B, C, H, W)
        # h_prev, c_prev: (B, Hc, H, W)
        # 如果是第一个时间步，初始化隐藏状态和细胞状态
        if h_prev is None:
            size_h, size_w = x.shape[-2:]
            h_prev = x.new_zeros((x.size(0), self.hidden_dim, size_h, size_w))
            c_prev = x.new_zeros((x.size(0), self.hidden_dim, size_h, size_w))
        
        # 将输入和隐藏状态拼接
        combined = torch.cat([x, h_prev], dim=1)
        # 通过卷积生成门控信号
        gates = self.conv(combined)
        # 将门控信号分成4部分：输入门、遗忘门、候选值、输出门
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        
        # 应用激活函数
        i = torch.sigmoid(i)  # 输入门
        f = torch.sigmoid(f)  # 遗忘门
        o = torch.sigmoid(o)  # 输出门
        g = torch.tanh(g)     # 候选值
        
        # 更新细胞状态：遗忘旧信息 + 添加新信息
        c = f * c_prev + i * g
        # 计算隐藏状态
        h = o * torch.tanh(c)
        
        return h, c


class ConvLSTMForecast(nn.Module):
    """
    基于多层卷积LSTM的序列预测模型
    
    输入: (B, T, C, H, W) - 批次大小, 时间步数, 通道数, 高度, 宽度
    输出: (B, C, H, W) - 下一时间步的预测
    
    参数:
        in_ch: 输入通道数（变量数）
        hidden: 隐藏层维度 (默认: 64)
        depth: LSTM层数 (默认: 2)
    """
    def __init__(self, in_ch: int, hidden: int = 64, depth: int = 2):
        super().__init__()
        self.cells = nn.ModuleList()
        ch = in_ch
        # 创建多层LSTM单元
        for _ in range(depth):
            self.cells.append(ConvLSTMCell(ch, hidden, 3))
            ch = hidden  # 后续层输入为前一层的隐藏维度
        # 输出层：将隐藏状态映射回原始通道数
        self.head = nn.Conv2d(hidden, in_ch, kernel_size=3, padding=1)

    def forward(self, x):
        """
        前向传播：处理时间序列并预测下一时间步
        
        参数:
            x: 输入序列 (B, T, C, H, W)
        
        返回:
            out: 预测结果 (B, C, H, W)
        """
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        h = c = None  # 初始化隐藏状态和细胞状态
        
        # 按时间步处理序列
        for t in range(T):
            inp = x[:, t]  # 当前时间步输入
            # 逐层处理：每层以上一层的隐藏状态作为输入
            for i, cell in enumerate(self.cells):
                h, c = cell(inp if i == 0 else h, h, c)
        
        # 使用最后一层的隐藏状态生成预测
        out = self.head(h)
        return out


class PatchEmbed(nn.Module):
    """
    图像分块嵌入模块 - 将空间图像转换为序列化的token
    
    参数:
        in_ch: 输入通道数
        embed_dim: 嵌入维度
        patch: 分块大小
    """
    def __init__(self, in_ch, embed_dim, patch):
        super().__init__()
        # 使用卷积进行分块嵌入
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        
    def forward(self, x):  # x: (B, C, H, W)
        """
        将图像分块并嵌入为token序列
        
        返回:
            tokens: 嵌入后的token序列 (B, N, D)
            hw: 分块后的空间维度 (Hp, Wp)
        """
        x = self.proj(x)  # (B, D, H', W') - 分块并嵌入
        B, D, Hp, Wp = x.shape
        # 重排维度：将空间维度展平为序列
        x = x.permute(0, 2, 3, 1).contiguous().view(B, Hp * Wp, D)  # (B, N, D)
        return x, (Hp, Wp)


class PatchUnembed(nn.Module):
    """
    分块反嵌入模块 - 将token序列还原为空间图像
    
    参数:
        embed_dim: 嵌入维度
        out_ch: 输出通道数
        patch: 分块大小
    """
    def __init__(self, embed_dim, out_ch, patch):
        super().__init__()
        # 使用转置卷积进行反嵌入
        self.deproj = nn.ConvTranspose2d(embed_dim, out_ch, kernel_size=patch, stride=patch)
        
    def forward(self, tokens, hw):  # tokens: (B, N, D), hw=(Hp,Wp)
        """
        将token序列还原为空间图像
        
        返回:
            x: 还原后的图像 (B, C_out, H, W)
        """
        B, N, D = tokens.shape
        Hp, Wp = hw
        # 将序列还原为空间网格
        x = tokens.view(B, Hp, Wp, D).permute(0, 3, 1, 2).contiguous()  # (B, D, Hp, Wp)
        # 使用转置卷积还原原始分辨率
        x = self.deproj(x)  # (B, C_out, H, W)
        return x


class MLP(nn.Module):
    """
    多层感知机模块 - Transformer中的前馈网络
    
    参数:
        dim: 输入维度
        mlp_ratio: 隐藏层维度扩展比例 (默认: 4.0)
        p: Dropout概率 (默认: 0.0)
    """
    def __init__(self, dim, mlp_ratio=4.0, p=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)  # 计算隐藏层维度
        self.fc1 = nn.Linear(dim, hidden)  # 第一层线性变换
        self.fc2 = nn.Linear(hidden, dim)  # 第二层线性变换
        self.drop = nn.Dropout(p)  # Dropout层
        
    def forward(self, x):
        """前向传播：GELU激活 + Dropout"""
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class SpatioTemporalTransformer(nn.Module):
    """
    时空Transformer模型 - 结合空间和时间注意力机制
    
    输入: (B, T, C, H, W) -> 输出: (B, C_out, H, W)
    
    参数:
        in_ch: 输入通道数
        out_ch: 输出通道数
        patch: 分块大小 (默认: 4)
        embed_dim: 嵌入维度 (默认: 256)
        depth: Transformer层数 (默认: 4)
        num_heads: 注意力头数 (默认: 8)
        t_max: 最大时间步数，用于位置编码 (默认: 24)
        dropout: Dropout概率 (默认: 0.0)
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        patch: int = 4,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        t_max: int = 24,   # 位置编码的最大时间步配置
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch = patch
        # 分块嵌入和反嵌入模块
        self.embed = PatchEmbed(in_ch, embed_dim, patch)
        self.unembed = PatchUnembed(embed_dim, out_ch, patch)

        self.depth = depth
        # 空间和时间注意力模块
        self.s_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(depth)
        ])
        self.t_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(depth)
        ])
        # 前馈网络和层归一化
        self.s_ffn = nn.ModuleList([MLP(embed_dim) for _ in range(depth)])
        self.t_ffn = nn.ModuleList([MLP(embed_dim) for _ in range(depth)])
        self.s_norm1 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(depth)])
        self.s_norm2 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(depth)])
        self.t_norm1 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(depth)])
        self.t_norm2 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(depth)])

        # 空间位置编码（延迟初始化）
        self.register_parameter('pos_spatial', None)  # 将根据N延迟初始化
        # 时间位置编码
        self.time_pos = nn.Parameter(torch.zeros(t_max + 1, embed_dim))
        nn.init.trunc_normal_(self.time_pos, std=0.02)

        # 预测模块：用于预测下一时间步
        self.pred_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.pred_ffn = MLP(embed_dim)
        self.pred_norm1 = nn.LayerNorm(embed_dim)
        self.pred_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        前向传播：处理时空序列并预测下一时间步
        
        参数:
            x: 输入序列 (B, T, C, H, W)
        
        返回:
            y: 下一时间步预测 (B, C_out, H, W)
        """
        # x: (B, T, C, H, W) -> y_{T+1}: (B, C_out, H, W)
        B, T, C, H, W = x.shape
        # 检查空间维度是否可分块
        assert H % self.patch == 0 and W % self.patch == 0, "H/W必须能被分块大小整除"

        # 1. 对每一帧进行分块嵌入
        frames = []
        hw = None
        for t in range(T):
            tok, hw = self.embed(x[:, t])        # (B, N, D)
            frames.append(tok)
        tokens = torch.stack(frames, dim=1)      # (B, T, N, D)
        B, T, N, D = tokens.shape

        # 2. 空间位置编码（根据token数量延迟初始化）
        if self.pos_spatial is None or self.pos_spatial.shape[1] != N:
            self.pos_spatial = nn.Parameter(torch.zeros(1, N, D, device=tokens.device))
            nn.init.trunc_normal_(self.pos_spatial, std=0.02)

        # 3. 时间位置编码
        time_pos = self.time_pos[:T].unsqueeze(0).unsqueeze(2)          # (1, T, 1, D)
        # 添加位置编码到token
        tokens = tokens + self.pos_spatial.unsqueeze(1) + time_pos      # (B, T, N, D)

        # 4. 交替的空间/时间注意力块
        for l in range(self.depth):
            # 空间自注意力（每帧内部）
            z = tokens.view(B * T, N, D)  # 展平批次和时间维度
            q = self.s_norm1[l](z)
            z2, _ = self.s_attn[l](q, q, q, need_weights=False)
            z = z + z2  # 残差连接
            z = z + self.s_ffn[l](self.s_norm2[l](z))  # 前馈网络 + 残差
            tokens = z.view(B, T, N, D)  # 恢复原始形状

            # 时间自注意力（每个空间位置的时间序列）
            z = tokens.permute(0, 2, 1, 3).contiguous().view(B * N, T, D)
            q = self.t_norm1[l](z)
            z2, _ = self.t_attn[l](q, q, q, need_weights=False)
            z = z + z2
            z = z + self.t_ffn[l](self.t_norm2[l](z))
            tokens = z.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()

        # 5. 通过因果注意力预测下一时间步的token
        seq = tokens.permute(0, 2, 1, 3).contiguous().view(B * N, T, D)    # (B*N, T, D)
        # 创建下一时间步的查询
        next_query = torch.zeros(B * N, 1, D, device=seq.device)
        next_pos = self.time_pos[T].view(1, 1, D).to(seq.device)
        next_query = next_query + next_pos
        # 拼接序列和查询
        seq_plus = torch.cat([seq, next_query], dim=1)  # (B*N, T+1, D)

        # 创建因果掩码（防止未来信息泄露）
        L = T + 1
        causal_mask = torch.full((L, L), float("-inf"), device=seq.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)  # 上三角矩阵

        # 预测注意力
        h = self.pred_norm1(seq_plus)
        h2, _ = self.pred_attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        h = seq_plus + h2
        h = h + self.pred_ffn(self.pred_norm2(h))
        # 提取下一时间步的token
        next_tokens = h[:, -1]                               # (B*N, D)
        next_tokens = next_tokens.view(B, N, D)              # (B, N, D)

        # 6. 反嵌入得到预测结果
        y = self.unembed(next_tokens, hw)                    # (B, out_ch, H, W)
        return y

