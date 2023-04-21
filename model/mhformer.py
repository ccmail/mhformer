import torch
import torch.nn as nn
from einops import rearrange
from model.module.trans import Transformer as Transformer_encoder
from model.module.trans_hypothesis import Transformer as Transformer_hypothesis


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        # MHG
        self.norm_1 = nn.LayerNorm(args.frames)
        self.norm_2 = nn.LayerNorm(args.frames)
        self.norm_3 = nn.LayerNorm(args.frames)

        # 三层Transformer独立参数
        self.Transformer_encoder_1 = Transformer_encoder(
            4, args.frames, args.frames*2, length=2*args.n_joints, h=9)
        self.Transformer_encoder_2 = Transformer_encoder(
            4, args.frames, args.frames*2, length=2*args.n_joints, h=9)
        self.Transformer_encoder_3 = Transformer_encoder(
            4, args.frames, args.frames*2, length=2*args.n_joints, h=9)

        # Embedding
        # 以27帧为分界线, 两侧位置编码方式不同, 大于17帧使用1维卷积(在一串一维数组中滑动), 小于17帧会额外使用BN和ReLU
        if args.frames > 27:
            self.embedding_1 = nn.Conv1d(
                2*args.n_joints, args.channel, kernel_size=1)
            self.embedding_2 = nn.Conv1d(
                2*args.n_joints, args.channel, kernel_size=1)
            self.embedding_3 = nn.Conv1d(
                2*args.n_joints, args.channel, kernel_size=1)
        else:
            self.embedding_1 = nn.Sequential(
                nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_2 = nn.Sequential(
                nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_3 = nn.Sequential(
                nn.Conv1d(2*args.out_joints, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

        # SHR & CHI,
        # SHR是由MHSA+混合MLP组成,SHR中存在多个平级的MHSA, 将不同的坐标假设送入其中, 但是这么多假设之间没有信息交互
        # MHSA后添加了混合MLP, 以使得多个假设进行合并
        # CHI 是由MHCA+混合MLP组成, MHCA是个交叉的MHSA
        self.Transformer_hypothesis = Transformer_hypothesis(
            args.layers, args.channel, args.d_hid, length=args.frames)

        # 回归, 得出人体关节点坐标(3*out_joints, 推测为x,y,z坐标)
        self.regression = nn.Sequential(
            nn.BatchNorm1d(args.channel*3, momentum=0.1),
            nn.Conv1d(args.channel*3, 3*args.out_joints, kernel_size=1)
        )

    def forward(self, x):
        # Batchs,Frames,Joints,Channels
        B, F, J, C = x.shape
        x = rearrange(x, 'b f j c -> b (j c) f').contiguous()

        # MHG, 通过三套参数生成三套坐标, 不过不是很明白这里为什么要使用前一次的结果进行计算下一层
        x_1 = x + self.Transformer_encoder_1(self.norm_1(x))
        x_2 = x_1 + self.Transformer_encoder_2(self.norm_2(x_1))
        x_3 = x_2 + self.Transformer_encoder_3(self.norm_3(x_2))

        # Embedding,调整至下一层的输入要求, 'b (j c) f -> b f (j c)'
        x_1 = self.embedding_1(x_1).permute(0, 2, 1).contiguous()
        x_2 = self.embedding_2(x_2).permute(0, 2, 1).contiguous()
        x_3 = self.embedding_3(x_3).permute(0, 2, 1).contiguous()

        ## SHR & CHI, 整合三个变量的关系
        x = self.Transformer_hypothesis(x_1, x_2, x_3)

        # Regression, `b f (j c) -> b (j c) f`
        x = x.permute(0, 2, 1).contiguous()
        x = self.regression(x)
        x = rearrange(x, 'b (j c) f -> b f j c', j=J).contiguous()
        
        return x
