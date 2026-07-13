import torch
import torch.nn as nn
import copy
from einops import rearrange
# from ddpm.network import BiFlowNet


def zero_conv3d(in_channels, out_channels):
    conv = nn.Conv3d(in_channels, out_channels, 1, padding=0)
    nn.init.zeros_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv

class ControlNet(nn.Module):
    def __init__(self, base_model, condition_channels=8):
        super().__init__()
        
        self.base_model = base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad_(False)

        self.ControlNet_downs = copy.deepcopy(base_model.downs)
        self.ControlNet_mid_block1 = copy.deepcopy(base_model.mid_block1)
        self.ControlNet_mid_spatial_attn = copy.deepcopy(base_model.mid_spatial_attn)
        self.ControlNet_mid_cross_attn = copy.deepcopy(base_model.mid_cross_attn)
        self.ControlNet_mid_block2 = copy.deepcopy(base_model.mid_block2)
        for module in [self.ControlNet_downs, self.ControlNet_mid_block1, self.ControlNet_mid_spatial_attn, 
                      self.ControlNet_mid_cross_attn, self.ControlNet_mid_block2]:
            for param in module.parameters():
                param.requires_grad_(True)
        
        self.ControlNet_downs.load_state_dict(base_model.downs.state_dict())
        self.ControlNet_mid_block1.load_state_dict(base_model.mid_block1.state_dict())
        self.ControlNet_mid_spatial_attn.load_state_dict(base_model.mid_spatial_attn.state_dict())
        self.ControlNet_mid_cross_attn.load_state_dict(base_model.mid_cross_attn.state_dict())
        self.ControlNet_mid_block2.load_state_dict(base_model.mid_block2.state_dict())
        
        self.ControlNet_downs.train()
        self.ControlNet_mid_block1.train()
        self.ControlNet_mid_spatial_attn.train()
        self.ControlNet_mid_cross_attn.train()
        self.ControlNet_mid_block2.train()


        self.condition_conv = zero_conv3d(condition_channels, base_model.dim)
    
        dims = [base_model.dim, *map(lambda m: base_model.dim * m, base_model.dim_mults)]  # [72, 72, 72, 144, 288, 576], same with base_model
        zero_conv_channels = dims[1:]  # [72, 72, 144, 288, 576]
        self.zero_convs = nn.ModuleList([zero_conv3d(channel, channel) 
                        for channel in zero_conv_channels 
                        for _ in range(2)])  # for each zero_conv_channel, hs_control has two elements which all need zero_conv
        
        self.mid_zero_conv = zero_conv3d(dims[-1], dims[-1])  # 576


    def forward(
        self, 
        x, 
        time, 
        y=None,
        condition=None,
        res=None,
    ):
        b = x.shape[0]
        ori_shape = (x.shape[2]*8, x.shape[3]*8, x.shape[4]*8) # 切Patch后的sub_volume大小是(8, 8, 8), PatchSize是4*3*4, 则latent是(32, 24, 32), 变回image是(256, 192, 256)

        # ========== [Base] Initializing x_IntraPatch (1) ==========
        x_IntraPatch = x.clone()
        p = self.base_model.sub_volume_size[0]  # (8, 8, 8)
        x_IntraPatch = x_IntraPatch.unfold(2,p,p).unfold(3,p,p).unfold(4,p,p)  # (B, 8, 4, 3, 4, 8, 8, 8)
        p1, p2, p3 = x_IntraPatch.size(2), x_IntraPatch.size(3), x_IntraPatch.size(4)  # PatchSize is 4*3*4
        x_IntraPatch = rearrange(x_IntraPatch, 'b c p1 p2 p3 d h w -> (b p1 p2 p3) c d h w')  # (B*4*3*4, 8, 8, 8, 8)

        # ========== [Base] Processing x ==========
        x_base = self.base_model.init_conv(x)  #  (B, 8, 32, 24, 32) -> (B, dim=72, 32, 24, 32)
        r = x_base.clone()  # serve as residual

        # ========== [ControlNet] Processing condition and x ==========
        condition = self.condition_conv(condition)   #  (B, 8, 32, 24, 32) -> (B, dim=72, 32, 24, 32)
        x_control = x_base + condition # (B, 72, 32, 24, 32)

        # ========== [Base] Processing t ==========
        t = self.base_model.time_mlp(time)  # time: (B,) -> t: (B, dim*4=288)
        c = t.shape[-1]
        t_DiT = t.unsqueeze(1).repeat(1,p1*p2*p3,1).view(-1,c) # (B, dim*4=288) -> (B*4*3*4, 288)
        
        # ========== [Base] Processing y ==========
        y_DiT = y.unsqueeze(1).repeat(1,p1*p2*p3,1,1).view(-1,y.shape[1],y.shape[2])  # (B, L=100, D=768) -> (B, p1*p2*p3, L, D) -> (B*p1*p2*p3, L, D)

        # ========== [Base] Processing x_IntraPatch (2) ==========
        x_IntraPatch = self.base_model.x_embedder(x_IntraPatch)  # (B*4*3*4, 8, 8, 8, 8) -> (B*4*3*4, 72, 8*8*8) -> (B*4*3*4, 512, 72) 
        x_IntraPatch = x_IntraPatch + self.base_model.pos_embed  # pos_embed: (B, 512, 72)
        # h_DiT, h_Unet, h = [], [], []
        h_DiT, h_Unet, h_Unet_control = [], [], []
        for Block, MlpLayer in self.base_model.IntraPatchFlow_input:
            x_IntraPatch = Block(x_IntraPatch, t_DiT, y_DiT) # (B*4*3*4, 512, 72) 
            h_DiT.append(x_IntraPatch)  # Recording the middile features of DiT_blocks
            Unet_feature = self.base_model.unpatchify_voxels(MlpLayer(x_IntraPatch, t_DiT)) # (B*4*3*4, 512, 72) ->MLP-> (B*4*3*4, 512, 72) -> (B*4*3*4, 72, 8, 8, 8)
            Unet_feature = rearrange(Unet_feature, '(b p) c d h w -> b p c d h w', b=b)  # (B, 4*3*4, 72, 8, 8, 8)
            Unet_feature = rearrange(Unet_feature, 'b (p1 p2 p3) c d h w -> b c (p1 d) (p2 h) (p3 w)',
                        p1=ori_shape[0]//self.base_model.vq_size, p2=ori_shape[1]//self.base_model.vq_size, p3=ori_shape[2]//self.base_model.vq_size)  # (B, 72, 32, 24, 32)
            h_Unet.append(Unet_feature) # will iterate 2 times, thus h_Unet has 2 elements
            h_Unet_control.append(Unet_feature)  # for ControlNet

        for Block in self.base_model.IntraPatchFlow_mid:
            x_IntraPatch = Block(x_IntraPatch, t_DiT, y_DiT) # (B*4*3*4, 512, 72) 

        for Block, MlpLayer in self.base_model.IntraPatchFlow_output:
            x_IntraPatch = Block(x_IntraPatch, t_DiT, y_DiT, h_DiT.pop()) # x_IntraPatch:(B*4*3*4, 512, 72), t_DiT:(B*4*3*4, 288), h_DiT:list of (B*4*3*4, 512, 72)
            Unet_feature = self.base_model.unpatchify_voxels(MlpLayer(x_IntraPatch, t_DiT)) # (B*4*3*4, 512, 72) ->MLP-> (B*4*3*4, 512, 72) -> (B*4*3*4, 72, 8, 8, 8)
            Unet_feature = rearrange(Unet_feature, '(b p) c d h w -> b p c d h w', b=b)  # (B, 4*3*4, 72, 8, 8, 8)
            Unet_feature = rearrange(Unet_feature, 'b (p1 p2 p3) c d h w -> b c (p1 d) (p2 h) (p3 w)',
                        p1=ori_shape[0]//self.base_model.vq_size, p2=ori_shape[1]//self.base_model.vq_size, p3=ori_shape[2]//self.base_model.vq_size)  # (B, 72, 32, 24, 32)
            h_Unet.append(Unet_feature) # will iterate 2 times, thus h_Unet has 2+2 elements
            h_Unet_control.append(Unet_feature)  # for ControlNet
        

        # ========== [Base] downs ==========
        hs = []
        for idx, (block1, spatial_attn1, cross_attn1, block2, spatial_attn2, cross_attn2, downsample) in enumerate(self.base_model.downs):
            if idx < self.base_model.feature_fusion: # idx=0 and 1
                x_base = x_base + h_Unet.pop(0) # x:(B, 72, 32, 24, 32)
            x_base = block1(x_base, t)  # (B, 72, 32, 24, 32)
            x_base = spatial_attn1(x_base)  # (B, 72, 32, 24, 32)
            x_base = cross_attn1(x_base, y=y)
            hs.append(x_base)
            x_base = block2(x_base, t)
            x_base = spatial_attn2(x_base)
            x_base = cross_attn2(x_base, y=y)
            hs.append(x_base)
            x_base = downsample(x_base)

        # ========== [Base] middle ==========
        x_base = self.base_model.mid_block1(x_base, t)
        x_base = self.base_model.mid_spatial_attn(x_base)
        x_base = self.base_model.mid_cross_attn(x_base, y=y)
        x_base = self.base_model.mid_block2(x_base, t)


        # ========== [ControlNet] downs ==========
        hs_control = []
        for idx, (block1, spatial_attn1, cross_attn1, block2, spatial_attn2, cross_attn2, downsample) in enumerate(self.ControlNet_downs):
            if idx < self.base_model.feature_fusion:
                x_control = x_control + h_Unet_control.pop(0)
            x_control = block1(x_control, t)
            x_control = spatial_attn1(x_control)
            x_control = cross_attn1(x_control, y=y)
            hs_control.append(x_control)
            x_control = block2(x_control, t)
            x_control = spatial_attn2(x_control)
            x_control = cross_attn2(x_control, y=y)
            hs_control.append(x_control)
            x_control = downsample(x_control)
        
        # ========== [ControlNet] middle ==========
        x_control = self.ControlNet_mid_block1(x_control, t)
        x_control = self.ControlNet_mid_spatial_attn(x_control)
        x_control = self.ControlNet_mid_cross_attn(x_control, y=y)
        x_control = self.ControlNet_mid_block2(x_control, t)
        hs_control.append(x_control)  # hs_control has different length with hs, because hs_control has the middle features
        
        # ========== [ControlNet] zero_conv ==========        
        hs_control_zero_conv = []
        for i, feat in enumerate(hs_control):  
            if i == len(hs_control) - 1:  # last one is from middle layer
                hs_control_zero_conv.append(self.mid_zero_conv(feat))
            else:
                hs_control_zero_conv.append(self.zero_convs[i](feat))

        # ========== [Base+ControlNet] ups ==========
        # hs_control和hs可以不一样长，因为hs_control可以append经过middle的，然后下面一行相加的时候pop掉!!!!!!!!!!!!!!
        # x = x_base + x_control   # 经过middle之后得到的这个x_control要过一遍zero_conv!!!!!!!!!!!!!!!!!!!!!!!!!!!
        x = x_base + hs_control_zero_conv.pop()
        for idx, (block1, spatial_attn1, cross_attn1, block2, spatial_attn2, cross_attn2, upsample) in enumerate(self.base_model.ups):
            if len(self.base_model.ups)-idx <= 2:
                x = x + h_Unet.pop(0)
            x = torch.cat((x, hs.pop() + hs_control_zero_conv.pop()), dim=1)  #hs_control剩下的和hs应该保持一样的长度，也就是没有middle的!!!!!!!!!!!!!!!!!!
            x = block1(x, t)
            x = spatial_attn1(x)
            x = cross_attn1(x, y=y)
            x = torch.cat((x, hs.pop() + hs_control_zero_conv.pop()), dim=1)
            x = block2(x, t)
            x = spatial_attn2(x)
            x = cross_attn2(x, y=y)
            x = upsample(x)

        # ========== [Base] final ==========
        x = torch.cat((x, r), dim=1)  # r is residual, thus x:(B, 72*2, 32, 24, 32)
        return self.base_model.final_conv(x)  # (B, 72*2, 32, 24, 32) -> (B, 72, 32, 24, 32) ->conv3d-> (B, 8, 32, 24, 32)
    


