

import math
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F#
from einops import rearrange
from torch.optim.optimizer import Optimizer
from AutoEncoder.utils import shift_dim, adopt_weight
from AutoEncoder.model.lpips import LPIPS
from AutoEncoder.model.codebook import Codebook
from AutoEncoder.model.MedicalNetPerceptual import MedicalNetPerceptual
from pytorch_lightning.callbacks import BaseFinetuning
from einops_exts import rearrange_many
from os.path import join 
import os 
import numpy as np

def silu(x):
    return x*torch.sigmoid(x)

class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return silu(x)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def non_saturating_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class Perceptual_Loss(nn.Module):
    def __init__(self, is_3d: bool = True, sample_ratio: float = 0.2):
        super().__init__()
        self.is_3d = is_3d 
        self.sample_ratio = sample_ratio
        if is_3d:
            
            self.perceptual_model = MedicalNetPerceptual(net_path=os.path.dirname(os.path.abspath(__file__))+'/../../warvito_MedicalNet-models_main').eval()
        else:
            self.perceptual_model = LPIPS().eval()
    def forward(self, input:torch.Tensor, target: torch.Tensor):
        if self.is_3d:
            p_loss =  torch.mean(self.perceptual_model(input , target))
        else:
            B,C,D,H,W = input.shape

            input_slices_xy = input.permute((0,2,1,3,4)).contiguous()
            input_slices_xy = input_slices_xy.view(-1, C, H, W)
            indices_xy = torch.randperm(input_slices_xy.shape[0])[: int(input_slices_xy.shape[0] * self.sample_ratio)].to(input.device)
            input_slices_xy = torch.index_select(input_slices_xy, dim=0, index=indices_xy)
            target_slices_xy = target.permute((0,2,1,3,4)).contiguous()
            target_slices_xy = target_slices_xy.view(-1, C, H, W)
            target_slices_xy = torch.index_select(target_slices_xy, dim=0, index=indices_xy)

            input_slices_xz = input.permute((0,3,1,2,4)).contiguous()
            input_slices_xz = input_slices_xz.view(-1, C, D, W)
            indices_xz = torch.randperm(input_slices_xz.shape[0])[: int(input_slices_xz.shape[0] * self.sample_ratio)].to(input.device)
            input_slices_xz = torch.index_select(input_slices_xz, dim=0, index=indices_xz)
            target_slices_xz = target.permute((0,3,1,2,4)).contiguous()
            target_slices_xz = target_slices_xz.view(-1, C, D, W)
            target_slices_xz = torch.index_select(target_slices_xz, dim=0, index=indices_xz)

            input_slices_yz = input.permute((0,4,1,2,3)).contiguous()
            input_slices_yz = input_slices_yz.view(-1, C, D, H)
            indices_yz = torch.randperm(input_slices_yz.shape[0])[: int(input_slices_yz.shape[0] * self.sample_ratio)].to(input.device)
            input_slices_yz = torch.index_select(input_slices_yz, dim=0, index=indices_yz)
            target_slices_yz = target.permute((0,4,1,2,3)).contiguous()
            target_slices_yz = target_slices_yz.view(-1, C, D, H)
            target_slices_yz = torch.index_select(target_slices_yz, dim=0, index=indices_yz)
            p_loss = torch.mean(self.perceptual_model(input_slices_xy,target_slices_xy)) + torch.mean(self.perceptual_model(input_slices_xz,target_slices_xz)) + torch.mean(self.perceptual_model(input_slices_yz,target_slices_yz))
        return p_loss



class volumeAE(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.automatic_optimization = False  
        self.cfg = cfg
        self.embedding_dim = cfg.model.embedding_dim
        self.n_codes = cfg.model.n_codes
        self.patch_size = cfg.dataset.patch_size

        self.encoder = Encoder(cfg.model.n_hiddens, cfg.model.downsample,
                               cfg.dataset.image_channels, cfg.model.norm_type,
                               cfg.model.num_groups,cfg.model.embedding_dim,
                               )
        self.decoder = Decoder(
            cfg.model.n_hiddens, cfg.model.downsample, cfg.dataset.image_channels, cfg.model.norm_type, cfg.model.num_groups,cfg.model.embedding_dim)
        # self.enc_out_ch = self.encoder.out_channels
        self.pre_vq_conv = nn.Conv3d(cfg.model.embedding_dim, cfg.model.embedding_dim, 1, 1)
        self.post_vq_conv = nn.Conv3d(cfg.model.embedding_dim, cfg.model.embedding_dim, 1, 1)

        # need_init = False if cfg.model.resume_from_checkpoint else True
        # self.codebook = Codebook(cfg.model.n_codes, cfg.model.embedding_dim,
        #                          no_random_restart=cfg.model.no_random_restart, restart_thres=cfg.model.restart_thres, 
        #                          need_init=need_init)
        self.codebook = Codebook(cfg.model.n_codes, cfg.model.embedding_dim,
                                 no_random_restart=cfg.model.no_random_restart, restart_thres=cfg.model.restart_thres, 
                                 need_init=False)
        
        # self.perceptual_loss = Perceptual_Loss(is_3d=cfg.model.perceptual_3d).eval()

        # self.l1_weight = cfg.model.l1_weight   # weight of loss
        # self.perceptual_weight = cfg.model.perceptual_weight    # weight of loss

        # if not hasattr(self.cfg.model, 'need_discriminator'):
        #     self.cfg.model.need_discriminator = False
        # if self.cfg.model.need_discriminator:
        #     self.volume_discriminator = NLayerDiscriminator3D(
        #         cfg.dataset.image_channels, cfg.model.disc_channels, cfg.model.disc_layers, norm_layer=nn.BatchNorm3d)
        #     if cfg.model.disc_loss_type == 'vanilla':
        #         self.disc_loss = vanilla_d_loss
        #     elif cfg.model.disc_loss_type == 'hinge':
        #         self.disc_loss = hinge_d_loss
        #     self.gan_feat_weight = cfg.model.gan_feat_weight    # weight of loss
        #     self.volume_gan_weight = cfg.model.volume_gan_weight   # weight of loss

        self.save_hyperparameters()


    def forward(self, x, optimizer_idx=None, log_volume=False):
        B, C, D, H, W = x.shape ##ｂ　ｃ　ｚ　ｘ　ｙ

        x_input = x

        z = self.pre_vq_conv(self.encoder(x_input)) 
        vq_output = self.codebook(z)
        embeddings = vq_output['embeddings']

        x_recon = self.decoder(self.post_vq_conv(embeddings))

        recon_loss = self.l1_weight * F.l1_loss(x_recon, x)

        if log_volume:  # for saving during training and validation
            return x, x_recon

        if optimizer_idx is None:  # infer (validation or test) if optimizer_idx is None
            perceptual_loss = self.perceptual_weight * self.perceptual_loss(x, x_recon)
            return recon_loss, x_recon, vq_output, perceptual_loss

        elif optimizer_idx == 0:
            perceptual_loss = self.perceptual_weight * self.perceptual_loss(x, x_recon)

            # To train the generator, aimed to mislead discriminator to believe this is real volume
            if self.cfg.model.need_discriminator:
            # if self.cfg.model.need_discriminator and self.volume_gan_weight > 0:
                logits_volume_fake, pred_volume_fake = self.volume_discriminator(x_recon)
                g_loss = -torch.mean(logits_volume_fake)   # hope to be closer to 1 rather than 0
                aeloss = self.volume_gan_weight * g_loss

                logits_volume_real, pred_volume_real = self.volume_discriminator(x)
                volume_gan_feat_loss = 0
                for i in range(len(pred_volume_fake)-1):
                    volume_gan_feat_loss += F.l1_loss(pred_volume_fake[i], pred_volume_real[i].detach())   # hope the intermidiate features of predicted volume to be same with that of real volume
                gan_feat_loss = self.gan_feat_weight * volume_gan_feat_loss
                
            else:
                aeloss = torch.tensor(0.0, requires_grad=True)
                gan_feat_loss =  torch.tensor(0.0, requires_grad=True)
                
            self.log("train/recon_loss", recon_loss, 
                     prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/perceptual_loss", perceptual_loss,
                     prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/commitment_loss", vq_output['commitment_loss'],
                     prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log('train/perplexity', vq_output['perplexity'],
                     prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/gan_feat_loss", gan_feat_loss,
                     prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/aeloss", aeloss, 
                     prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return recon_loss, x_recon, vq_output, aeloss, perceptual_loss, gan_feat_loss

        elif optimizer_idx == 1:
            # To train the discriminator, aimed to make discriminator able to discriminate the predicted volume is fake
            logits_volume_real, _ = self.volume_discriminator(x.detach())
            logits_volume_fake, _= self.volume_discriminator(x_recon.detach())
            d_volume_loss = self.disc_loss(logits_volume_real, logits_volume_fake)
            discloss = self.volume_gan_weight * d_volume_loss

            self.log("train/logits_volume_real", logits_volume_real.mean().detach(),
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/logits_volume_fake", logits_volume_fake.mean().detach(),
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/discloss", discloss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            return discloss

    def encode(self, x):
        B, C, D, H, W = x.shape
        z = self.pre_vq_conv(self.encoder(x)) 
        vq_output = self.codebook(z)
        embeddings = vq_output['embeddings']
        return embeddings
    
    # def decode(self, z):
    #     x_recon = self.decoder(self.post_vq_conv(z))
    #     return x_recon
    def decode(self, z, quantize=True):
        if quantize:
            vq_output = self.codebook(z)
            latent = vq_output['encodings']  # (B, 32, 24, 32)
        h = F.embedding(latent, self.codebook.embeddings)  # (B, 32, 24, 32, 8)
        h = self.post_vq_conv(shift_dim(h, -1, 1))   # (B, 32, 24, 32, 8) ->shift-> (B, 8, 32, 24, 32)
        h = self.decoder(h)
        return h


    def training_step(self, batch, batch_idx):
        x = batch['data']
        
        if self.cfg.model.need_discriminator:   # iteratively train if needing discriminator
            optimizers = self.optimizers()  # len(optimizers)=2
            optimizer_idx = batch_idx % len(optimizers)
            opt = optimizers[optimizer_idx]
        else:
            optimizer_idx = 0
            opt = self.optimizers()  # len(optimizers)=1
        # print("optimizer_idx", optimizer_idx)
        
        opt.zero_grad()

        if optimizer_idx == 0:
            recon_loss, _, vq_output, aeloss, perceptual_loss, gan_feat_loss = self.forward(x, optimizer_idx)
            commitment_loss = vq_output['commitment_loss']
            loss = recon_loss + commitment_loss + aeloss + perceptual_loss + gan_feat_loss

        elif optimizer_idx == 1:
            discloss = self.forward(x, optimizer_idx)
            loss = discloss
        
        self.manual_backward(loss)
        opt.step()
        return loss
        

    def validation_step(self, batch, batch_idx):
        x = batch['data']
        recon_loss, _, vq_output, perceptual_loss = self.forward(x, optimizer_idx=None)
        self.log('val/recon_loss', recon_loss, 
                 prog_bar=True, sync_dist=True)
        self.log('val/perceptual_loss', perceptual_loss, 
                 prog_bar=True, sync_dist=True)
        self.log('val/perplexity', vq_output['perplexity'], 
                 prog_bar=True, sync_dist=True)
        self.log('val/commitment_loss', vq_output['commitment_loss'], 
                 prog_bar=True, sync_dist=True)


    def configure_optimizers(self):
        lr = self.cfg.model.lr

        if self.cfg.model.need_discriminator:
            opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                    list(self.decoder.parameters()) +
                                    list(self.pre_vq_conv.parameters()) +
                                    list(self.post_vq_conv.parameters()) +
                                    list(self.codebook.parameters()),
                                    lr=lr, betas=(0.5, 0.9))   # use customized GAN settings
            opt_disc = torch.optim.Adam(list(self.volume_discriminator.parameters()),
                                        lr=lr, betas=(0.5, 0.9))   # use customized GAN settings
            return [opt_ae, opt_disc]

        else:
            opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                    list(self.decoder.parameters()) +
                                    list(self.pre_vq_conv.parameters()) +
                                    list(self.post_vq_conv.parameters()) +
                                    list(self.codebook.parameters()),
                                    lr=lr, betas=(0.9, 0.999))   # use default settings
            return opt_ae



    def log_volumes(self, batch, **kwargs):
        log = dict()
        x = batch['data']
        # x, x_rec = self(x, log_volume=True, val=(kwargs['split']=='val'))
        x, x_rec = self(x, log_volume=True)
        log["inputs"] = x
        log["reconstructions"] = x_rec

        return log





def Normalize(in_channels, norm_type='group', num_groups=32):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        # TODO Changed num_groups from 32 to 8
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return torch.nn.SyncBatchNorm(in_channels)

class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32,norm_type='group',num_groups=32):
        super().__init__()
        self.norm = Normalize(dim, norm_type=norm_type, num_groups=num_groups)
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads # 256
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, z, h, w = x.shape
        x_norm = self.norm(x)
        x_norm = rearrange(x_norm,'b c z x y -> b (z x y) c').contiguous()
        qkv = self.to_qkv(x_norm).chunk(3, dim=2)
        q, k, v = rearrange_many(
            qkv, 'b d (h c) -> b h d c ', h=self.heads)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale, dropout_p=0.0, is_causal=False)
        out = rearrange(out, 'b h (z x y) c -> b (h c) z x y ',z = z, x = h ,y = w ).contiguous()
        out = self.to_out(out)
        return out+x


class Encoder(nn.Module):
    def __init__(self, n_hiddens, downsample, image_channel=1, norm_type='group', num_groups=32 , embedding_dim = 8):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.conv_blocks = nn.ModuleList()
        max_ds = n_times_downsample.max()
        self.embedding_dim = embedding_dim
        self.conv_first = nn.Conv3d(
            image_channel , n_hiddens, kernel_size=3, stride=1, padding=1
        )
    
        channels = [n_hiddens * 2 ** i for i in range(max_ds)]
        channels = channels +[channels[-1]]
        in_channels = channels[0]
        for i in range(max_ds + 1):
            block = nn.Module()
            if i != 0 :
                in_channels = channels[i-1]
            out_channels = channels[i]
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            if in_channels!= out_channels:
                block.res1 = ResBlockXY(in_channels , out_channels,norm_type=norm_type, num_groups=num_groups )
            else:
                block.res1 = ResBlockX(in_channels , out_channels,norm_type=norm_type, num_groups=num_groups)

            block.res2  = ResBlockX(out_channels , out_channels, norm_type=norm_type, num_groups=num_groups)
            if i != max_ds:
                block.down = nn.Conv3d(out_channels,out_channels,kernel_size=(4, 4, 4),stride=stride,padding=1)
            else:
                block.down = nn.Identity()
            self.conv_blocks.append(block)
            n_times_downsample -= 1
        self.mid_block = nn.Module()
        self.mid_block.res1 = ResBlockX(out_channels , out_channels,norm_type=norm_type, num_groups=num_groups)
        self.mid_block.attn = AttentionBlock(out_channels, heads=4,norm_type=norm_type,num_groups=num_groups)
        self.mid_block.res2 = ResBlockX(out_channels , out_channels,norm_type=norm_type, num_groups=num_groups)
        self.final_block = nn.Sequential(
            Normalize(out_channels, norm_type, num_groups=num_groups),
            SiLU(),
            nn.Conv3d(out_channels, self.embedding_dim, 3 , 1 ,1)
        )

        self.out_channels = out_channels
    def forward(self, x):
        h = self.conv_first(x)
        for idx , block in enumerate(self.conv_blocks):
            h = block.res1(h)
            h = block.res2(h)
            h = block.down(h)
        h = self.mid_block.res1(h)
        h = self.mid_block.attn(h)
        h = self.mid_block.res2(h)
        h = self.final_block(h)
        return h

class Decoder(nn.Module):
    def __init__(self, n_hiddens, upsample, image_channel, norm_type='group', num_groups=32 , embedding_dim=8 ):
        super().__init__()

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()
        channels = [n_hiddens * 2 ** i for i in range(max_us)]
        channels = channels+[channels[-1]]
        channels.reverse()
        self.embedding_dim = embedding_dim
        self.conv_first = nn.Conv3d(self.embedding_dim, channels[0],3,1,1)
        self.mid_block = nn.Module()
        self.mid_block.res1 = ResBlockX(channels[0] , channels[0],norm_type=norm_type, num_groups=num_groups)
        self.mid_block.attn = AttentionBlock(channels[0], heads=4,norm_type=norm_type,num_groups=num_groups)
        self.mid_block.res2 = ResBlockX(channels[0] , channels[0],norm_type=norm_type, num_groups=num_groups)
        self.conv_blocks = nn.ModuleList()
        in_channels = channels[0]
        for i in range(max_us + 1):
            block = nn.Module()
            if i != 0:
                in_channels = channels[i-1]
            out_channels = channels[i]
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            if in_channels != out_channels:
                block.res1 = ResBlockXY(in_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            else:
                block.res1 = ResBlockX(in_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            block.res2 = ResBlockX(out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            if i != max_us :
                block.up = Upsample(out_channels)
            else:
                block.up = nn.Identity(out_channels)
            self.conv_blocks.append(block)
            n_times_upsample -= 1

        self.final_block = nn.Sequential(
            Normalize(out_channels, norm_type, num_groups=num_groups),
            SiLU(),
            nn.Conv3d(out_channels, image_channel, 3 , 1 ,1)
        )

    def forward(self, x):
        h = self.conv_first(x)
        h = self.mid_block.res1(h)
        h = self.mid_block.attn(h)
        h = self.mid_block.res2(h)
        for i, block in enumerate(self.conv_blocks):
            h = block.res1(h)
            h = block.res2(h)
            h = block.up(h)
        h = self.final_block(h)
        return h



class ResBlockX(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0, norm_type='group', num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        
        self.norm1 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.norm2 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3 , padding=1, stride=1)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)

        return x+h


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_trans = nn.ConvTranspose3d(in_channels, in_channels, 4,
                                        stride=2, padding=1)

    def forward(self, x):
        x = self.conv_trans(x)
        return x

class ResBlockXY(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0, norm_type='group', num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        
        self.resConv = nn.Conv3d(in_channels, out_channels, (1, 1, 1)) 
        self.norm1 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.conv1 = nn.Conv3d(in_channels , out_channels, kernel_size=3, padding=1, stride=1)
        self.norm2 = Normalize(out_channels, norm_type, num_groups=num_groups)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels , out_channels, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        residual = self.resConv(x)
        h = self.norm1(x)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)

        return h+residual


class ResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer = nn.SyncBatchNorm):
        super().__init__()

        self.leakyRELU = nn.LeakyReLU()
        self.pool = nn.AvgPool3d((2, 2, 2))
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.resConv = nn.Conv3d(in_channels, out_channels, 1)
        
    def forward(self, x):
        residual = self.resConv(self.pool(x))
        x = self.conv1(x)
        x = self.leakyRELU(x)

        x = self.pool(x) 

        x = self.conv2(x)
        x = self.leakyRELU(x)

        return (x+residual)/math.sqrt(2)







class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator3D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv3d(input_nc, ndf, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), None
        
class AE_finetuning(BaseFinetuning):
    def freeze_before_training(self, pl_module: pl.LightningModule):
        pl_module.stage = 2
        self.freeze(pl_module.encoder)
        self.freeze(pl_module.pre_vq_conv)
        self.freeze(pl_module.codebook)
        # self.freeze(pl_module.volume_discriminator)
    def finetune_function(self, pl_module: pl.LightningModule, epoch: int, optimizer: Optimizer) -> None:
        pl_module.encoder.eval()
        pl_module.codebook.eval()
        pl_module.pre_vq_conv.eval()
        # pl_module.volume_discriminator.eval()
        