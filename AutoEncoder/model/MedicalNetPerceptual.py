import torch
import torch.nn as nn
import os 
def mednet_norm(input):
    mean = input.mean() 
    std = input.std()
    return (input - mean) / std
def mednet_norm_feature(x , eps  = 1e-7):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)
def spatial_average(x, keepdim = True) -> torch.Tensor:
    return x.mean([2, 3], keepdim=keepdim)
class MedicalNetPerceptual(nn.Module):
    def __init__(
        self, net_path = None, verbose = True, channel_wise = False):
        super().__init__()
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        net = "medicalnet_resnet10_23datasets"
        if net_path is not None:
            self.model = torch.hub.load(repo_or_dir=net_path, model=net, trust_repo=True, source='local',model_dir = os.path.dirname(os.path.abspath(__file__))+'/../../warvito_MedicalNet-models_main/medicalnet' )
        else:
            self.model = torch.hub.load("warvito/MedicalNet-models", model=net, verbose=verbose)
        self.eval()

        self.channel_wise = channel_wise

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        input = mednet_norm(input)
        target = mednet_norm(target)

        # Get model outputs
        feats_per_ch = 0
        for ch_idx in range(input.shape[1]):
            input_channel = input[:, ch_idx, ...].unsqueeze(1)
            target_channel = target[:, ch_idx, ...].unsqueeze(1)

            if ch_idx == 0:
                outs_input = self.model.forward(input_channel)
                outs_target = self.model.forward(target_channel)
                feats_per_ch = outs_input.shape[1]
            else:
                outs_input = torch.cat([outs_input, self.model.forward(input_channel)], dim=1)
                outs_target = torch.cat([outs_target, self.model.forward(target_channel)], dim=1)


        feats_input = mednet_norm_feature(outs_input)
        feats_target = mednet_norm_feature(outs_target)

        feats_diff: torch.Tensor = (feats_input - feats_target) ** 2
        if self.channel_wise:
            results = torch.zeros(
                feats_diff.shape[0], input.shape[1], feats_diff.shape[2], feats_diff.shape[3], feats_diff.shape[4]
            )
            for i in range(input.shape[1]):
                l_idx = i * feats_per_ch
                r_idx = (i + 1) * feats_per_ch
                results[:, i, ...] = feats_diff[:, l_idx : i + r_idx, ...].sum(dim=1)
        else:
            results = feats_diff.sum(dim=1, keepdim=True)

        results = results.mean([2, 3, 4], keepdim=True)

        return results