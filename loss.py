import torch
from torch import nn
from torch.nn import functional as F


class CELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits, label):
        ce_loss = self.xent_loss(logits, label)
        return {
            'loss': ce_loss,
            'ce_loss': ce_loss
        }


class CLLoss(nn.Module):

    def __init__(self, temperature=0.5, lambda_cl=0.1):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.temperature = temperature
        self.lambda_cl = lambda_cl

    def forward(self, logits, label, feature, feature_aug):
        # Cross-entropy loss
        ce_loss = self.xent_loss(logits, label)

        # Contrastive loss
        norm_feature = F.normalize(feature, dim=1)
        norm_feature_aug = F.normalize(feature_aug, dim=1)

        pos_score = (norm_feature * norm_feature_aug).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temperature)
        total_score = torch.matmul(norm_feature, norm_feature_aug.transpose(0, 1))
        total_score = torch.exp(total_score / self.temperature).sum(dim=1)

        cl_loss = -torch.log(pos_score / total_score).mean()

        if torch.isnan(ce_loss).any():
            loss = self.lambda_cl * cl_loss
        else:
            loss = ce_loss + self.lambda_cl * cl_loss

        return {
            'loss': loss,
            'ce_loss': ce_loss,
            'cl_loss': cl_loss,
        }
