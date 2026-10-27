import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.mish(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.mish(out + residual)

class CadenceNet(nn.Module):
    """
    9-Channel 2D Spectrogram Convolutional Neural Network
    Channels 0..5: Normalized panels (p0, p1, p2, p3, p4, p5)
    Channels 6..8: Differential ON-OFF panels (p0-p1, p2-p3, p4-p5)
    Input shape: (B, 9, 16, 128)
    """
    def __init__(self, num_classes=9):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=(3, 5), padding=(1, 2), bias=False),
            nn.BatchNorm2d(32),
            nn.Mish(),
            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=(2, 2)) # -> (B, 64, 8, 64)
        )
        
        # Stage 1
        self.stage1 = nn.Sequential(
            ResBlock2D(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=(2, 2)) # -> (B, 128, 4, 32)
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            ResBlock2D(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=(2, 2)) # -> (B, 256, 2, 16)
        )
        
        # Dual Pooling (Avg + Max captures both smooth patterns and sharp spikes!)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.BatchNorm1d(128),
            nn.Mish(),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x: (B, 9, 16, 128)
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        
        avg_feat = self.avg_pool(feat).flatten(1)
        max_feat = self.max_pool(feat).flatten(1)
        pooled = torch.cat([avg_feat, max_feat], dim=1) # (B, 512)
        
        out = self.classifier(pooled) # (B, 9)
        return out

def prepare_cadence_tensor(cad_raw):
    """
    Converts raw uint8 cadence (B, 6, 16, 128) or (6, 16, 128) into 9-channel standardized tensor.
    Channels: [p0, p1, p2, p3, p4, p5, p0-p1, p2-p3, p4-p5]
    """
    is_batch = (cad_raw.ndim == 4)
    if not is_batch:
        cad_raw = cad_raw[None, ...] # (1, 6, 16, 128)
        
    B, P, R, C = cad_raw.shape
    x = cad_raw.astype(np.float32)
    # Mask dropped rows (0)
    x[x == 0] = np.nan
    # Zero out center-channel spike (channels 63..65)
    x[:, :, :, 63:66] = np.nan
    
    # Subtract panel median
    p_med = np.nanmedian(x, axis=(2, 3), keepdims=True)
    centered = x - p_med
    
    # Standardize by robust MAD scale per cadence
    abs_dev = np.abs(centered)
    c_mad = np.nanmedian(abs_dev, axis=(1, 2, 3), keepdims=True) * 1.4826
    norm = centered / (c_mad + 1e-4)
    norm = np.nan_to_num(norm, nan=0.0) # (B, 6, 16, 128)
    
    # Construct differential ON-OFF channels:
    # ON: 0, 2, 4 | OFF: 1, 3, 5
    d0 = norm[:, 0] - norm[:, 1] # (B, 16, 128)
    d1 = norm[:, 2] - norm[:, 3]
    d2 = norm[:, 4] - norm[:, 5]
    diffs = np.stack([d0, d1, d2], axis=1) # (B, 3, 16, 128)
    
    # Stack into 9 channels
    t9 = np.concatenate([norm, diffs], axis=1) # (B, 9, 16, 128)
    
    if not is_batch:
        return t9[0]
    return t9
