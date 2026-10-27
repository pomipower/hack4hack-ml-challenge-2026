import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import os
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Squeeze-and-Excitation Attention
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.Mish(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w

class SEResBlock2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)
        
    def forward(self, x):
        residual = x
        out = F.mish(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.mish(out + residual)

class CadenceNetV4(nn.Module):
    """
    Telescope-Invariant 9-Channel 2D Spectrogram Network
    Channels 0..5: Invariant normalized linear-power spectrograms
    Channels 6..8: Strict differential ON-OFF spectrograms (p0-p1, p2-p3, p4-p5)
    Input shape: (B, 9, 16, 128)
    """
    def __init__(self, num_classes=9):
        super().__init__()
        # Stem: asymmetric kernel (3 time rows x 5 freq channels)
        self.stem = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=(3, 5), padding=(1, 2), bias=False),
            nn.BatchNorm2d(32),
            nn.Mish(),
            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=(2, 2)) # -> (B, 64, 8, 64)
        )
        
        # Stage 1 with Squeeze-and-Excitation attention
        self.stage1 = nn.Sequential(
            SEResBlock2D(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=(2, 2)) # -> (B, 128, 4, 32)
        )
        
        # Stage 2 with Squeeze-and-Excitation attention
        self.stage2 = nn.Sequential(
            SEResBlock2D(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=(2, 2)) # -> (B, 256, 2, 16)
        )
        
        # Dual Pooling: captures both smooth features and sharp spikes
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.BatchNorm1d(128),
            nn.Mish(),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        
        avg_f = self.avg_pool(feat).flatten(1)
        max_f = self.max_pool(feat).flatten(1)
        pooled = torch.cat([avg_f, max_f], dim=1) # (B, 512)
        return self.classifier(pooled)

class CadenceDatasetV4(Dataset):
    def __init__(self, tensors, labels=None, is_train=False):
        self.tensors = tensors
        self.labels = labels
        self.is_train = is_train
        
    def __len__(self):
        return len(self.tensors)
        
    def __getitem__(self, idx):
        x = torch.from_numpy(self.tensors[idx].astype(np.float32)) # (9, 16, 128)
        
        if self.is_train:
            # 1. Random frequency scaling (0.72x to 1.28x) - bridges GBT (2.79Hz) and PKS (3.81Hz)!
            if random.random() < 0.6:
                scale = random.uniform(0.72, 1.28)
                C, H, W = x.shape
                new_W = int(round(W * scale))
                # Add batch dim for interpolate
                scaled = F.interpolate(x.unsqueeze(0), size=(H, new_W), mode='bilinear', align_corners=False).squeeze(0)
                if new_W < W:
                    pad_left = (W - new_W) // 2
                    pad_right = W - new_W - pad_left
                    x = F.pad(scaled, (pad_left, pad_right, 0, 0), value=0.0)
                else:
                    crop_start = (new_W - W) // 2
                    x = scaled[:, :, crop_start:crop_start + W]
                    
            # 2. Random horizontal flip (reverses drift direction - 100% physically valid!)
            if random.random() < 0.5:
                x = torch.flip(x, dims=[-1])
                
            # 3. Random frequency jitter (small shift +/- 8 channels)
            if random.random() < 0.5:
                shift = random.randint(-8, 8)
                if shift != 0:
                    x = torch.roll(x, shifts=shift, dims=-1)
                    if shift > 0:
                        x[:, :, :shift] = 0.0
                    else:
                        x[:, :, shift:] = 0.0
                        
            # 4. Small noise scale perturbation
            if random.random() < 0.4:
                noise_scale = random.uniform(0.85, 1.15)
                x = x * noise_scale
                
        if self.labels is not None:
            return x, torch.tensor(self.labels[idx], dtype=torch.long)
        return x

def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(x)
            loss = criterion(logits, y)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate_val(model, loader, device):
    model.eval()
    all_probs = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            # TTA: original + horizontal flip
            logits_orig = model(x)
            logits_flip = model(torch.flip(x, dims=[-1]))
            probs = 0.5 * torch.softmax(logits_orig, dim=-1) + 0.5 * torch.softmax(logits_flip, dim=-1)
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)

@torch.no_grad()
def predict_test_tta(model, loader, device):
    model.eval()
    all_probs = []
    for x in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            # TTA: original + horizontal flip
            logits_orig = model(x)
            logits_flip = model(torch.flip(x, dims=[-1]))
            probs = 0.5 * torch.softmax(logits_orig, dim=-1) + 0.5 * torch.softmax(logits_flip, dim=-1)
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)

def main():
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('===================================================================')
    print(f'   Training CadenceNetV4 on {device} ({torch.cuda.get_device_name(0)})   ')
    print('   Features: Linear Power Flattening + SE-Attention + Drift Scaling  ')
    print('===================================================================')
    
    train_meta = pd.read_csv('train_meta.csv')
    test_meta = pd.read_csv('test_meta.csv')
    sample_sub = pd.read_csv('sample_submission.csv')
    
    classes = np.sort(train_meta['label'].unique())
    label_to_code = {c: i for i, c in enumerate(classes)}
    code_to_label = {i: c for i, c in enumerate(classes)}
    y = train_meta['label'].map(label_to_code).values
    
    print('Loading invariant v4 tensors...')
    t0 = time.time()
    X_train_tensors = np.load('train_tensors_v4.npy')
    X_test_tensors = np.load('test_tensors_v4.npy')
    print(f'Loaded train {X_train_tensors.shape} and test {X_test_tensors.shape} in {time.time() - t0:.1f}s')
    
    # Square-root class weights for Macro-F1
    counts = np.bincount(y)
    weights = (len(y) / (len(counts) * counts)) ** 0.5
    weights = weights / np.mean(weights)
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    
    test_dataset = CadenceDatasetV4(X_test_tensors, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros((len(y), len(classes)), dtype=np.float32)
    test_probs = np.zeros((len(test_meta), len(classes)), dtype=np.float32)
    
    epochs = 14
    batch_size = 64
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_tensors, y)):
        t_fold = time.time()
        print(f'\n--- Fold {fold + 1} / 5 ---')
        
        tr_ds = CadenceDatasetV4(X_train_tensors[tr_idx], y[tr_idx], is_train=True)
        va_ds = CadenceDatasetV4(X_train_tensors[va_idx], y[va_idx], is_train=False)
        
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        va_loader = DataLoader(va_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
        
        model = CadenceNetV4(num_classes=len(classes)).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(tr_loader), eta_min=1e-5)
        scaler = torch.amp.GradScaler('cuda')
        
        best_va_f1 = 0.0
        best_va_probs = None
        
        for ep in range(epochs):
            t_ep = time.time()
            loss = train_one_epoch(model, tr_loader, optimizer, scheduler, criterion, scaler, device)
            va_probs = evaluate_val(model, va_loader, device)
            va_preds = np.argmax(va_probs, axis=1)
            va_f1 = f1_score(y[va_idx], va_preds, average='macro')
            
            if va_f1 > best_va_f1:
                best_va_f1 = va_f1
                best_va_probs = va_probs.copy()
                torch.save(model.state_dict(), f'cadence_v4_fold{fold}.pt')
                
            print(f'  Epoch {ep+1:2d}/{epochs:2d} ({time.time() - t_ep:.1f}s) | Loss: {loss:.4f} | Val Macro-F1 (TTA): {va_f1:.4f} {"*" if va_f1 == best_va_f1 else ""}')
            
        print(f'Fold {fold + 1} Best Macro-F1: {best_va_f1:.4f} in {time.time() - t_fold:.1f}s')
        oof_probs[va_idx] = best_va_probs
        
        # Load best fold model for test prediction with TTA
        model.load_state_dict(torch.load(f'cadence_v4_fold{fold}.pt'))
        fold_test_probs = predict_test_tta(model, test_loader, device)
        test_probs += fold_test_probs / 5.0
        
    np.save('oof_v4_probs.npy', oof_probs)
    np.save('test_v4_probs.npy', test_probs)
    
    oof_preds = np.argmax(oof_probs, axis=1)
    overall_f1 = f1_score(y, oof_preds, average='macro')
    print(f'\n============================================================')
    print(f'>>> OVERALL 5-FOLD CADENCENET-V4 CV MACRO-F1: {overall_f1:.4f} <<<')
    print(f'============================================================')
    print(classification_report(y, oof_preds, target_names=classes))
    
    # Generate Final Submission
    test_preds = np.argmax(test_probs, axis=1)
    test_labels = [code_to_label[c] for c in test_preds]
    
    sub = pd.DataFrame({
        'id': test_meta['id'],
        'label': test_labels
    })
    
    out_file = 'submission_upload_final.csv'
    sub.to_csv(out_file, index=False)
    print(f'\nSaved final submission to {out_file} successfully!')
    
    # Validation checks
    assert len(sub) == 10000, f'Expected 10000 rows, got {len(sub)}'
    assert list(sub.columns) == ['id', 'label'], f'Incorrect columns: {sub.columns}'
    assert (sub['id'] == sample_sub['id']).all(), 'IDs mismatch sample_submission.csv!'
    assert sub['label'].isnull().sum() == 0, 'Null predictions detected!'
    assert set(sub['label']).issubset(set(classes)), f'Invalid labels: {set(sub["label"]) - set(classes)}'
    
    print('\n=== Final Predicted Test Distribution ===')
    dist = sub['label'].value_counts()
    for lbl, cnt in dist.items():
        pct = cnt / len(sub) * 100
        print(f'  {lbl:20s}: {cnt:5d} ({pct:5.2f}%)')
        
    print('\n>>> ALL 5 VALIDATION CHECKS PASSED PERFECTLY! <<<')

if __name__ == '__main__':
    main()
