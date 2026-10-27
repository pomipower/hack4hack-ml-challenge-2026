import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import os
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from cadence_cnn import CadenceNet

# Reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CadenceDataset(Dataset):
    def __init__(self, tensors, labels=None, is_train=False):
        self.tensors = tensors # float16 numpy array or tensor
        self.labels = labels
        self.is_train = is_train
        
    def __len__(self):
        return len(self.tensors)
        
    def __getitem__(self, idx):
        # Convert to float32 tensor: (9, 16, 128)
        x = torch.from_numpy(self.tensors[idx].astype(np.float32))
        
        if self.is_train:
            # 1. Random horizontal flip (reverses frequency drift direction - 100% physically valid!)
            if random.random() < 0.5:
                x = torch.flip(x, dims=[-1])
                
            # 2. Random frequency shift (small jitter in start channel +/- 6 channels)
            if random.random() < 0.5:
                shift = random.randint(-6, 6)
                if shift != 0:
                    x = torch.roll(x, shifts=shift, dims=-1)
                    if shift > 0:
                        x[:, :, :shift] = 0.0
                    else:
                        x[:, :, shift:] = 0.0
                        
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
def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)

@torch.no_grad()
def predict_test(model, loader, device):
    model.eval()
    all_probs = []
    for x in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)

def main():
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training CadenceNet on device: {device} ({torch.cuda.get_device_name(0)})')
    
    # 1. Load Data
    t0 = time.time()
    train_meta = pd.read_csv('train_meta.csv')
    test_meta = pd.read_csv('test_meta.csv')
    
    classes = np.sort(train_meta['label'].unique())
    label_to_code = {c: i for i, c in enumerate(classes)}
    code_to_label = {i: c for i, c in enumerate(classes)}
    y = train_meta['label'].map(label_to_code).values
    
    print('Loading precomputed 9-channel tensors...')
    X_train_tensors = np.load('train_tensors_9ch.npy')
    X_test_tensors = np.load('test_tensors_9ch.npy')
    print(f'Train tensors: {X_train_tensors.shape}, Test tensors: {X_test_tensors.shape} (loaded in {time.time() - t0:.1f}s)')
    
    # Compute balanced square-root class weights for Macro-F1
    counts = np.bincount(y)
    weights = (len(y) / (len(counts) * counts)) ** 0.5
    weights = weights / np.mean(weights)
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    print('Class weights:')
    for c, w in zip(classes, weights):
        print(f'  {c:20s}: {w:.3f}')
        
    test_dataset = CadenceDataset(X_test_tensors, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_cnn_probs = np.zeros((len(y), len(classes)), dtype=np.float32)
    test_cnn_probs = np.zeros((len(test_meta), len(classes)), dtype=np.float32)
    
    epochs = 12
    batch_size = 64
    
    print(f'\n=== Starting 5-Fold Stratified Training ({epochs} epochs/fold) ===')
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_tensors, y)):
        t_fold = time.time()
        print(f'\n--- Fold {fold + 1} / 5 ---')
        
        tr_ds = CadenceDataset(X_train_tensors[tr_idx], y[tr_idx], is_train=True)
        va_ds = CadenceDataset(X_train_tensors[va_idx], y[va_idx], is_train=False)
        
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        va_loader = DataLoader(va_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
        
        model = CadenceNet(num_classes=len(classes)).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(tr_loader), eta_min=1e-5)
        scaler = torch.amp.GradScaler('cuda')
        
        best_va_f1 = 0.0
        best_va_probs = None
        
        for ep in range(epochs):
            t_ep = time.time()
            loss = train_one_epoch(model, tr_loader, optimizer, scheduler, criterion, scaler, device)
            va_probs = evaluate(model, va_loader, device)
            va_preds = np.argmax(va_probs, axis=1)
            va_f1 = f1_score(y[va_idx], va_preds, average='macro')
            
            if va_f1 > best_va_f1:
                best_va_f1 = va_f1
                best_va_probs = va_probs.copy()
                torch.save(model.state_dict(), f'cadence_cnn_fold{fold}.pt')
                
            print(f'  Epoch {ep+1:2d}/{epochs:2d} ({time.time() - t_ep:.1f}s) | Loss: {loss:.4f} | Val Macro-F1: {va_f1:.4f} {"*" if va_f1 == best_va_f1 else ""}')
            
        print(f'Fold {fold + 1} Best Macro-F1: {best_va_f1:.4f} in {time.time() - t_fold:.1f}s')
        oof_cnn_probs[va_idx] = best_va_probs
        
        # Load best weights for test inference
        model.load_state_dict(torch.load(f'cadence_cnn_fold{fold}.pt'))
        fold_test_probs = predict_test(model, test_loader, device)
        test_cnn_probs += fold_test_probs / 5.0
        
    np.save('oof_cnn_probs.npy', oof_cnn_probs)
    np.save('test_cnn_probs.npy', test_cnn_probs)
    print('\nSaved oof_cnn_probs.npy and test_cnn_probs.npy!')
    
    # Evaluate CNN Overall
    cnn_oof_preds = np.argmax(oof_cnn_probs, axis=1)
    cnn_overall_f1 = f1_score(y, cnn_oof_preds, average='macro')
    print(f'\n>>> OVERALL 5-FOLD CADENCENET CV MACRO-F1: {cnn_overall_f1:.4f} <<<')
    print(classification_report(y, cnn_oof_preds, target_names=classes))

if __name__ == '__main__':
    main()
