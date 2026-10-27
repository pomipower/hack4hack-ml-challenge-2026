import numpy as np
import time
import os

def prepare_cadence_tensor_v4(cad_raw):
    # cad_raw: (B, 6, 16, 128) or (6, 16, 128)
    is_batch = (cad_raw.ndim == 4)
    if not is_batch:
        cad_raw = cad_raw[None, ...]
    
    x = cad_raw.astype(np.float32)
    # Mask dropped rows (0)
    x[x == 0] = np.nan
    
    # 1. Linear power
    power = 2.0 ** (x / 24.0 - 4.0)
    
    # 2. Bandpass flattening per channel: median across all 6 panels and 16 rows
    bp = np.nanmedian(power, axis=(1, 2), keepdims=True) # (B, 1, 1, 128)
    p_flat = power / (bp + 1e-6)
    
    # 3. Robust panel-level centering
    p_med = np.nanmedian(p_flat, axis=(2, 3), keepdims=True) # (B, 6, 1, 1)
    centered = p_flat - p_med
    
    # 4. Standardize by cadence MAD
    abs_dev = np.abs(centered)
    c_mad = np.nanmedian(abs_dev, axis=(1, 2, 3), keepdims=True) * 1.4826
    norm = centered / (c_mad + 1e-4)
    norm = np.nan_to_num(norm, nan=0.0)
    
    # 5. Differential ON-OFF channels: [p0-p1, p2-p3, p4-p5]
    d0 = norm[:, 0] - norm[:, 1]
    d1 = norm[:, 2] - norm[:, 3]
    d2 = norm[:, 4] - norm[:, 5]
    diffs = np.stack([d0, d1, d2], axis=1) # (B, 3, 16, 128)
    
    # Stack into 9 channels
    t9 = np.concatenate([norm, diffs], axis=1) # (B, 9, 16, 128)
    return t9 if is_batch else t9[0]

def main():
    print('=== Preparing Invariant v4 Tensors (Linear Power + Bandpass Flattening) ===')
    
    # 1. Train Tensors
    train_out = 'train_tensors_v4.npy'
    print('Loading train_cadences.npy (18,000 samples)...')
    X_train = np.load('train_cadences.npy', mmap_mode='r')
    batch_size = 500
    n_batches = int(np.ceil(len(X_train) / batch_size))
    
    t0 = time.time()
    tensors = []
    for b in range(n_batches):
        start = b * batch_size
        end = min(len(X_train), (b + 1) * batch_size)
        chunk = prepare_cadence_tensor_v4(X_train[start:end])
        tensors.append(chunk.astype(np.float16))
        if (b + 1) % 5 == 0 or (b + 1) == n_batches:
            print(f'  Train: Processed {end}/{len(X_train)} ({time.time() - t0:.1f}s)')
            
    all_train = np.concatenate(tensors, axis=0)
    print(f'Saving {train_out} (shape: {all_train.shape}, dtype: {all_train.dtype})...')
    np.save(train_out, all_train)
    print(f'Train tensors saved successfully in {time.time() - t0:.1f}s!')
    
    # 2. Test Tensors
    test_out = 'test_tensors_v4.npy'
    print('\nLoading test_cadences.npy (10,000 samples)...')
    X_test = np.load('test_cadences.npy', mmap_mode='r')
    batch_size = 500
    n_batches = int(np.ceil(len(X_test) / batch_size))
    
    t0 = time.time()
    tensors = []
    for b in range(n_batches):
        start = b * batch_size
        end = min(len(X_test), (b + 1) * batch_size)
        chunk = prepare_cadence_tensor_v4(X_test[start:end])
        tensors.append(chunk.astype(np.float16))
        if (b + 1) % 5 == 0 or (b + 1) == n_batches:
            print(f'  Test: Processed {end}/{len(X_test)} ({time.time() - t0:.1f}s)')
            
    all_test = np.concatenate(tensors, axis=0)
    print(f'Saving {test_out} (shape: {all_test.shape}, dtype: {all_test.dtype})...')
    np.save(test_out, all_test)
    print(f'Test tensors saved successfully in {time.time() - t0:.1f}s!')

if __name__ == '__main__':
    main()
