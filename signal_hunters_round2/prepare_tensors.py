import numpy as np
import time
import os
from cadence_cnn import prepare_cadence_tensor

def main():
    print('=== Preparing 9-Channel Tensors for CadenceNet ===')
    
    # 1. Train Tensors
    train_out = 'train_tensors_9ch.npy'
    if os.path.exists(train_out):
        print(f'{train_out} already exists! Skipping.')
    else:
        print('Loading train_cadences.npy (18,000 samples)...')
        X_train = np.load('train_cadences.npy', mmap_mode='r')
        batch_size = 500
        n_batches = int(np.ceil(len(X_train) / batch_size))
        
        t0 = time.time()
        tensors = []
        for b in range(n_batches):
            start = b * batch_size
            end = min(len(X_train), (b + 1) * batch_size)
            chunk = prepare_cadence_tensor(X_train[start:end])
            tensors.append(chunk.astype(np.float16))
            if (b + 1) % 5 == 0 or (b + 1) == n_batches:
                print(f'  Train: Processed {end}/{len(X_train)} ({time.time() - t0:.1f}s)')
                
        all_train = np.concatenate(tensors, axis=0)
        print(f'Saving {train_out} (shape: {all_train.shape}, dtype: {all_train.dtype})...')
        np.save(train_out, all_train)
        print(f'Train tensors saved successfully in {time.time() - t0:.1f}s!')
        
    # 2. Test Tensors
    test_out = 'test_tensors_9ch.npy'
    if os.path.exists(test_out):
        print(f'{test_out} already exists! Skipping.')
    else:
        print('\nLoading test_cadences.npy (10,000 samples)...')
        X_test = np.load('test_cadences.npy', mmap_mode='r')
        batch_size = 500
        n_batches = int(np.ceil(len(X_test) / batch_size))
        
        t0 = time.time()
        tensors = []
        for b in range(n_batches):
            start = b * batch_size
            end = min(len(X_test), (b + 1) * batch_size)
            chunk = prepare_cadence_tensor(X_test[start:end])
            tensors.append(chunk.astype(np.float16))
            if (b + 1) % 5 == 0 or (b + 1) == n_batches:
                print(f'  Test: Processed {end}/{len(X_test)} ({time.time() - t0:.1f}s)')
                
        all_test = np.concatenate(tensors, axis=0)
        print(f'Saving {test_out} (shape: {all_test.shape}, dtype: {all_test.dtype})...')
        np.save(test_out, all_test)
        print(f'Test tensors saved successfully in {time.time() - t0:.1f}s!')

if __name__ == '__main__':
    main()
