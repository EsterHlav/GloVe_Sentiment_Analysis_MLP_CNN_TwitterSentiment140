# Ester Hlav
# Oct 6, 2019
# utils.py 

import torch
import matplotlib.pyplot as plt

def convertToTorchFloat(x):
    '''
        Converting np.array to torch.tensor (float) and, if availabe, convert to cuda.
        
        
        inputs:
            - x (np.array):            tensor
            
        return:
            - tensor (torch.tensor):   converted float tensor
    '''
    x = torch.from_numpy(x).float()
    return x.cuda() if torch.cuda.is_available() else x

def convertToTorchInt(x):
    '''
        Converting np.array to torch.tensor (int64) and, if availabe, convert to cuda.
        
        
        inputs:
            - x (np.array):            tensor
            
        return:
            - tensor (torch.tensor):   converted int64 tensor
    '''
    x = torch.from_numpy(x).to(torch.int64)
    return x.cuda() if torch.cuda.is_available() else x


def plot_perf(history, final_perf):
    '''
        Function to plot performance plots, i.e. evolution of metrics during training
        on train, val and test sets.
    '''
    epochs = range(1, len(history['loss'])+1)
    for key in ['loss', 'accuracy', 'precision', 'recall', 'f1']:
        plt.figure(figsize=(6,4))
        plt.plot(epochs, history[key], '+-b', label=key)
        plt.plot(epochs, history['val_'+key], '+-g', label='val_'+key)
        plt.axhline(y=final_perf[key], color='r', linestyle='--', label='test_'+key)
        plt.legend()
        plt.title('Evolution of {} during training'.format(key))
        plt.plot()