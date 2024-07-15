import os
import numpy as np
import matplotlib.pyplot as plt
from data.serialize import SerializerSettings
from models.utils import grid_iter
from models.gaussian_process import get_gp_predictions_data
from models.darts import get_TCN_predictions_data, get_NHITS_predictions_data, get_NBEATS_predictions_data
from models.llmtime import get_llmtime_predictions_data
from models.darts import get_arima_predictions_data
import pickle
import matplotlib.pyplot as plt
from models.validation_likelihood_tuning import get_autotuned_predictions_data
from data.wanT import get_want_dataset, get_scaled_dataset
import time
import gc
import torch
import gc

torch.cuda.empty_cache()
gc.collect()

llama_hypers = dict(
    temp=1.0,
    alpha=0.99,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, time_sep=',', bit_sep='', plus_sign='', minus_sign='-', signed=True), 
)

gp_hypers = dict(lr=[5e-3, 1e-2, 5e-2, 1e-1])

arima_hypers = dict(p=[12,20,30], d=[1,2], q=[0,1,2])

TCN_hypers = dict(in_len=[10, 100, 400], out_len=[1],
    kernel_size=[3, 5], num_filters=[1, 3], 
    likelihood=['laplace', 'gaussian']
)

NHITS_hypers = dict(in_len=[10, 100, 400], out_len=[1],
    layer_widths=[64, 16], num_layers=[1, 2], 
    likelihood=['laplace', 'gaussian']
)


NBEATS_hypers = dict(in_len=[10, 100, 400], out_len=[1],
    layer_widths=[64, 16], num_layers=[1, 2], 
    likelihood=['laplace', 'gaussian']
)

model_hypers = {
    'gp': gp_hypers,
    'arima': arima_hypers,
    'TCN': TCN_hypers,
    'N-BEATS': NBEATS_hypers,
    'N-HiTS': NHITS_hypers,
    'llama-7b': {'model': 'llama-7b', **llama_hypers},
}

model_predict_fns = {
    'gp': get_gp_predictions_data,
    'arima': get_arima_predictions_data,
    'TCN': get_TCN_predictions_data,
    'N-BEATS': get_NBEATS_predictions_data,
    'N-HiTS': get_NHITS_predictions_data,
    'llama-7b': get_llmtime_predictions_data,
}

def is_gpt(model):
    return any([x in model for x in ['ada', 'babbage', 'curie', 'davinci', 'text-davinci-003', 'gpt-4']])

# Specify the output directory for saving results
output_dir = 'outputs/want'
os.makedirs(output_dir, exist_ok=True)

start_time = time.time()
# datasets, scaler = get_scaled_dataset()
datasets = get_want_dataset()
loading_time = time.time() - start_time
print(f"Loading datasets took {loading_time:.2f} seconds")
for dsname,data in datasets.items():
    train, test = data
    if os.path.exists(f'{output_dir}/{dsname}.pkl'):
        with open(f'{output_dir}/{dsname}.pkl','rb') as f:
            out_dict = pickle.load(f)
    else:
        out_dict = {}
    
    for model in ['llama-7b']:
        if model in out_dict and not is_gpt(model):
            if out_dict[model]['samples'] is not None:
                print(f"Skipping {dsname} {model}")
                continue
            else:
                print('Using best hyper...')
                hypers = [out_dict[model]['best_hyper']]
        else:
            print(f"Starting {dsname} {model}")
            hypers = list(grid_iter(model_hypers[model]))
        parallel = True if is_gpt(model) else False
        # num_samples = 20 if is_gpt(model) else 100
        num_samples = 1
        hyper_start_time = time.time() - start_time
        print(f"Starting hyperparameter tuning after {hyper_start_time:.2f} seconds")

        try:
            print(torch.cuda.memory_summary(device=None, abbreviated=False))
            preds = get_autotuned_predictions_data(train, test, hypers, num_samples, model_predict_fns[model], verbose=0, parallel=parallel)
            print(torch.cuda.memory_summary(device=None, abbreviated=False))
            hyper_end_time = time.time() - (hyper_start_time + start_time)
            print(f"Hyperparameter tuning took {hyper_end_time:.2f} seconds")
            if preds.get('NLL/D', np.inf) < np.inf:
                out_dict[model] = preds
            else:
                print(f"Failed {dsname} {model}")
        except Exception as e:
            print(f"Failed {dsname} {model}")
            print(e)
            continue
        with open(f'{output_dir}/{dsname}.pkl','wb') as f:
            # out_dict['scaler'] = scaler
            pickle.dump(out_dict,f)
    

    print(f"Finished {dsname}")