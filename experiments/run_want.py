import os
import pickle
from data.wanT import get_want_dataset
from data.serialize import SerializerSettings
from models.validation_likelihood_tuning import get_autotuned_predictions_data
from models.utils import grid_iter
from models.llmtime import get_llmtime_predictions_data
import numpy as np
import time

llama_hypers = dict(
    temp=1.0,
    alpha=0.99,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, time_sep=',', bit_sep='', plus_sign='', minus_sign='-', signed=True), 
)


model_hypers = {
    # 'text-davinci-003': {'model': 'text-davinci-003', **gpt3_hypers},
    'llama-7b': {'model': 'llama-7b', **llama_hypers},
    # 'llama-70b': {'model': 'llama-70b', **llama_hypers},
}

# Specify the function to get predictions for each model
model_predict_fns = {
    # 'text-davinci-003': get_llmtime_predictions_data,
    'llama-7b': get_llmtime_predictions_data,
    # 'llama-70b': get_llmtime_predictions_data,
}

def is_gpt(model):
    return any([x in model for x in ['ada', 'babbage', 'curie', 'davinci', 'text-davinci-003', 'gpt-4']])

# Specify the output directory for saving results
output_dir = 'outputs/want'
os.makedirs(output_dir, exist_ok=True)

models_to_run = [
    # 'text-davinci-003',
    'llama-7b',
    # 'llama-70b',
]

datasets_to_run =  [
    "nn5_daily"
]

start_time = time.time()
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
    
    for model in ['llama-7b', 'gp', 'arima', 'N-HiTS']:
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
        num_samples = 20 if is_gpt(model) else 100
        hyper_start_time = time.time() - start_time
        print(f"Starting hyperparameter tuning after {hyper_start_time:.2f} seconds")

        try:
            preds = get_autotuned_predictions_data(train, test, hypers, num_samples, model_predict_fns[model], verbose=0, parallel=parallel)
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
            pickle.dump(out_dict,f)
    

    print(f"Finished {dsname}")