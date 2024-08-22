import os
import pickle
from data.monash import get_datasets
from data.serialize import SerializerSettings
from models.validation_likelihood_tuning import get_autotuned_predictions_data
from models.utils import grid_iter
from models.llmtime import get_llmtime_predictions_data
import numpy as np
import time 

# import openai
# openai.api_key = os.environ['OPENAI_API_KEY']
# openai.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

# Specify the hyperparameter grid for each model
# gpt3_hypers = dict(
#     temp=0.7,
#     alpha=0.9,
#     beta=0,
#     basic=False,
#     settings=SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True),
# )

llama_hypers = dict(
    temp=1.0,
    alpha=0.99,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, time_sep=',', bit_sep='', plus_sign='', minus_sign='-', signed=True), 
)


llama31_hypers = dict(
    temp=1.0,
    alpha=0.99,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, time_sep=',', bit_sep='', plus_sign='', minus_sign='-', signed=True, max_val=1e11), 
)

model_hypers = {
    # 'text-davinci-003': {'model': 'text-davinci-003', **gpt3_hypers},
    'llama-7b': {'model': 'llama-7b', **llama_hypers},
    'llama3.1-8b': {'model': 'llama3.1-8b', **llama31_hypers},
    # 'llama-70b': {'model': 'llama-70b', **llama_hypers},
}

# Specify the function to get predictions for each model
model_predict_fns = {
    # 'text-davinci-003': get_llmtime_predictions_data,
    'llama-7b': get_llmtime_predictions_data,
    'llama3.1-8b': get_llmtime_predictions_data,
    # 'llama-70b': get_llmtime_predictions_data,
}

def is_gpt(model):
    return any([x in model for x in ['ada', 'babbage', 'curie', 'davinci', 'text-davinci-003', 'gpt-4']])

# Specify the output directory for saving results
output_dir = 'outputs/monash'
os.makedirs(output_dir, exist_ok=True)

models_to_run = [
    # 'text-davinci-003',
    'llama3.1-8b',
    # 'llama-70b',
]
datasets_to_run =  [
    "nn5_daily"
]

max_history_len = 500
start_time = time.time()
datasets = get_datasets()
loading_time = time.time() - start_time
print(f"Loading datasets took {loading_time:.2f} seconds")
for dsname in datasets_to_run:
    print(f"Starting {dsname}")
    data = datasets[dsname]
    train, test = data
    train = [x[-max_history_len:] for x in train]
    if os.path.exists(f'{output_dir}/{dsname}.pkl'):
        with open(f'{output_dir}/{dsname}.pkl','rb') as f:
            out_dict = pickle.load(f)
    else:
        out_dict = {}
    
    for model in models_to_run:
        if model in out_dict:
            print(f"Skipping {dsname} {model}")
            continue
        else:
            print(f"Starting {dsname} {model}")
            hypers = list(grid_iter(model_hypers[model]))
        parallel = True if is_gpt(model) else False
        num_samples = 5
        hyper_start_time = time.time() - start_time
        print(f"Starting hyperparameter tuning after {hyper_start_time:.2f} seconds")
        
        try:
            preds = get_autotuned_predictions_data(train, test, hypers, num_samples, model_predict_fns[model], verbose=False, parallel=parallel)
            hyper_end_time = time.time() - (hyper_start_time + start_time)
            print(f"Hyperparameter tuning took {hyper_end_time:.2f} seconds")
            medians = preds['median']
            targets = np.array(test)
            maes = np.mean(np.abs(medians - targets), axis=1) # (num_series)        
            preds['maes'] = maes
            preds['mae'] = np.mean(maes)
            out_dict[model] = preds
        except Exception as e:
            print(f"Failed {dsname} {model}")
            print(e)
            continue
        with open(f'{output_dir}/{dsname}.pkl','wb') as f:
            pickle.dump(out_dict,f)
    print(f"Finished {dsname}")
    total_time = time.time() - start_time
    print(f"Total took {total_time:.2f} seconds")