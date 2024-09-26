from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd

import numpy as np
import pickle

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = '/NS/ssdecl/work')

ds = load_dataset("lmsys/lmsys-chat-1m", split='train', cache_dir='/NS/ssdecl/work')


turns_list = list(ds['turn'])


with open('lmsys_trace_prefil.pkl','rb') as f1:
    prefil_list = pickle.load(f1)


with open('lmsys_trace_decode.pkl','rb') as f2:
    decode_list = pickle.load(f2)

mod_prefil_list = []
start = 0
count = 0
for turn_value in turns_list:
    end = start + turn_value
    count  = count +1

    current_prefil = prefil_list[start:end]
    current_decode = decode_list[start:end]
    current_sum = 0

    for item_index in range(len(current_prefil)):
        current_sum =  current_sum + current_prefil[item_index] +  current_decode[item_index]
        mod_prefil_list.append(current_sum)

    start = end

    #print(count)


print(sorted(mod_prefil_list)[-10:])

print('Average tokens From Lmsys chat string', np.mean(mod_prefil_list))


with open('lmsys_trace_prefil_running.pkl', 'wb') as f3:
    pickle.dump( mod_prefil_list, f3)






