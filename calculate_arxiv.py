from datasets import load_dataset

from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd

import numpy as np
import pickle

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = '/NS/ssdecl/work')


def tokenization(example):
    return len(tokenizer(example)['input_ids'])


#tokenizer.pad_token = tokenizer.eos_token

print('starting')


dataset = load_dataset("ccdv/arxiv-summarization", "section", split= 'train', cache_dir='/NS/ssdecl/work')


# Load the ArXiv summarization dataset
#dataset = load_dataset("arxiv")

# Initialize lists to hold the token counts
num_prefill_tokens = []
num_decode_tokens = []
num_total_tokens = []
pd_ratio_tokens = []

# Iterate through the dataset
count = 0
for item in dataset:
    print(count)
    count = count+1
    # Get the article and abstract
    article = item['article']
    abstract = item['abstract']
    
    # Tokenize the article and abstract
    article_tokens = tokenizer(article)
    abstract_tokens = tokenizer(abstract)

    # Calculate the number of tokens
    num_article_tokens = len(article_tokens['input_ids'])
    num_abstract_tokens = len(abstract_tokens['input_ids'])
    #numel() - abstract_tokens['input_ids'].eq(tokenizer.pad_token_id).sum().item()

    # Store the counts
    num_prefill_tokens.append(num_article_tokens)
    num_decode_tokens.append(num_abstract_tokens)
    num_total_tokens.append(num_article_tokens + num_abstract_tokens)
    pd_ratio_tokens.append(num_article_tokens/num_abstract_tokens)

# Create a DataFrame from the token counts
df = pd.DataFrame({
    'num_prefill_tokens': num_prefill_tokens,
    'num_decode_tokens': num_decode_tokens,
    'num_total_tokens': num_total_tokens,
    'pd_ratio': pd_ratio_tokens
})

# Save the DataFrame to a CSV file
df.to_csv('arxiv_token_counts.csv', index=False)

print("Token counts saved to 'arxiv_token_counts.csv'.")
