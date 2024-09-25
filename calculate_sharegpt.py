from datasets import load_dataset

from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd

import numpy as np
import pickle
import json
import csv

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = '/NS/ssdecl/work')



def calculate_tokens(conversation):
    # Split the conversation into turns using markers
    # The first turn starts with '<s> Human', others start with 'Human:'
    all_turns = conversation.split('Human:')

    #print(len(turns))


    
    turn_details = []
    accumulated_tokens = 0  # Keeps track of all tokens for prefill accumulation

    # Process the first turn separately since it starts with "<s> Human"



    if len(all_turns) ==2 and conversation.split('<s>Human:')[1].endswith('<|end_of_turn|>'):
        turns = conversation.split('<s>Human:')

        turn = turns[1]



        human_content = turn.split('<|end_of_turn|>')[0].strip()
        assistant_content = turn.split('<|end_of_turn|>')[1].strip().replace("Assistant:", "").strip()

        # Calculate tokens for the first human turn
        current_human_tokens = len(tokenizer(human_content)['input_ids'])

        # Calculate decode tokens for the assistant response in this turn
        num_decode_tokens = len(tokenizer(assistant_content)['input_ids'])

        # Prefill tokens (initially just the current human tokens for the first turn)
        num_prefill_tokens = current_human_tokens

        # Collect the details for the first turn
        num_total_tokens = num_prefill_tokens +  num_decode_tokens
        pd_ratio = num_prefill_tokens / num_decode_tokens

        turn_details.append({
            "turn_index": 1,
            "num_prefill_tokens": num_prefill_tokens,
            "num_decode_tokens": num_decode_tokens,
            "num_total_tokens": num_total_tokens,
            "pd_ratio": pd_ratio
        })

        # Update accumulated tokens to include both the human and assistant tokens
        accumulated_tokens = current_human_tokens + num_decode_tokens

    elif len(all_turns) > 2:

        # Process subsequent turns that start with 'Human:'
        remaining_turns = conversation.split('Human:')[1:]  # Ignore the initial 'Human:'
        accumulated_tokens = 0

        for turn_index, turn in enumerate(remaining_turns, start=2):

            if not turn.endswith('<|end_of_turn|>'):
                continue
            # Splitting to extract Human and Assistant content
            human_content = turn.split('<|end_of_turn|>')[0].strip()
            assistant_content = turn.split('<|end_of_turn|>')[1].strip().replace("Assistant:", "").strip()

            # Calculate tokens for the current human turn
            current_human_tokens = len(tokenizer(human_content)['input_ids'])

            # Calculate decode tokens for the assistant response in this turn
            num_decode_tokens = len(tokenizer(assistant_content)['input_ids'])

            # Prefill tokens should be the accumulation of all tokens up to this turn
            num_prefill_tokens = accumulated_tokens + current_human_tokens

            num_total_tokens = num_prefill_tokens + num_decode_tokens
            pd_ratio = num_prefill_tokens / num_decode_tokens
            # Collect the details for each turn

            turn_details.append({
            "turn_index": turn_index-1,
            "num_prefill_tokens": num_prefill_tokens,
            "num_decode_tokens": num_decode_tokens,
            "num_total_tokens": num_total_tokens,
            "pd_ratio": pd_ratio
            })

            # Update accumulated tokens to include both the human and assistant tokens for future turns
            accumulated_tokens += current_human_tokens + num_decode_tokens

    return turn_details

def main():
    # Load JSON data
    with open('openchat_8192.train.text.json', 'r', encoding='utf-8') as file:
        data = json.load(file)


    all_turn_details = []

    # Iterating over each conversation string
    for conversation_index, conversation in enumerate(data, start=1):
        print('conv index', conversation_index)
        turn_details = calculate_tokens(conversation)
        for turn in turn_details:
            turn["conversation_index"] = conversation_index  # Update with actual conversation index
        all_turn_details.extend(turn_details)

    # Save all turn details into a CSV file
    with open('conversation_tokens.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["conversation_index", "turn_index", "num_prefill_tokens", "num_decode_tokens", "num_total_tokens", "pd_ratio"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for turn in all_turn_details:
            writer.writerow(turn)

    print("Data saved to conversation_tokens.csv")

if __name__ == "__main__":
    main()



