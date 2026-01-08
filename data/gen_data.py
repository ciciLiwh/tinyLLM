import os
import requests
import numpy as np
from tokenizers import ByteLevelBPETokenizer

input_file_path = 'input.txt'

if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
    
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=[input_file_path],
    vocab_size=3000,         
    min_frequency=2,
    special_tokens=["<|endoftext|>"],
)

if not os.path.exists("bbpe"):
    os.mkdir("bbpe")
    
tokenizer.save_model("bbpe")

tokenizer = ByteLevelBPETokenizer(
    "bbpe/vocab.json",
    "bbpe/merges.txt",
)

train_ids = tokenizer.encode(train_data).ids
val_ids   = tokenizer.encode(val_data).ids

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile('train.bin')
val_ids.tofile('val.bin')