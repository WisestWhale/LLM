import torch
from torch.utils.data import DataLoader
from Tokenizer import Tokenizer
from SlidingWindow import GPTDatasetV1


sample_text = "The dog barks at the cat, The cat sleeps on the mat"


tokenizer = Tokenizer(sample_text)
dataset = GPTDatasetV1(sample_text, tokenizer, max_length = 4 , stride = 1 )

dataloader = DataLoader(dataset , batch_size= 2 , shuffle= False)


for batch_inputs , batch_targets in dataloader:
    print(batch_inputs)
    print(batch_targets)
    break