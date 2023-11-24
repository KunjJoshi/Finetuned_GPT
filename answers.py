import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup

model=GPT2LMHeadModel.from_pretrained('trained_gpt_py')
tokenizer=GPT2Tokenizer.from_pretrained('gpt2')


user_q=input('Enter Question')
input_ids=tokenizer.encode(user_q,add_special_tokens=True, return_tensors='pt')
padding_token_id=tokenizer.eos_token_id
attention_mask=input_ids.ne(padding_token_id)
output=model.generate(input_ids, attention_mask=attention_mask, max_length=300, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
ans=tokenizer.decode(output[0], skip_special_tokens=True)
print(ans)
