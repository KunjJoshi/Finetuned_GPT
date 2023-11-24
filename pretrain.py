import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup

df=pd.read_csv('preprocessedDataBricks.csv')

tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
model=GPT2LMHeadModel.from_pretrained('gpt2')

class CustomDataset(Dataset):
    def __init__(self,data,tokenizer,max_length):
        self.data=data
        self.tokenizer=tokenizer
        self.max_length=max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        text=self.data.iloc[idx]['input_text']
        tokens=self.tokenizer.encode(text,add_special_tokens=True,max_length=self.max_length,truncation=True)
        attention_mask=[1]*len(tokens)

        return{
            'input_ids':torch.tensor(tokens),
            'attention_mask':torch.tensor(attention_mask)
        }


max_length=max(df['input_text'].apply(lambda x: len(x)))
dataset=CustomDataset(data=df,max_length=max_length,tokenizer=tokenizer)

def collate_fn(batch):
    input_ids=pad_sequence([item['input_ids'] for item in batch],batch_first=True)
    attention_mask=pad_sequence([item['attention_mask'] for item in batch],batch_first=True)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

data_loader=DataLoader(dataset,batch_size=8,shuffle=True,collate_fn=collate_fn)
device=torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
model.to(device)
optimizer=AdamW(model.parameters(),lr=1e-5)
scheduler=get_linear_schedule_with_warmup(optimizer=optimizer,num_warmup_steps=0,num_training_steps=len(data_loader))
num_epochs=5



for epoch in range(num_epochs):
    print(f'Initiating Epoch {epoch}/{num_epochs}')
    model.train()
    total_loss=0
    i=0
    for batch in data_loader:
        print('Processing Batch ',i)
        input_ids=batch['input_ids'].to(device)
        attention_mask=batch['attention_mask'].to(device)
        outputs=model(input_ids,attention_mask=attention_mask,labels=input_ids)
        loss=outputs.loss
        print(f'Loss for Batch {i} : {loss}')
        total_loss=total_loss+loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        i=i+1

    avg_loss=total_loss/len(data_loader)

    print(f"Epoch{epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

    model.save_pretrained('example_gpt')