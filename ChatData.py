from torch.utils.data import Dataset
import pandas as pd

class ChatData(Dataset):
    def __init__(self, path:str, tokenizer):
        self.data = list(pd.read_csv(path)[0])
            
        self.X = self.X[:-1]
        self.X_encoded = tokenizer(self.data, truncation = True, padding = True)
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

            


    def __len__(slef):
        return len(slef.data)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])