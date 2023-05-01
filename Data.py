import torch
import config
from torch.utils.data import Dataset


class BERTDataset(Dataset):
    def __init__(self, q1, q2, target):
        self.q1 = q1
        self.q2 = q2
        self.target = target

    def __len__(self):
        return len(self.q1)

    def __getitem__(self, item):
        q1 = str(self.q1[item])
        q2 = str(self.q2[item])
        q1 = " ".join(q1.split())
        q2 = " ".join(q2.split())
        # [CLS] question1 [SEP] questions2 [SEP] ... [PAD]
        inputs = config.TOKENIZER.encode_plus(q1, q2, add_special_tokens=True, max_length=config.MAX_LEN,
                                              pad_to_max_length=True,)
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(int(self.target[item]), dtype=torch.long),
        }


def get_data_loader(df, targets, batch_size, shuffle):
    dataset = BERTDataset(
        q1=df["question1"].values,
        q2=df["question2"].values,
        target=targets,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return data_loader
