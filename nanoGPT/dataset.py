from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import time
import json
import random
import librosa
import pickle
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

#logger = RankedLogger(__name__, rank_zero_only=False)


class SVSLMDataset(Dataset):
    def __init__(
        self,
        filelist: str,
        sample_rate: int = 32000,
        hop_length: int = 640,
        slice_frames: Optional[int] = None,
        num_codebooks : int = 4,
        codebook_size: int = 1024,
        mode: str = 'all',
        max_seq_len: int = 2000,
        data_type: str = 'pyramid',
        is_train: bool = True,
    ):
        super().__init__()

        self.files = []
        if isinstance(filelist, str):
            filelists = [filelist]
        else:
            filelists = filelist

        for filelist in filelists:
            filelist = Path(filelist)
            root = filelist.parent

            self.files += [
                root / line.strip()
                for line in filelist.read_text().splitlines()
                if line.strip()
            ]
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.slice_frames = slice_frames

        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.data_type = data_type
        self.is_train = is_train

        self.pad_token = codebook_size
        self.bos_token = codebook_size + 1
        self.eos_token = codebook_size + 2
        self.sep_token = codebook_size + 3

    def __len__(self):
        return len(self.files)

    def stage1(self, res):
        max_len = self.max_seq_len
        codec1 = res[0].cpu().squeeze().tolist()
        if len(codec1) > max_len:
            s = random.randint(0, len(codec1)-max_len-1)
            tokens = codec1[s: s + max_len]
            valid_len = len(tokens)
        elif len(codec1) <= max_len:
            pad = [self.codebook_size] * (max_len - len(codec1))
            tokens = codec1 + pad
            valid_len = len(codec1)

        tokens = torch.tensor(tokens)
        attention_mask = torch.ones((len(tokens),), dtype=torch.bool)
        loss_mask = [1] * valid_len + [0] * (max_len - valid_len)
        loss_mask = torch.tensor(loss_mask)
        attention_mask[0: valid_len] = False

        labels = torch.tensor(tokens).long()
        labels[loss_mask == 0] = -100

        results = {
            "inputs": torch.tensor(tokens)[:-1].long(),
            "attention_masks": attention_mask[:-1],
            "labels": labels[1: ],
            "loss_masks": loss_mask[1: ],
            "prompt_len": 0,
            "valid_len": valid_len,
        }

        return results

    def process(self, ress, file_path=None):
        max_len = self.max_seq_len
        codec1, codec2, codec3 = [], [], []
        for res in ress:
            codec1 += np.squeeze(res[0]).tolist()
            codec2 += np.squeeze(res[1]).tolist()
            codec3 += np.squeeze(res[2]).tolist()

        assert len(codec1) >= max_len

        if self.is_train:
            s = random.randint(0, len(codec1)-max_len)
        else:
            #s = random.randint(0, len(codec1)-max_len)
            if len(codec1)-max_len > 30:
                s = 30
            else:
                s = 0

        mode = self.mode
        if mode == 'all':
            tokens = [self.bos_token] + codec1[s: s + max_len] + [self.sep_token] + codec2[s*2: (s+max_len)*2] + [self.sep_token] + codec3[s*4: (s+max_len)*4] + [self.eos_token]
            valid_len = len(tokens)
            loss_mask = [1] * valid_len + [0] * (max_len * 7 - valid_len + 4)
            codec1 = torch.tensor(codec1[s: s + max_len])
            codec2 = torch.tensor(codec2[s*2: (s+max_len)*2])
            codec3 = torch.tensor(codec3[s*4: (s+max_len)*4])

        elif mode == 'single':
            tokens = [self.bos_token] + codec1[s: s + max_len] + [self.eos_token]
            valid_len = len(tokens)
            loss_mask = [1] * valid_len + [0] * (max_len - valid_len + 2)

        elif mode == 'stage2':
            tokens = [self.bos_token] + codec1[s: s + max_len] + [self.sep_token] + codec2[s*2: (s+max_len)*2] + [self.eos_token]
            valid_len = len(tokens)
            loss_mask = [0] * (max_len + 2) + [1] * (2 * max_len + 1) + [0] * (max_len * 3 - valid_len + 3)

        elif mode == 'stage3':
            tokens = [self.bos_token] + codec1[s: s + max_len] + [self.sep_token] + codec2[s*2: (s+max_len)*2] + [self.sep_token] + codec3[s*4: (s+max_len)*4] + [self.eos_token]
            valid_len = len(tokens)
            loss_mask = [0] * (max_len * 3 + 3) + [1] * (4 * max_len + 1) + [0] * (max_len * 7 - valid_len + 4)

        tokens = torch.tensor(tokens)
        attention_mask = torch.ones((len(tokens),), dtype=torch.bool)
        #loss_mask = [1] * valid_len + [0] * (max_len * 7 - valid_len + 4)
        #loss_mask = [1] * valid_len + [0] * (max_len - valid_len + 2)
        loss_mask = torch.tensor(loss_mask)
        attention_mask[0: valid_len] = False

        labels = tokens.long().clone()
        labels[loss_mask == 0] = -1

        results = {
            "inputs": tokens[:-1].long(),
            "attention_masks": attention_mask[:-1],
            "labels": labels[1: ],
            "loss_masks": loss_mask[1: ],
            "prompt_len": 0,
            "valid_len": valid_len,
            "codec1": codec1,
            "codec2": codec2,
            "codec3": codec3,
            "path": file_path,
        }

        return results

    def process2(self, ress, file_path=None):
        max_len = self.max_seq_len
        codec1, codec2 = [], []
        for res in ress:
            res = np.squeeze(res)
            codec1 += res[0].tolist()
            codec2 += res[1].tolist()

        assert len(codec1) >= max_len
        if self.is_train:
            s = random.randint(0, len(codec1)-max_len)
        else:
            #s = random.randint(0, len(codec1)-max_len)
            if len(codec1)-max_len > 43 * 3:
                s = 43 * 3
            else:
                s = 0

        mode = self.mode
        if mode == 'all':
            tokens = [self.bos_token] + codec1[s: s + max_len] + [self.sep_token] + codec2[s: (s+max_len)] + [self.eos_token]
            valid_len = len(tokens)
            loss_mask = [1] * valid_len + [0] * (max_len * 2 - valid_len + 2)

        elif mode == 'single':
            tokens = [self.bos_token] + codec1[s: s + max_len] + [self.eos_token]
            valid_len = len(tokens)
            loss_mask = [1] * valid_len + [0] * (max_len - valid_len + 2)

        tokens = torch.tensor(tokens)
        attention_mask = torch.ones((len(tokens),), dtype=torch.bool)
        #loss_mask = [1] * valid_len + [0] * (max_len * 7 - valid_len + 4)
        #loss_mask = [1] * valid_len + [0] * (max_len - valid_len + 2)
        loss_mask = torch.tensor(loss_mask)
        attention_mask[0: valid_len] = False

        labels = tokens.long()
        labels[loss_mask == 0] = -100

        results = {
            "inputs": tokens[:-1].long(),
            "attention_masks": attention_mask[:-1],
            "labels": labels[1: ],
            "loss_masks": loss_mask[1: ],
            "prompt_len": 0,
            "valid_len": valid_len,
            "path": file_path,
        }

        return results



    def get_item(self, idx):
        pkl_file = str(self.files[idx])

        with open(pkl_file, 'rb') as f:
            res = pickle.load(f)
        c1_len = len(res[0].squeeze().tolist())
        ress = [res]
        while c1_len < self.max_seq_len:
            idx_ = random.randint(0, len(self.files)-1)
            pkl_file2 = str(self.files[idx_])
            with open(pkl_file2, 'rb') as f:
                res2 = pickle.load(f)
            ress.append(res2)
            c1_len = c1_len + len(res2[0].squeeze().tolist())

        results = self.process(ress, pkl_file)

        return results

    def get_item2(self, idx):
        pkl_file = str(self.files[idx])

        with open(pkl_file, 'rb') as f:
            res = pickle.load(f)
        c1_len = len(res.squeeze()[-1].tolist())
        ress = [res]
        while c1_len < self.max_seq_len:
            idx_ = random.randint(0, len(self.files)-1)
            pkl_file2 = str(self.files[idx_])
            with open(pkl_file2, 'rb') as f:
                res2 = pickle.load(f)
            ress.append(res2)
            c1_len = c1_len + len(res2.squeeze()[-1].tolist())

        results = self.process2(ress, pkl_file)

        return results


    def __getitem__(self, idx):
        #try:
        if self.data_type == 'pyramid':
            return self.get_item(idx)
        elif self.data_type == 'dac':
            return self.get_item2(idx)
        #except Exception as e:
        #    print(f"Error loading {self.files[idx]}: {e}")
        #    return None


class SVSLMDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: SVSLMDataset,
        val_dataset: SVSLMDataset,
        batch_size: int = 32,
        num_workers: int = 4,
        val_batch_size: Optional[int] = None,
    ):
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    import time
    #dataset = SVSLMDataset("/data/chenjianyi/data/LMD_6tracks/new_pyramid_p222_n111/vq_train_filelist.txt")
    dataset = SVSLMDataset("/data/chenjianyi/data/LMD_6tracks/new_DAC_512_2/vq_train_filelist.txt", data_type='dac', mode='single')
    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=True
    )

    max_len = 0
    _dict = {}
    for batch in dataloader:
        t1 = time.time()
        valid_len = batch['valid_len'][0].item()
        print(batch['inputs'].size())
        print(time.time() - t1)
        continue
        max_len = max(valid_len, max_len)
        if valid_len not in _dict:
            _dict[valid_len] = 0
        _dict[valid_len] += 1
    print(max_len, 111)
    _list = [(k, _dict[k]) for k in _dict]
    _list = sorted(_list, key=lambda x: x[0], reverse=True)
    print(_list[0: 10])
    print(_list[-10:])
