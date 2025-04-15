"""
Sample from a trained model
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pickle
from contextlib import nullcontext
import torch
import tiktoken
import pickle
from model import GPTConfig, GPT

from dataset import SVSLMDataset
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir1 = '../weights/gpt/out_single' # ignored if init_from is not 'resume'
out_dir2 = '../weights/gpt/out_single_s2'
out_dir3 = '../weights/gpt/out_single_s3'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 #10 # number of samples to draw
max_new_tokens = 1000 #500 # number of tokens generated in each sample
temperature = 1.3 #1.0 #0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 100 #20 #200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' #'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir1, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    print(checkpoint['model_args'])
    gptconf = GPTConfig(**checkpoint['model_args'])
    print(gptconf)
    model1 = GPT(gptconf)
    print('After loading model')
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model1.load_state_dict(state_dict, strict=False)

    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir2, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    print(checkpoint['model_args'])
    gptconf = GPTConfig(**checkpoint['model_args'])
    print(gptconf)
    model2 = GPT(gptconf)
    print('After loading model')
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model2.load_state_dict(state_dict, strict=False)

    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir3, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    print(checkpoint['model_args'])
    gptconf = GPTConfig(**checkpoint['model_args'])
    print(gptconf)
    model3 = GPT(gptconf)
    print('After loading model')
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model3.load_state_dict(state_dict, strict=False)


elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model1.eval()
model1.to(device)
model2.eval()
model2.to(device)
model3.eval()
model3.to(device)

if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
"""
# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
"""

#dataset = SVSLMDataset("/data/chenjianyi/data/LMD_6tracks/new_pyramid_p222_n111/vq_test_filelist.txt", max_seq_len=2000, data_type='pyramid', mode='all', is_train=False)
#dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

# run generation
#for i, batch in enumerate(val_dataloader):
for i in range(10):
    #if i <= 27:
    #    continue
    #x = batch['inputs'].to(device)[:, 0: 51]
    #c2 = batch['codec2'].to(device)[:, 0: 50*2]
    #c3 = batch['codec3'].to(device)[:, 0: 50*4]
    #valid_len = batch['valid_len'][0]
    #max_new_tokens = min(200, valid_len-20)
    x = torch.tensor([1025]).unsqueeze(0).to(device)
    c2 = torch.empty(1, 0).to(device).long()
    c3 = torch.empty(1, 0).to(device).long()
    valid_len = (2000 + 4000 + 8000) + 4 
    max_new_tokens = valid_len
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                print(x, valid_len)

                print('Stage1 ...', x.size())
                print(x.size())
                y = model1.generate(x, 2002 - x.size(1), temperature=temperature, top_k=top_k)
                print(y)
                yy1 = y[0].tolist()[0: 2001] + [1027]
                x = torch.tensor(yy1).unsqueeze(0).cuda()

                x = torch.cat([x, c2], dim=1)
                print('Stage2 ...', x.size())
                y = model2.generate(x, 4001-c2.size(1), temperature=temperature, top_k=top_k)
                print(y)
                yy2 = y[0].tolist()[0: 2002+4000] + [1027]
                x = torch.tensor(yy2).unsqueeze(0).cuda()
                
                x = torch.cat([x, c3], dim=1)
                print('Stage3 ...', x.size())
                y = model3.generate(x, 8001-c3.size(1), temperature=temperature, top_k=top_k)
                print(y)
                yy3 = y[0].tolist()[0: 2002+4000+1+8000] + [1027]

                y = yy3
                y_ = [(i, y[i]) for i in range(len(y))]
                print([(i, y[i]) for i in range(len(y_)) if y[i] == 1027])
                #gt_y = batch['inputs'].to(device)[0, 0: len(y)].tolist()

                y = y[1: -1]
                #gt_y = gt_y[1: -1]

                y1 = y[0: 2000]
                y2 = y[2001: 2001+4000]
                y3 = y[2001+4000+1: 2001+4000+1+8000]

                #gt_y1 = gt_y[0: 2000]
                #gt_y2 = gt_y[2001: 2001+4000]
                #gt_y3 = gt_y[2001+4000+1: 2001+4000+1+8000]

                print(y[1995: 2005], y[5995: 6005], '???')
                print(min(y1), max(y1), min(y2), max(y2), min(y3), max(y3), '!!!', len(y1), len(y2), len(y3))

                if max(y1) > 1023 or max(y2) > 1023 or max(y3) > 1023:
                    print("Error!!!!")
                    continue

                y = [torch.tensor(y1).unsqueeze(0).unsqueeze(0), torch.tensor(y2).unsqueeze(0).unsqueeze(0), torch.tensor(y3).unsqueeze(0).unsqueeze(0)]
                #gt_y = [torch.tensor(gt_y1).unsqueeze(0).unsqueeze(0), torch.tensor(gt_y2).unsqueeze(0).unsqueeze(0), torch.tensor(gt_y3).unsqueeze(0).unsqueeze(0)]
                print('---------------')
    os.makedirs('results', exist_ok=True)
    with open('results/%s.pkl' % i, 'wb') as f:
        pickle.dump([y, y], f)
