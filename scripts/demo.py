import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
path = os.path.abspath(__file__)

sys.path.append(os.path.dirname(path))

sys.path.append(os.path.dirname(os.path.dirname(path)))

import math
import torch, torchaudio
import dac
from audiotools import AudioSignal
from audiotools.data.datasets import AudioDataset
from audiotools.data.datasets import AudioLoader

from train import build_transform

def build_dataset(sample_rate, folders):
    loader = AudioLoader(sources=folders)
    transform = build_transform()
    dataset = AudioDataset(loader, sample_rate, transform=transform)
    dataset.transform = transform
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=dataset.collate)
    return dataset, dataloader

#dataset, dataloader = build_dataset(22050, '/data/chenjianyi/data/LMD_6tracks/wav_render_10s/')
#for data in dataloader:
#    print(data)
#    break

import julius

# Download a model
#model_path = dac.utils.download(model_type="44khz")
#model_path = '/ssddata/chenjianyi/code/descript-audio-codec/runs/2024-1-14/best/dac/weights.pth'
model_path = '/data/chenjianyi/code/descript-audio-codec/runs/0418_p3/best/dac/weights.pth'
model_dict = torch.load(model_path, "cpu")
metadata = model_dict["metadata"]
print(metadata, model_dict.keys())

model = dac.DAC.load(model_path).eval()

model.to('cuda')

# Load audio signal file
#path = '/ssddata/zheqid/descript-audio-codec/testwav/M.wav'
path = '/data/chenjianyi/data/LMD_6tracks/wav_render/0_8f0a8860f2f1dceec616eafbe07e1.wav'
path = '/data/chenjianyi/data/LMD_6tracks/wav_render_10s/0_7_712cd67360d037233105c9f190d2bafa/000.wav'
signal = AudioSignal(path)

# Encode audio signal as one long file
# (may run out of GPU memory on long files)
signal.to(model.device)
#signal.audio_data = signal.audio_data[:, :, 0: 44000]
sample_rate = 22050
#signal.audio_data = signal.audio_data.mean(1).unsqueeze(0)
signal = signal.to_mono()
signal.audio_data = julius.resample_frac(
    signal.audio_data, signal.sample_rate, sample_rate
)
signal.sample_rate = sample_rate

print(signal.audio_data.size(), 111, signal.sample_rate)
signal = signal.resample(signal.sample_rate)
bs, _, T = signal.audio_data.size()
#K = 2 ** math.floor(math.log(T, 2))
K = math.floor(T / 16384) * 16384
print(signal.audio_data.size(), 222)
signal.audio_data = signal.audio_data[:, :, 0: K]
print(signal.audio_data.size(), signal.sample_rate)
print(signal.audio_data)

with torch.no_grad():
    x, m = model.preprocess(signal.audio_data, signal.sample_rate)
    print(x.size())
    z, codes, latents, _, _ = model.encode(x, mel_data=m)
    for code in codes:
        print(code.size(), code[0, 0, 0: 60].tolist())

    # Decode audio signal
    y = model.decode(z).squeeze(0).cpu()
    print(y.size(), 'yyy')
    print(y)
    torchaudio.save('test111.wav', y[-1], sample_rate=22050)

    out = model(signal.audio_data, signal.sample_rate)
    recons = AudioSignal(out["audio"], signal.sample_rate)
    print(recons.audio_data.size())
    torchaudio.save('test222.wav', recons.audio_data[-1].cpu(), sample_rate=22050)

# Alternatively, use the `compress` and `decompress` functions
# to compress long files.
"""
signal = signal.cpu()
x = model.compress(signal)

# Save and load to and from disk
x.save("compressed.dac")
x = dac.DACFile.load("compressed.dac")

# Decompress it back to an AudioSignal
y = model.decompress(x)

# Write to file

y.write('output.wav')
"""
