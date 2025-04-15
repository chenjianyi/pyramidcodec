import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

path = os.path.abspath(__file__)

sys.path.append(os.path.dirname(path))

sys.path.append(os.path.dirname(os.path.dirname(path)))

from pathlib import Path
import math, random, julius
import random
import argbind
import torch
from audiotools import AudioSignal
from audiotools.core import util
from audiotools.ml.decorators import Tracker
from train import Accelerator
from train import DAC
#from train_dac import Accelerator
#from train_dac import DAC
import dac
import pickle

#from dac.compare.encodec import Encodec

#Encodec = argbind.bind(Encodec)


#DAC = argbind.bind(dac.model.OrigDAC)

def load_state(
    accel: Accelerator,
    tracker: Tracker,
    save_path: str,
    tag: str = "latest",
    load_weights: bool = True,
    model_type: str = "dac",
    bandwidth: float = 24.0,
):
    kwargs = {
        "folder": f"{save_path}/{tag}",
        "map_location": "cpu",
        "package": not load_weights,
    }
    tracker.print(f"Resuming from {str(Path('.').absolute())}/{kwargs['folder']}")

    if model_type == "dac":
        #DAC = argbind.bind(dac.model.DAC)
        #print(DAC.n_codebooks)
        #print(DAC.paramids)
        #generator = DAC()
        #print(generator)
        #generator = generator.load(os.path.join(kwargs['folder'], 'dac', 'weights.pth'))
        #generator.load_state_dict(state_dict=ckpt)
        generator, _ = DAC.load_from_folder(**kwargs)
        #print(DAC.paramids)
    elif model_type == "encodec":
        generator = Encodec(bandwidth=bandwidth)

    generator = accel.prepare_model(generator)
    return generator


@torch.no_grad()
def process(signal, accel, generator, **kwargs):
    signal = signal.to(accel.device)

    """
    bs, d, T = signal.audio_data.size()
    K = math.ceil(T / 16384) * 16384
    #K = 2 ** math.ceil(math.log(T, 2))
    print(T, K)
    padding = torch.zeros(bs, d, K-T).to(accel.device)
    signal.audio_data = torch.cat([signal.audio_data, padding], dim=-1)
    """

    #K = 2 ** math.floor(math.log(T, 2))
    #signal.audio_data = signal.audio_data[:, :, 0: K]

    sample_rate = 22050
    signal = signal.to_mono()
    signal.audio_data = julius.resample_frac(signal.audio_data, signal.sample_rate, sample_rate)

    bs, d, T = signal.audio_data.size()
    K = math.floor(T / 16384) * 16384
    signal.audio_data = signal.audio_data[:, :, 0: K] 
    print(signal.audio_data.size())

    signal.sample_rate = sample_rate
    data = generator(signal.audio_data, signal.sample_rate, **kwargs)
    codes = data['codes']
    print('\n')
    #for code in codes:
    #    print(code[0, 0, 0: 20].tolist())
    #    print(code.size())
    print('recon:', data["audio"].size())
    recons = data["audio"][0, :, 0: T]
    recons = AudioSignal(recons, signal.sample_rate)
    recons = recons.normalize(signal.loudness())
    return recons.cpu(), codes


@argbind.bind(without_prefix=True)
@torch.no_grad()
def get_samples(
    accel,
    path: str = "/data/chenjianyi/code/descript-audio-codec/exps/pyramid_p222_n111_3/",
    #path: str = "/data/chenjianyi/code/descript-audio-codec/exps/DAC_512_2", 
    input: str = "/data/chenjianyi/data/LMD_6tracks/wav_render",
    output: str = "samples/pyramid_p222_n111_3",
    #output: str = "samples/DAC_512_2", 
    #output: str = "/data/chenjianyi/data/LMD_6tracks/pyramid_p222_n111",
    #output: str = "/data/chenjianyi/data/LMD_6tracks/DAC_512_2",
    model_type: str = "dac",
    model_tag: str = "best",  # "latest",
    bandwidth: float = 24.0,
    n_quantizers: int = None,
):
    tracker = Tracker(log_file=f"{path}/eval.txt", rank=accel.local_rank)
    generator = load_state(
        accel,
        tracker,
        save_path=path,
        model_type=model_type,
        bandwidth=bandwidth,
        tag=model_tag,
    )
    generator.eval()
    kwargs = {"n_quantizers": n_quantizers} if model_type == "dac" else {}

    audio_files = util.find_audio(input)
    print(len(audio_files))

    random.seed(31415926)
    random.shuffle(audio_files)
    audio_files = audio_files[0: 1000000]

    global process
    process = tracker.track("process", len(audio_files))(process)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    with tracker.live:
        for i in range(len(audio_files)):
            print(audio_files[i])
            signal = AudioSignal(audio_files[i])
            try:
                recons, codes = process(signal, accel, generator, **kwargs)
            except:
                continue
            signal = signal.cpu()
            des_path = os.path.join(output, audio_files[i].name.replace('.wav', '.pkl'))
            print(des_path)
            with open(des_path, 'wb') as f:
                pickle.dump(codes, f)

        tracker.done("test", f"N={len(audio_files)}")


if __name__ == "__main__":
    args = argbind.parse_args()
    print(args)
    with argbind.scope(args):
        with Accelerator() as accel:
            get_samples(accel)
