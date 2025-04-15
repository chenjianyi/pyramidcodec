import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

path = os.path.abspath(__file__)

sys.path.append(os.path.dirname(path))

sys.path.append(os.path.dirname(os.path.dirname(path)))

from pathlib import Path
import math, random, julius
import random
import argbind
import torch, torchaudio
from audiotools import AudioSignal
from audiotools.core import util
from audiotools.ml.decorators import Tracker
from train import Accelerator
from train3 import PyramidCodec
#from train_dac import Accelerator
#from train_dac import PyramidCodec
import dac
import pickle

#from dac.compare.encodec import Encodec

#Encodec = argbind.bind(Encodec)


#PyramidCodec = argbind.bind(dac.model.OrigPyramidCodec)

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
        #PyramidCodec = argbind.bind(dac.model.PyramidCodec)
        #print(PyramidCodec.n_codebooks)
        #print(PyramidCodec.paramids)
        #generator = PyramidCodec()
        #print(generator)
        #generator = generator.load(os.path.join(kwargs['folder'], 'dac', 'weights.pth'))
        #generator.load_state_dict(state_dict=ckpt)
        print('1111', kwargs)
        print(PyramidCodec)
        generator, _ = PyramidCodec.load_from_folder(**kwargs)
        print(generator, 2222)
        #print(PyramidCodec.paramids)
    elif model_type == "encodec":
        generator = Encodec(bandwidth=bandwidth)

    generator = accel.prepare_model(generator)
    return generator


@torch.no_grad()
def process(z1s, z2s, accel, generator, **kwargs):
    q1s, q2s = [], []
    for i in range(3):
        z1 = z1s[i].to(accel.device)[:, :, :1995*(2**i)]  # 1995 396 405
        z2 = z2s[i].to(accel.device)[:, :, :1995*(2**i)]
        print(z1.max(), z1.min(), z1[:, :, -1], z1.size())
        print(z2.max(), z2.min(), z2[:, :, -1], z2.size())
        z1, _, _ = generator.encoder.quantizers[i].from_codes(z1)
        z2, _, _ = generator.encoder.quantizers[i].from_codes(z2)
        q1s.append(z1)
        q2s.append(z2)

    sample_rate = 22050

    data1 = generator.decode(q1s)
    recons1 = data1[-1, :, :]
    recons1 = AudioSignal(recons1, sample_rate)
    #recons1 = recons.normalize(signal.loudness())
    data2 = generator.decode(q2s)
    recons2 = data2[-1, :, :]
    recons2 = AudioSignal(recons2, sample_rate)

    return recons1.cpu(), recons2.cpu()


@argbind.bind(without_prefix=True)
@torch.no_grad()
def get_samples(
    accel,
    path: str = "./weights/pyramid_p222_n111",
    #path: str = "/data/chenjianyi/code/descript-audio-codec/exps/PyramidCodec_512_9", 
    input: str = "./nanoGPT/results",
    output: str = "results",
    #output: str = "samples/PyramidCodec_512_9_q9", 
    #output: str = "/data/chenjianyi/data/LMD_6tracks/pyramid_p222_n111_3",
    model_type: str = "dac",
    model_tag: str = "best",  # "latest",
    bandwidth: float = 24.0,
    n_quantizers: int = None,
):
    os.makedirs(output, exist_ok=True)
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
    print(len(audio_files), input, '???')

    random.seed(31415926)
    random.shuffle(audio_files)
    audio_files = audio_files[0: 10000]

    global process
    process = tracker.track("process", len(audio_files))(process)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    recon_out = Path(os.path.join(output, 'gen'))
    #gt_out = Path(os.path.join(output, 'gt'))
    recon_out.mkdir(parents=True, exist_ok=True)
    #gt_out.mkdir(parents=True, exist_ok=True)

    with tracker.live:
        for i in range(len(audio_files)):
            print(audio_files[i])
            name = os.path.basename(audio_files[i]).replace('.pkl', '.wav')
            with open(audio_files[i], 'rb') as f:
                gt_codes, gen_codes = pickle.load(f)

            try:
                gt, gen_out = process(gt_codes, gen_codes, accel, generator, **kwargs)
            except:
                continue
            gen_out = gen_out.cpu()
            #gen_out.write(recon_out / ('%04d.wav' % i))
            gen_out.write(recon_out / name)
            #signal.write(gt_out / audio_files[i].name)
            #gt.write(gt_out / ('%04d.wav' % i))
            #gt.write(gt_out / name)

        tracker.done("test", f"N={len(audio_files)}")


if __name__ == "__main__":
    args = argbind.parse_args()
    print(args)
    with argbind.scope(args):
        with Accelerator() as accel:
            get_samples(accel)
