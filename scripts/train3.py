import os, sys
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
import math
import julius

import argbind
import torch, torchaudio
from audiotools import AudioSignal
from audiotools import ml
from audiotools.core import util
from audiotools.data import transforms
from audiotools.data.datasets import AudioDataset
from audiotools.data.datasets import AudioLoader
from audiotools.data.datasets import ConcatDataset
from audiotools.ml.decorators import timer
from audiotools.ml.decorators import Tracker
from audiotools.ml.decorators import when
from torch.utils.tensorboard import SummaryWriter

import dac

warnings.filterwarnings("ignore", category=UserWarning)

# Enable cudnn autotuner to speed up training
# (can be altered by the funcs.seed function)
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))
# Uncomment to trade memory for speed.

# Optimizers
AdamW = argbind.bind(torch.optim.AdamW, "generator", "discriminator")
Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)


@argbind.bind("generator", "discriminator")
def ExponentialLR(optimizer, gamma: float = 1.0):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)


# Models
PyramidCodec = argbind.bind(dac.model.PyramidCodec)
Discriminator = argbind.bind(dac.model.Discriminator)

# Data
AudioDataset = argbind.bind(AudioDataset, "train", "val")
AudioLoader = argbind.bind(AudioLoader, "train", "val")

# Transforms
filter_fn = lambda fn: hasattr(fn, "transform") and fn.__qualname__ not in [
    "BaseTransform",
    "Compose",
    "Choose",
]
tfm = argbind.bind_module(transforms, "train", "val", filter_fn=filter_fn)

# Loss
filter_fn = lambda fn: hasattr(fn, "forward") and "Loss" in fn.__name__
losses = argbind.bind_module(dac.nn.loss, filter_fn=filter_fn)

stft_window_lengths = [[2048, 1024], [1024, 512], [2048, 512]]
mel_window_lengths = [[512, 1024, 2048], [128, 256, 512, 1024, 2048], [32, 64, 128, 256, 512, 1024, 2048]]
n_mels = [[80, 160, 320], [20, 40, 80, 160, 320], [5, 10, 20, 40, 80, 160, 320]]
mel_fmins = [[0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
mel_fmaxs = [[None, None, None], [None, None, None, None, None], [None, None, None, None, None, None, None]]

def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


@argbind.bind("train", "val")
def build_transform(
    augment_prob: float = 1.0,
    preprocess: list = ["Identity"],
    augment: list = ["Identity"],
    postprocess: list = ["Identity"],
):
    to_tfm = lambda l: [getattr(tfm, x)() for x in l]
    preprocess = transforms.Compose(*to_tfm(preprocess), name="preprocess")
    augment = transforms.Compose(*to_tfm(augment), name="augment", prob=augment_prob)
    postprocess = transforms.Compose(*to_tfm(postprocess), name="postprocess")
    transform = transforms.Compose(preprocess, augment, postprocess)
    return transform


@argbind.bind("train", "val", "test")
def build_dataset(
    sample_rate: int,
    folders: dict = None,
):
    # Give one loader per key/value of dictionary, where
    # value is a list of folders. Create a dataset for each one.
    # Concatenate the datasets with ConcatDataset, which
    # cycles through them.
    datasets = []
    for _, v in folders.items():
        transform = build_transform()
        loader = AudioLoader(sources=v, transform=transform)
        dataset = AudioDataset(loader, sample_rate, transform=transform)
        datasets.append(dataset)

    dataset = ConcatDataset(datasets)
    dataset.transform = transform
    return dataset


@dataclass
class State:
    generator: PyramidCodec
    optimizer_g: AdamW
    scheduler_g: ExponentialLR

    discriminator: Discriminator
    optimizer_d: AdamW
    scheduler_d: ExponentialLR

    stft_loss: losses.MultiScaleSTFTLoss
    mel_loss: losses.MelSpectrogramLoss
    gan_loss: losses.GANLoss
    waveform_loss: losses.L1Loss

    train_data: AudioDataset
    val_data: AudioDataset

    tracker: Tracker


@argbind.bind(without_prefix=True)
def load(
    args,
    accel: ml.Accelerator,
    tracker: Tracker,
    save_path: str,
    resume: bool = False,
    tag: str = "latest",
    load_weights: bool = True,
):
    generator, g_extra = None, {}
    discriminator, d_extra = None, {}

    if resume:
        kwargs = {
            "folder": f"{save_path}/{tag}",
            "map_location": "cpu",
            "package": not load_weights,
            "strict": False,
        }
        tracker.print(f"Resuming from {str(Path('.').absolute())}/{kwargs['folder']}")
        if (Path(kwargs["folder"]) / "dac").exists():
            generator, g_extra = PyramidCodec.load_from_folder(**kwargs)
        if (Path(kwargs["folder"]) / "discriminator").exists():
            discriminator, d_extra = Discriminator.load_from_folder(**kwargs)

    generator = PyramidCodec(use_spec_input=True) if generator is None else generator
    discriminator = Discriminator() if discriminator is None else discriminator

    tracker.print(generator)
    tracker.print(discriminator)

    generator = accel.prepare_model(generator)
    discriminator = accel.prepare_model(discriminator)

    with argbind.scope(args, "generator"):
        optimizer_g = AdamW(generator.parameters(), use_zero=accel.use_ddp)
        scheduler_g = ExponentialLR(optimizer_g)
    with argbind.scope(args, "discriminator"):
        optimizer_d = AdamW(discriminator.parameters(), use_zero=accel.use_ddp)
        scheduler_d = ExponentialLR(optimizer_d)

    #if "optimizer.pth" in g_extra:
    #    optimizer_g.load_state_dict(g_extra["optimizer.pth"])
    #if "scheduler.pth" in g_extra:
    #    scheduler_g.load_state_dict(g_extra["scheduler.pth"])
    #if "tracker.pth" in g_extra:
    #    tracker.load_state_dict(g_extra["tracker.pth"])

    #if "optimizer.pth" in d_extra:
    #    optimizer_d.load_state_dict(d_extra["optimizer.pth"])
    #if "scheduler.pth" in d_extra:
    #    scheduler_d.load_state_dict(d_extra["scheduler.pth"])

    sample_rate = accel.unwrap(generator).sample_rate
    with argbind.scope(args, "train"):
        train_data = build_dataset(sample_rate)
        #train_data.transform = None
    with argbind.scope(args, "val"):
        val_data = build_dataset(sample_rate)
        #val_data.transform = None

    waveform_loss = losses.L1Loss()
    stft_loss = losses.MultiScaleSTFTLoss()
    mel_loss = losses.MelSpectrogramLoss()
    gan_loss = losses.GANLoss(discriminator)

    return State(
        generator=generator,
        optimizer_g=optimizer_g,
        scheduler_g=scheduler_g,
        discriminator=discriminator,
        optimizer_d=optimizer_d,
        scheduler_d=scheduler_d,
        waveform_loss=waveform_loss,
        stft_loss=stft_loss,
        mel_loss=mel_loss,
        gan_loss=gan_loss,
        tracker=tracker,
        train_data=train_data,
        val_data=val_data,
    )


def get_multiscale_audio(audio, sr=22050):
    res = [audio]
    for i in range(4):
        t_sr = int(sr / 2 ** (i+1))
        audio1 = julius.resample_frac(audio, sr, t_sr)
        audio2 = julius.resample_frac(audio1, t_sr, sr)
        res.append(audio2)
    L = min([a.size(-1) for a in res])
    res = [a[..., 0: L] for a in res]
    res = res[::-1]
    return torch.cat(res, dim=0)

@timer()
@torch.no_grad()
def val_loop(batch, state, accel):
    state.generator.eval()
    batch = util.prepare_batch(batch, accel.device)
    #signal = state.val_data.transform(
    #    batch["signal"].clone(), **batch["transform_args"]
    #)
    signal = batch["signal"]

    #K = 2 ** math.floor(math.log(T, 2))
    #ori_audio_data = signal.audio_data
    #signal.audio_data = get_multiscale_audio(signal.audio_data)
    bs, _, T = signal.audio_data.size()
    K = math.floor(T / 16384) * 16384
    signal.audio_data = signal.audio_data[:, :, 0: K]

    #gt_audio = batch["m_signal"].clone()[..., 0: K]
    #gt_audio = torch.cat([gt_audio[:, i, ...] for i in range(gt_audio.size(1))])
    #gt_signal = AudioSignal(gt_audio, signal.sample_rate)

    out = state.generator(signal.audio_data, signal.sample_rate)
    recons = AudioSignal(out["audio"], signal.sample_rate)

    ##signal.audio_data = torch.cat([signal.audio_data] * 5, dim=0)
    #gt_audios = batch["m_signal"].clone()[..., 0: K]  # (bs, N, 1, K)
    #gt_audios = [gt_audios[:, i, ...] for i in range(gt_audios.size(1))] # [(bs, 1, K), ...]
    #gt_audios[-1] = signal.audio_data

    gt_audios = [signal.audio_data] * 3
    recon_audios = out["audio"].split(bs, dim=0) # [(bs, 1, K), ...]
    N = len(gt_audios)

    loss_dict = {}
    for i in range(len(gt_audios)):

        state.stft_loss.window_lengths = stft_window_lengths[i]
        state.mel_loss.window_lengths = mel_window_lengths[i]
        state.mel_loss.n_mels = n_mels[i]
        state.mel_loss.mel_fmin = mel_fmins[i]
        state.mel_loss.mel_fmax = mel_fmaxs[i]

        ratio = 1 / (2 ** (N - i - 1))
        sr = int( signal.sample_rate // (2 ** (N - i - 1)) )
        mel_fmax = [sr] * 6 + [signal.sample_rate]
        #gt_audio_i = torch.nn.functional.interpolate(gt_audios[i], scale_factor=ratio) #(bs, 1, K')
        #recon_audio_i= torch.nn.functional.interpolate(recon_audios[i], scale_factor=ratio)
        gt_audio_i = gt_audios[i]
        recon_audio_i = recon_audios[i]
        gt_signal = AudioSignal(gt_audio_i, sr)
        recons = AudioSignal(recon_audio_i, sr)

        loss_dict.setdefault("loss", []).append(state.mel_loss(recons, gt_signal))
        loss_dict["mel/loss(%s)" % i] = state.mel_loss(recons, gt_signal) #, mel_fmax=mel_fmax)
        loss_dict.setdefault("mel/loss", []).append(loss_dict["mel/loss(%s)" % i])
        loss_dict["stft/loss(%s)" % i] = state.stft_loss(recons, gt_signal)
        loss_dict.setdefault("stft/loss", []).append(loss_dict["stft/loss(%s)" % i])
        loss_dict.setdefault("waveform/loss", []).append(state.waveform_loss(recons, gt_signal))

    weights = [0.2, 0.3, 0.5]
    for k in loss_dict:
        if isinstance(loss_dict[k], list):
            loss_dict[k] = [tmp * weights[ii] for ii, tmp in enumerate(loss_dict[k])]
            loss_dict[k] = sum(loss_dict[k])
    #loss_dict["loss"] = sum([v * loss_dict[k] for k, v in lambdas.items() if k in loss_dict]).mean

    return loss_dict

@timer()
def train_loop(state, batch, accel, lambdas):
    state.generator.train()
    state.discriminator.train()
    output = {}

    batch = util.prepare_batch(batch, accel.device)
    #with torch.no_grad():
    #    signal = state.train_data.transform(
    #        batch["signal"].clone(), **batch["transform_args"]
    #    )

    signal = batch["signal"]

    #K = 2 ** math.floor(math.log(T, 2))
    #ori_audio_data = signal.audio_data
    #signal.audio_data = get_multiscale_audio(signal.audio_data)
    bs, _, T = signal.audio_data.size()
    K = math.floor(T / 16384) * 16384
    signal.audio_data = signal.audio_data[:, :, 0: K]

    gt_audio = batch["m_signal"].clone()[..., 0: K]
    gt_audio = torch.cat([gt_audio[:, i, ...] for i in range(gt_audio.size(1))])
    gt_signal = AudioSignal(gt_audio, signal.sample_rate)

    with accel.autocast():
        out = state.generator(signal.audio_data, signal.sample_rate)
        recons = AudioSignal(out["audio"], signal.sample_rate)
        commitment_loss = out["vq/commitment_loss"]
        codebook_loss = out["vq/codebook_loss"]

    with accel.autocast():
        recons_dis = recons.clone()
        recons_dis.audio_data = recons_dis.audio_data[-bs: , ...]
        output["adv/disc_loss"] = state.gan_loss.discriminator_loss(recons_dis, signal)
        #output["adv/disc_loss"] = state.gan_loss.discriminator_loss(recons, gt_signal)

    state.optimizer_d.zero_grad()
    accel.backward(output["adv/disc_loss"])
    accel.scaler.unscale_(state.optimizer_d)
    output["other/grad_norm_d"] = torch.nn.utils.clip_grad_norm_(
        state.discriminator.parameters(), 10.0
    )
    accel.step(state.optimizer_d)
    state.scheduler_d.step()

    #signal.audio_data = torch.cat([signal.audio_data] * 5, dim=0)

    ##gt_audios = batch["m_signal"].clone()[..., 0: K].split(bs, dim=0)
    #gt_audios = batch["m_signal"].clone()[..., 0: K]  # (bs, N, 1, K)
    #gt_audios = [gt_audios[:, i, ...] for i in range(gt_audios.size(1))]  # [(bs, 1, K), ...]
    #gt_audios[-1] = signal.audio_data

    gt_audios = [signal.audio_data] * 3
    recon_audios = out["audio"].split(bs, dim=0)  # [(bs, 1, K), ...]
    N = len(gt_audios)

    for i in range(len(gt_audios)):

        state.stft_loss.window_lengths = stft_window_lengths[i]
        state.mel_loss.window_lengths = mel_window_lengths[i]
        state.mel_loss.n_mels = n_mels[i]
        state.mel_loss.mel_fmin = mel_fmins[i]
        state.mel_loss.mel_fmax = mel_fmaxs[i]

        ratio = 1 / (2 ** (N - i - 1))
        sr = int( signal.sample_rate // (2 ** (N - i - 1)) )
        mel_fmax = [sr] * 6 + [signal.sample_rate]
        #gt_audio_i = torch.nn.functional.interpolate(gt_audios[i], scale_factor=ratio) # (bs, 1, K')
        #recon_audio_i= torch.nn.functional.interpolate(recon_audios[i], scale_factor=ratio) #(bs, 1, K')
        gt_audio_i = gt_audios[i]
        recon_audio_i = recon_audios[i]
        gt_signal = AudioSignal(gt_audio_i, sr)
        recons = AudioSignal(recon_audio_i, sr)
        with accel.autocast():
            output["stft/loss(%s)" % i] = state.stft_loss(recons, gt_signal)
            output.setdefault("stft/loss", []).append(output["stft/loss(%s)" % i])
            output["mel/loss(%s)" % i] = state.mel_loss(recons, gt_signal) #, mel_fmax=mel_fmax)
            output.setdefault("mel/loss", []).append(output["mel/loss(%s)" % i])
            output.setdefault("waveform/loss", []).append(state.waveform_loss(recons, gt_signal))
            if i == N - 1:
                (
                    gen_loss,
                    feat_loss,
                ) = state.gan_loss.generator_loss(recons, gt_signal)
                output["adv/gen_loss"] = gen_loss
                output["adv/feat_loss"] = feat_loss
            #output.setdefault("adv/gen_loss", []).append(gen_loss)
            #output.setdefault("adv/feat_loss", []).append(feat_loss)
            output.setdefault("vq/commitment_loss", []).append(commitment_loss)
            output.setdefault("vq/codebook_loss", []).append(codebook_loss)
    weights = [0.2, 0.3, 0.5]
    for k in output:
        if isinstance(output[k], list):
            output[k] = [tmp * weights[ii] for ii, tmp in enumerate(output[k])] 
            output[k] = sum(output[k])
    output["loss"] = sum([v * output[k] for k, v in lambdas.items() if k in output]).mean()

    state.optimizer_g.zero_grad()
    accel.backward(output["loss"])
    accel.scaler.unscale_(state.optimizer_g)
    output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
        state.generator.parameters(), 5e2
    ) # 1e3
    accel.step(state.optimizer_g)
    state.scheduler_g.step()
    accel.update()

    output["other/learning_rate"] = state.optimizer_g.param_groups[0]["lr"]
    output["other/batch_size"] = signal.batch_size * accel.world_size

    return {k: v for k, v in sorted(output.items())}


def checkpoint(state, save_iters, save_path):
    metadata = {"logs": state.tracker.history}

    tags = ["latest"]
    state.tracker.print(f"Saving to {str(Path('.').absolute())}")
    if state.tracker.is_best("val", "mel/loss"):
        state.tracker.print(f"Best generator so far")
        tags.append("best")
    if state.tracker.step in save_iters:
        tags.append(f"{state.tracker.step // 1000}k")

    for tag in tags:
        generator_extra = {
            "optimizer.pth": state.optimizer_g.state_dict(),
            "scheduler.pth": state.scheduler_g.state_dict(),
            "tracker.pth": state.tracker.state_dict(),
            "metadata.pth": metadata,
        }
        accel.unwrap(state.generator).metadata = metadata
        accel.unwrap(state.generator).save_to_folder(
            f"{save_path}/{tag}", generator_extra, package=False
        )
        discriminator_extra = {
            "optimizer.pth": state.optimizer_d.state_dict(),
            "scheduler.pth": state.scheduler_d.state_dict(),
        }
        accel.unwrap(state.discriminator).save_to_folder(
            f"{save_path}/{tag}", discriminator_extra, package=False
        )

@torch.no_grad()
def save_samples(state, val_idx, writer, save_path, val_dataloader):
    state.tracker.print("Saving audio samples to TensorBoard")
    state.generator.eval()

    """
    samples = [state.val_data[idx] for idx in val_idx]
    batch = state.val_data.collate(samples)
    batch = util.prepare_batch(batch, accel.device)
    #signal = state.train_data.transform(
    #    batch["signal"].clone(), **batch["transform_args"]
    #)
    signal = batch["signal"]
    """

    for bidx, batch in enumerate(val_dataloader):
        signal = batch["signal"]
        batch = util.prepare_batch(batch, accel.device)
        #signal = batch["signal"]

        #ori_audio_data = signal.audio_data
        #signal.audio_data = get_multiscale_audio(signal.audio_data)
        bs, _, T = signal.audio_data.size()
        #K = 2 ** math.floor(math.log(T, 2))
        K = math.floor(T / 16384) * 16384
        signal.audio_data = signal.audio_data[:, :, 0: K]

        gt_audio = batch["m_signal"].clone()[:, :, 0: K]
        gt_audio = torch.cat([gt_audio[:, i, ...] for i in range(gt_audio.size(1))])

        out = state.generator(signal.audio_data, signal.sample_rate)

        gt_audios = batch["m_signal"].clone()[..., 0: K]  # (bs, N, 1, K)
        gt_audios = [gt_audios[:, i, ...] for i in range(gt_audios.size(1))]  # [(bs, 1, K), ...]
        gt_audios[-1] = signal.audio_data

        recon_audios = out["audio"].split(bs, dim=0)  #[(bs, 1, K), ...]
        N = len(recon_audios)

        os.makedirs(f"{save_path}/samples/", exist_ok=True)
        for i in range(len(gt_audios)):
            ratio = 1 / (2 ** (N - i - 1))
            sr = int( signal.sample_rate // (2 ** (N - i - 1)) )
            #gt_audio_i = torch.nn.functional.interpolate(gt_audios[i], scale_factor=ratio) #(bs, 1, K)
            #recon_audio_i= torch.nn.functional.interpolate(recon_audios[i], scale_factor=ratio)
            gt_audio_i = gt_audios[i]
            recon_audio_i = recon_audios[i]
            gt_signal = AudioSignal(gt_audio_i, signal.sample_rate)
            recons = AudioSignal(recon_audio_i, signal.sample_rate)

            audio_dict = {"recons": recons}
            if state.tracker.step == 0:
                #audio_dict["signal"] = gt_signal
                audio_dict["signal"] = signal

            for k, v in audio_dict.items():
                os.makedirs(f"{save_path}/samples/{k}", exist_ok=True)
                for nb in range(v.batch_size):
                    idx = i * v.batch_size + nb
                    v[nb].cpu().write_audio_to_tb(
                        f"{k}/sample_{idx}.wav", writer, state.tracker.step
                    )
                    if (i == len(gt_audios) - 1):
                        idx = bidx * v.batch_size + nb
                        torchaudio.save(f"{save_path}/samples/{k}/sample_{idx}.wav", v[nb].audio_data.cpu().squeeze(0), signal.sample_rate)
        if bidx > 10:
            break


def validate(state, val_dataloader, accel):
    for batch in val_dataloader:
        output = val_loop(batch, state, accel)
    # Consolidate state dicts if using ZeroRedundancyOptimizer
    if hasattr(state.optimizer_g, "consolidate_state_dict"):
        state.optimizer_g.consolidate_state_dict()
        state.optimizer_d.consolidate_state_dict()
    return output


@argbind.bind(without_prefix=True)
def train(
    args,
    accel: ml.Accelerator,
    seed: int = 0,
    save_path: str = "ckpt",
    num_iters: int = 250000,
    save_iters: list = [10000, 50000, 100000, 200000],
    sample_freq: int = 10000,
    valid_freq: int = 1000,
    batch_size: int = 12,
    val_batch_size: int = 10,
    num_workers: int = 8,
    val_idx: list = [0, 1, 2, 3, 4, 5, 6, 7],
    lambdas: dict = {
        "mel/loss": 100.0,
        "adv/feat_loss": 2.0,
        "adv/gen_loss": 1.0,
        "vq/commitment_loss": 0.25,
        "vq/codebook_loss": 1.0,
    },
):
    util.seed(seed)
    Path(save_path).mkdir(exist_ok=True, parents=True)
    writer = (
        SummaryWriter(log_dir=f"{save_path}/logs") if accel.local_rank == 0 else None
    )
    tracker = Tracker(
        writer=writer, log_file=f"{save_path}/log.txt", rank=accel.local_rank
    )

    state = load(args, accel, tracker, save_path, resume=False, load_weights=True)
    train_dataloader = accel.prepare_dataloader(
        state.train_data,
        start_idx=state.tracker.step * batch_size,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=state.train_data.collate,
    )
    print('train_dataloader', len(train_dataloader)) 
    train_dataloader = get_infinite_loader(train_dataloader)
    
    val_dataloader = accel.prepare_dataloader(
        state.val_data,
        start_idx=0,
        num_workers=num_workers,
        batch_size=val_batch_size,
        collate_fn=state.val_data.collate,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    print('val_dataloader', len(val_dataloader))
    # Wrap the functions so that they neatly track in TensorBoard + progress bars
    # and only run when specific conditions are met.
    global train_loop, val_loop, validate, save_samples, checkpoint
    train_loop = tracker.log("train", "value", history=False)(
        tracker.track("train", num_iters, completed=state.tracker.step)(train_loop)
    )
    val_loop = tracker.track("val", len(val_dataloader))(val_loop)
    validate = tracker.log("val", "mean")(validate)

    # These functions run only on the 0-rank process
    save_samples = when(lambda: accel.local_rank == 0)(save_samples)
    checkpoint = when(lambda: accel.local_rank == 0)(checkpoint)

    with tracker.live:
        for epoch in range(100000):
            for tracker.step, batch in enumerate(train_dataloader, start=tracker.step):
                train_loop(state, batch, accel, lambdas)

                last_iter = (
                    tracker.step == num_iters - 1 if num_iters is not None else False
                )
                if tracker.step % sample_freq == 0 or last_iter:
                    print('Saveing samples begining...')
                    save_samples(state, val_idx, writer, save_path, val_dataloader)
                    print('Saveing samples finished.')

                if tracker.step % valid_freq == 0 or last_iter:
                    validate(state, val_dataloader, accel)
                    checkpoint(state, save_iters, save_path)
                    # Reset validation progress bar, print summary since last validation.
                    tracker.done("val", f"Iteration {tracker.step}")

                if last_iter:
                    break


if __name__ == "__main__":
    args = argbind.parse_args()
    args["args.debug"] = int(os.getenv("LOCAL_RANK", 0)) == 0
    with argbind.scope(args):
        with Accelerator() as accel:
            if accel.local_rank != 0:
                sys.tracebacklimit = 0
            train(args, accel)
