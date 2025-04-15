# PyramidCodec

This is a PyTorch/GPU implementation of the EMNLP 2024 (findings) paper [PyramidCodec: Hierarchical Codec for Long-form Music Generation in Audio Domain](https://aclanthology.org/2024.findings-emnlp.246.pdf).
Demo page can be found at [demo](https://pyramidcodec.github.io/).

## Overview

PyramidCodec is a hierarchical codec designed for long-form music generation in the audio domain. This approach addresses the challenges of generating well-structured music compositions that span several minutes by utilizing a hierarchical discrete representation of audio.

## Features

- **Hierarchical Discrete Representation**: Utilizes residual vector quantization on multiple feature levels.
- **Compact Token Sequences**: Higher-level features have larger hop sizes, resulting in compact representations.
- **Hierarchical Training Strategy**: Gradually adds details with more levels of tokens to enhance composition quality.
- **Ultra-Long Music Generation**: Capable of generating music compositions up to several minutes long.

## Requirements

- Python 3.x
- Required libraries

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/chenjianyi/pyramidcodec
   cd pyramidcodec

2. Download checkpoints from [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/jchenil_connect_ust_hk/Eg_cNjcmzNtEmVI2nmdNY78B9n8qZNEMfybjjzo63y6khg?e=qN6Qjb), and put it in pyramidcodec directory.

3. Generation using a gpt model
   ```bash
   cd nanoGPT
   python3 sample.py

4. Decoding using pyramidcodec
   ```bash
   cd ..
   python3 scripts/get_samples_from_z.py --args.load conf/exps/pyramid_p222_n111.yml

