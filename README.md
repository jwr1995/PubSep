# PubSep 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/on-time-domain-conformer-models-for-monaural/speech-separation-on-whamr)](https://paperswithcode.com/sota/speech-separation-on-whamr?p=on-time-domain-conformer-models-for-monaural)

This repository provides training and evalution scripts for the DNN speech separation models described in various papers 
 * "Utterance Weighted Multi-Dilation Temporal Convolutional Networks for Monaural Speech Dereverberation" - https://ieeexplore.ieee.org/document/9914752
 * "Deformable Temporal Convolutional Networks for Monaural Noisy Reverberant Speech Separation" - https://ieeexplore.ieee.org/document/10095230
 * "On Time Domain Conformer Models for Monaural Speech Separation in Noisy Reverberant Acoustic Environments" - https://arxiv.org/pdf/2310.06125.pdf

A baseline TCN model (from SpeechBrain) is also provided along with tools for estimating computational efficiency.

This recipe is a fork of the WHAMandWHAMR recipe in the SpeechBrain library (required, see below). For more help and information on any SpeechBrain related issues:
 * https://speechbrain.github.io/
 * https://github.com/speechbrain/speechbrain

# Data and models
Data:
 * WHAMR
 * WSJ0-2Mix
 * LibriMix [WIP]

Models:
 * Time-Domain Conformers (TD-Conformer)
 * Deformable Temporal Convolutional Networks (DTCN)
 * Utterance-Weighted Multi-Dilation Temporal Convolutional Network (WD-TCN)
 * Temporal Convolutional Networks (Conv-TasNet without skip connections)

# Running basic script
First install SRMRpy and remaining required packages
```
git clone https://github.com/jfsantos/SRMRpy.git
cd SRMRpy
python setup.py install

pip install -r requirements.txt
```
Then to run basic training of a DTCN model firstly change the ```data_folder``` hyperparameter in the ```separation/hparams/deformable/dtcn-whamr.yaml``` folder. Then run
```
cd separation
HPARAMS=hparams/deformable/dtcn-whamr.yaml
python train.py $HPARAMS
```
or if you wish to use multi GPU (recommended) run
```
python -m torch.distributed.launch --nproc_per_node=$NGPU train.py $HPARAMS --distributed_launch --distributed_backend='nccl' 

```
replacing ```NGPU``` with the desired number of GPUs to use.
In order to use dynamic mixing you will also need to change the ```base_folder_dm``` and ```rir_path``` hyperparameters, refer to https://github.com/speechbrain/speechbrain/blob/develop/recipes/WHAMandWHAMR/separation/README.md for more info on setting up dynamic mixing in SpeechBrain recipes.

# Known issues
 * The main issue at present is mixed precision training for DTCN with ```autocast``` enabled. We do not recommend trying to use this functionality at present.

# Paper
Please cite the following papers if you make use of the respective part of this codebase:
```
@INPROCEEDINGS{tdconformer23,
  author={Ravenscroft, William and Goetze, Stefan and Hain, Thomas},
  booktitle={2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)}, 
  title={On Time Domain Conformer Models for Monaural Speech Separation in Noisy Reverberant Acoustic Environments}, 
  year={2023},
  volume={},
  number={},
  pages={1-7},
  doi={10.1109/ASRU57964.2023.10389669}}

@INPROCEEDINGS{dtcn23,
  author={Ravenscroft, William and Goetze, Stefan and Hain, Thomas},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Deformable Temporal Convolutional Networks for Monaural Noisy Reverberant Speech Separation},
  month={June}
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10095230}}

@INPROCEEDINGS{wdtcn22,
  author={Ravenscroft, William and Goetze, Stefan and Hain, Thomas},
  booktitle={2022 International Workshop on Acoustic Signal Enhancement (IWAENC)}, 
  title={Utterance Weighted Multi-Dilation Temporal Convolutional Networks for Monaural Speech Dereverberation}, 
  year={2022},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/IWAENC53105.2022.9914752}}
```
