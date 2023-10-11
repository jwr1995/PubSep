# PubSep 

NOTE: This repository is still under construction and may contain major bugs

This repository provides training and evalution scripts for the DNN speech separation models described in various papers 
 * "Deformable Temporal Convolutional Networks for Monaural Noisy Reverberant Speech Separation" - https://arxiv.org/pdf/2210.15305.pdf.
 * "On Time-Domain Conformer Models for Noisy Reverberant Speech Separation" - https://arxiv.org/pdf/2310.06125.pdf

As baseline TCN schema is also provided along with tools for estimating computational efficiency.

This recipe is a fork of the WHAMandWHAMR recipe in the SpeechBrain library (required, see below). For more help and information on any SpeechBrain related issues:
 * https://speechbrain.github.io/
 * https://github.com/speechbrain/speechbrain

# Data and models
Data:
 * WHAMR
 * WSJ0-2Mix
 * LibriMix [WIP]

Models:
 * Time-Domain Conformers (TD-Conformer) [WIP]
 * Deformable Temporal Convolutional Networks (DTCN)
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
  booktitle={Workshop on Automatic Speech Recognition and Understanding 2023 (ASRU 2023)}, 
  title={On Time-Domain Conformers for Monaural Noisy Reverberant Speech Separation}, 
  month={Dec}
  year={2023}}

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
```
