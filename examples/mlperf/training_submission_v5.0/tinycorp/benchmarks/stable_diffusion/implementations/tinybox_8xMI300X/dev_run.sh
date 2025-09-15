#!/bin/bash
for i in {0..7}; do sudo rocm-smi -d $i --setperfdeterminism 1500; done
sudo rocm-smi -d 0 1 2 3 4 5 6 7 --setpoweroverdrive 750

# dependencies
#pip install tqdm
#pip install numpy
#pip install ftfy
#pip install regex
#pip install pillow
#pip install scipy
# webdataset depends on torch.utils.data.DataLoader
#pip install --index-url https://download.pytorch.org/whl/cpu torch
#pip install webdataset
source venv/bin/activate
pip list
apt list --installed | grep amdgpu

export BEAM=5 BEAM_UOPS_MAX=8000 BEAM_UPCAST_MAX=256 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=5 IGNORE_JIT_FIRST_BEAM=1 HCQDEV_WAIT_TIMEOUT_MS=300000
export AMD_LLVM=0 # bf16 seems to require this

export BASEDIR="$HOME/stable_diffusion"
export DATADIR="/raid/datasets/stable_diffusion"
export CKPTDIR="/raid/weights/stable_diffusion"
export MODEL="stable_diffusion" PYTHONPATH="."

# set these if resuming from checkpoint
export RESUME_CKPTDIR="/home/hooved/stable_diffusion/checkpoints/training_checkpoints/09150331"
export RESUME_ITR=6740
export GPUS=8 BS=304
export CONTEXT_BS=816 DENOISE_BS=600 DECODE_BS=384 INCEPTION_BS=560 CLIP_BS=240

# use separate BS for the jits in eval to maximize throughput
#export RUN_EVAL=1 EVAL_ONLY=1 CONTEXT_BS=816 DENOISE_BS=600 DECODE_BS=384 INCEPTION_BS=560 CLIP_BS=240

export WANDB=1
export PARALLEL=0

#export TOTAL_CKPTS=6

DATETIME=$(date "+%m%d%H%M")
#LOGFILE="sd_red_${DATETIME}_${SEED}.log"
export UNET_CKPTDIR="${BASEDIR}/checkpoints/training_checkpoints/${DATETIME}"
LEARNING_RATE="1.875e-7" RUNMLPERF=1 python3 examples/mlperf/model_train.py



#mkdir -p $UNET_CKPTDIR
#mkdir -p $UNET_CKPTDIR/run_eval
#LEARNING_RATE="1.875e-7" RUNMLPERF=1 python3 examples/mlperf/model_train.py && \
#ln -s "${UNET_CKPTDIR}/8425.safetensors" "${UNET_CKPTDIR}/run_eval/8425.safetensors" && \
#ln -s "${UNET_CKPTDIR}/10110.safetensors" "${UNET_CKPTDIR}/run_eval/10110.safetensors" && \
#sleep 120 && \
#EVAL_CKPT_DIR="${UNET_CKPTDIR}/run_eval" RUN_EVAL=1 EVAL_ONLY=1 RUNMLPERF=1 python3 examples/mlperf/model_train.py
