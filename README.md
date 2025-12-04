# TODO List
- [x] Inference code and model weights.
- [ ] Train code of SemVAE and diffusion model (SFD).

# Prepare Environments
```bash
conda create -n sfd python=3.10.12
conda activate sfd
pip install -r requirements.txt
pip install numpy==1.24.3 protobuf==3.20.0
## guided-diffusion evaluation environment
git clone https://github.com/openai/guided-diffusion.git
pip install tensorflow==2.8.0
sed -i 's/dtype=np\.bool)/dtype=np.bool_)/g' guided-diffusion/evaluations/evaluator.py  # or will encounter the error: "AttributeError: module 'numpy' has no attribute 'bool'".
```

# Prepare Model Weights
```bash
# Prepare the decoder of SD-VAE
mkdir -p outputs/model_weights/va-vae-imagenet256-experimental-variants
wget https://huggingface.co/hustvl/va-vae-imagenet256-experimental-variants/resolve/main/ldm-imagenet256-f16d32-50ep.ckpt \
    --no-check-certificate -O outputs/model_weights/va-vae-imagenet256-experimental-variants/ldm-imagenet256-f16d32-50ep.ckpt

# Prepare evaluation batches of ImageNet 256x256 from guided-diffusion
mkdir -p outputs/ADM_npz
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz -O outputs/ADM_npz/VIRTUAL_imagenet256_labeled.npz

# Download files from huggingface
mkdir temp
mkdir -p outputs/dataset/imagenet1k-latents
mkdir -p outputs/train
# Prepare latent statistics
huggingface-cli download SFD-Project/SFD --include "imagenet1k-latents/*" --local-dir temp
mv temp/imagenet1k-latents/* outputs/dataset/imagenet1k-latents/
# Prepare the autoguidance model
huggingface-cli download SFD-Project/SFD --include "model_weights/sfd_autoguidance_b/*" --local-dir temp
mv temp/model_weights/sfd_autoguidance_b outputs/train/
# Prepare XL model (675M)
huggingface-cli download SFD-Project/SFD --include "model_weights/sfd_xl/*" --local-dir temp
mv temp/model_weights/sfd_xl outputs/train/
# Prepare XXL model (1.0B)
huggingface-cli download SFD-Project/SFD --include "model_weights/sfd_1p0/*" --local-dir temp
mv temp/model_weights/sfd_1p0 outputs/train/
rm -rf temp
# or you can directly download the checkpoints from huggingface: https://huggingface.co/SFD-Project/SFD. Put the files in model_weights/ of SFD-Project/SFD to outputs/train
```

# Inference and Evaluation
## Inference demo
```bash
PRECISION=bf16 bash run_fast_inference.sh $INFERENCE_CONFIG
# take XL model (675M) as an example. 
CFG_SCALE="1.5" \
AUTOGUIDANCE_MODEL_SIZE="b" \
AUTOGUIDANCE_CKPT_ITER="70" \
PRECISION=bf16 bash run_fast_inference.sh configs/sfd/lightningdit_xl/inference_4m_autoguidance_demo.yaml
```

## Inference on ImageNet 256x256

For w/o Guidance, run the following command:
```bash
# w/o Guidance
FID_NUM=50000 \
GPUS_PER_NODE=$GPU_NUM PRECISION=bf16 bash run_inference.sh \
    $INFERENCE_CONFIG

# take XL model (675M) as an example. 
FID_NUM=50000 \
GPUS_PER_NODE=8 PRECISION=bf16 bash run_inference.sh \
    configs/sfd/lightningdit_xl/inference_4m.yaml
```
More inference configs can be found in `configs/sfd/lightningdit_xl` and `configs/sfd/lightningdit_1p0`, corresponding to XL (675M) and XXL (1.0B) models, respectively.

For w/ Guidance, run the following command:
```bash
# w/ Guidance
CFG_SCALE="$GUIDANCE_SCALE" \
AUTOGUIDANCE_MODEL_SIZE="b" \
AUTOGUIDANCE_CKPT_ITER="$GUIDANCE_ITER" \
FID_NUM=50000 \
GPUS_PER_NODE=$GPU_NUM PRECISION=bf16 bash run_inference.sh \
    $INFERENCE_CONFIG

# take XL model (675M) as an example. 
CFG_SCALE="1.5" \
AUTOGUIDANCE_MODEL_SIZE="b" \
AUTOGUIDANCE_CKPT_ITER="70" \
FID_NUM=50000 \
GPUS_PER_NODE=8 PRECISION=bf16 bash run_inference.sh \
    configs/sfd/lightningdit_xl/inference_4m_autoguidance.yaml
```
More inference configs can be found in `configs/sfd/lightningdit_xl` and `configs/sfd/lightningdit_1p0`, corresponding to XL (675M) and XXL (1.0B) models, respectively. For w/ Guidance, the detailed parameters for each configuration are shown in the following table:

| Model | Epochs | Params | Degraded Model | Iterations | Guidance Scale |
|-------|--------|--------|----------------|------------|----------------|
| LightningDiT-XL | 80 | 675M | LightningDiT-B | 70K | 1.6 |
| LightningDiT-XL | 800 | 675M | LightningDiT-B | 70K | 1.5 |
| LightningDiT-XXL | 80 | 1.0B | LightningDiT-B | 60K | 1.5 |
| LightningDiT-XXL | 800 | 1.0B | LightningDiT-B | 120K | 1.5 |


## Evaluation
```bash
# get final scores via guided-diffusion's evaluation tools
bash run_eval_via_guided_diffusion.sh $OUTPUT_IMAGES_DIR
# e.g.,
bash run_eval_via_guided_diffusion.sh outputs/train/sfd_xl/lightningdit-xl-1-ckpt-4000000-dopri5-250-balanced
```

Note that our models were trained and evaluated on NPUs (consistent with the results in the paper), but we observed minor performance differences when testing on GPUs. The specific results are shown below:

*w/o Guidance*
| Model | Epochs | #Params | FID (NPU) | FID (GPU) |
|-------|--------|---------|-----------|-----------|
| SFD (XL) | 80 | 675M | 3.43 | 3.50 |
| SFD (XL) | 800 | 675M | 2.54 | 2.66 |
| SFD (XXL) | 80 | 1.0B | 2.84 | 2.92 |
| SFD (XXL) | 800 | 1.0B | 2.38 | 2.36 |

*w/ Guidance*
| Model | Epochs | #Params | FID (NPU) | FID (GPU) |
|-------|--------|---------|-----------|-----------|
| SFD (XL) | 80 | 675M | 1.30 | 1.29 |
| SFD (XL) | 800 | 675M | 1.06 | 1.03 |
| SFD (XXL) | 80 | 1.0B | 1.19 | 1.20 |
| SFD (XXL) | 800 | 1.0B | 1.04 | 1.04 |
