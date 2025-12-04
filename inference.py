"""
Sampling Scripts of SFD.
"""

import os, math, json, pickle, logging, argparse, yaml, torch, numpy as np
from time import time, strftime
from glob import glob
from copy import deepcopy
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
import torch
import os
import torch.distributed as dist
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torchvision
# local imports
from tokenizer.vavae import VA_VAE
from models.lightningdit import LightningDiT_models
from transport import create_transport, Sampler
from dataset.img_latent_dataset import ImgLatentDataset
from omegaconf import OmegaConf

enable_swandb = False
if enable_swandb:
    # pip install swanlab
    import swanlab
    swanlab.login(api_key="YOUR_SWANLAB_KEY", save=True)

def drop_semantic_chansels(config, latent):
    if config['model'].get('semantic_chans', 0) > 0:
        return latent[:, :-config['model']['semantic_chans']]
    else:
        return latent

def build_autoguidance_config_path(model_size):
    """
    Build autoguidance config path based on model size.

    Args:
        model_size: 's' for small or 'b' for base

    Returns:
        config_path: path to autoguidance config file
    """
    # hard code for unconditional generation, not elegant.
    if model_size == 'uns':
        return 'configs/ours/_unconditional/inference_ags_lr1e-4_beta0p999.yaml'
    
    assert model_size == 'b', "Only base model size is supported now."
    return f'configs/sfd/autoguidance_{model_size}/inference.yaml'

def update_autoguidance_ckpt_path(config, ckpt_iter):
    """
    Update the checkpoint path in autoguidance config based on iteration number.

    Args:
        config: autoguidance config dict
        ckpt_iter: checkpoint iteration number (e.g., 100 for 100k)

    Returns:
        updated config with new ckpt_path
    """
    import re
    original_ckpt_path = config['ckpt_path']

    # Replace the checkpoint iteration number (e.g., 0400000.pt -> 0100000.pt)
    # Pattern: matches numbers before .pt extension
    new_ckpt_path = re.sub(r'/(\d+)\.pt$', f'/{ckpt_iter*1000:07d}.pt', original_ckpt_path)

    config['ckpt_path'] = new_ckpt_path

    return config

# sample function
def do_sample(train_config, accelerator, ckpt_path=None, cfg_scale=None, cfg_interval_start=None, timestep_shift=None,
              autoguidance_model_size=None, autoguidance_ckpt_iter=None, cfg_scale_sem=None, cfg_scale_tex=None,
              fid_num=None, num_sampling_steps=None, model=None, vae=None, demo_sample_mode=False):
    """
    Run sampling.
    """

    if cfg_scale is None:
        cfg_scale = train_config['sample']['cfg_scale']
    if cfg_interval_start is None:
        cfg_interval_start = train_config['sample']['cfg_interval_start'] if 'cfg_interval_start' in train_config['sample'] else 0
    if timestep_shift is None:
        timestep_shift = train_config['sample']['timestep_shift'] if 'timestep_shift' in train_config['sample'] else 0

    # Override fid_num if specified
    if fid_num is not None:
        train_config['sample']['fid_num'] = fid_num
        if accelerator.process_index == 0:
            print_with_prefix(f'Overriding fid_num from command line: {fid_num}')

    # Handle autoguidance parameters
    use_autoguidance = train_config['sample'].get('autoguidance', False)
    if use_autoguidance:
        if autoguidance_model_size is None:
            # Try to extract from config (backward compatibility)
            autoguidance_config_path = train_config['sample'].get('autoguidance_config', None)
            if autoguidance_config_path:
                # Default: use config file
                autoguidance_model_size = None
                autoguidance_ckpt_iter = None
            else:
                autoguidance_model_size = None
                autoguidance_ckpt_iter = None
        else:
            # Convert list to single value (take first element if list)
            autoguidance_model_size = autoguidance_model_size[0] if isinstance(autoguidance_model_size, list) else autoguidance_model_size
            autoguidance_ckpt_iter = autoguidance_ckpt_iter[0] if isinstance(autoguidance_ckpt_iter, list) else autoguidance_ckpt_iter
    else:
        autoguidance_model_size = None
        autoguidance_ckpt_iter = None

    # Handle separate cfg_scale for semantic and texture (AutoGuidance only)
    # Default to cfg_scale if not specified
    if use_autoguidance:
        if cfg_scale_sem is not None:
            cfg_scale_sem = cfg_scale_sem[0] if isinstance(cfg_scale_sem, list) else cfg_scale_sem

        if cfg_scale_tex is not None:
            cfg_scale_tex = cfg_scale_tex[0] if isinstance(cfg_scale_tex, list) else cfg_scale_tex
    else:
        cfg_scale_sem = None
        cfg_scale_tex = None

    # Handle num_sampling_steps parameter
    if num_sampling_steps is not None:
        num_sampling_steps = num_sampling_steps[0] if isinstance(num_sampling_steps, list) else num_sampling_steps

    # Convert list parameters to single values (take first element if list)
    cfg_scale = cfg_scale[0] if isinstance(cfg_scale, list) else cfg_scale
    cfg_interval_start = cfg_interval_start[0] if isinstance(cfg_interval_start, list) else cfg_interval_start
    timestep_shift = timestep_shift[0] if isinstance(timestep_shift, list) else timestep_shift

    # Override num_sampling_steps if specified
    if num_sampling_steps is not None:
        train_config['sample']['num_sampling_steps'] = num_sampling_steps

    # Determine effective cfg_scale_sem and cfg_scale_tex
    # If not specified, use cfg_scale
    effective_cfg_sem = cfg_scale_sem if cfg_scale_sem is not None else cfg_scale
    effective_cfg_tex = cfg_scale_tex if cfg_scale_tex is not None else cfg_scale

    # Assign simplified variable names for consistency with original code
    ag_model_size = autoguidance_model_size
    ag_ckpt_iter = autoguidance_ckpt_iter

    folder_name = f"{train_config['model']['model_type'].replace('/', '-')}-ckpt-{ckpt_path.split('/')[-1].split('.')[0]}-{train_config['sample']['sampling_method']}-{train_config['sample']['num_sampling_steps']}".lower()
    if cfg_scale > 1.0:
        folder_name += f"-interval{cfg_interval_start:.2f}"+f"-cfg{cfg_scale:.2f}"
        folder_name += f"-shift{timestep_shift:.2f}"
    if train_config['model'].get('semfirst_infer_interval_mode', None):
        folder_name += f"-cfgmode{train_config['model']['semfirst_infer_interval_mode']}"
    if train_config['sample'].get('balanced_sampling', False):
        folder_name += "-balanced"
    # Add autoguidance params to folder name
    if use_autoguidance and ag_model_size is not None and ag_ckpt_iter is not None:
        folder_name += f"-ag{ag_model_size}{ag_ckpt_iter}k"
    # Add separate sem/tex cfg scale to folder name (only if different from default)
    if use_autoguidance and (cfg_scale_sem is not None or cfg_scale_tex is not None):
        folder_name += f"-cfgsem{effective_cfg_sem:.2f}-cfgtex{effective_cfg_tex:.2f}"

    if demo_sample_mode:
        cfg_interval_start = 0
        timestep_shift = 0
        cfg_scale = 1.5
        effective_cfg_sem = effective_cfg_tex = cfg_scale
        sample_folder_dir = None  # No need for sample folder in demo mode
    else:
        sample_folder_dir = os.path.join(train_config['train']['output_dir'], train_config['train']['exp_name'], folder_name)
        if accelerator.process_index == 0:
            print_with_prefix('Sample_folder_dir=', sample_folder_dir)
            print_with_prefix('ckpt_path=', ckpt_path)
            print_with_prefix('cfg_scale=', cfg_scale)
            print_with_prefix('cfg_interval_start=', cfg_interval_start)
            print_with_prefix('timestep_shift=', timestep_shift)

        if not os.path.exists(sample_folder_dir):
            if accelerator.process_index == 0:
                os.makedirs(sample_folder_dir, exist_ok=True)
        else:
            png_files = [f for f in os.listdir(sample_folder_dir) if f.endswith('.png')]
            png_count = len(png_files)
            if png_count > train_config['sample']['fid_num']:
                if accelerator.process_index == 0:
                    print_with_prefix(f"Found {png_count} PNG files in {sample_folder_dir}, skip sampling.")
                return [sample_folder_dir]

    torch.backends.cuda.matmul.allow_tf32 = True  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup accelerator:
    device = accelerator.device

    # Setup DDP:
    device = accelerator.device
    seed = train_config['train']['global_seed'] * accelerator.num_processes + accelerator.process_index
    torch.manual_seed(seed)
    # torch.cuda.set_device(device)
    print_with_prefix(f"Starting rank={accelerator.local_process_index}, seed={seed}, world_size={accelerator.num_processes}.")
    rank = accelerator.local_process_index

    # Load model:
    if 'downsample_ratio' in train_config['vae']:
        downsample_ratio = train_config['vae']['downsample_ratio']
    else:
        downsample_ratio = 16
    latent_size = train_config['data']['image_size'] // downsample_ratio

    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    model.load_state_dict(checkpoint)
    model.eval()  # important!
    model.to(device)

    # Load autoguidance model if enabled
    autoguidance_model = None
    if use_autoguidance:
        # Determine autoguidance config path
        if ag_model_size is not None and ag_ckpt_iter is not None:
            # Use command-line parameters
            autoguidance_config_path = build_autoguidance_config_path(ag_model_size)
        else:
            # Use config file (backward compatibility)
            autoguidance_config_path = train_config['sample'].get('autoguidance_config', None)

        if autoguidance_config_path:
            if accelerator.process_index == 0:
                print_with_prefix(f'Loading autoguidance model from config: {autoguidance_config_path}')

            # Load autoguidance config
            autoguidance_config = load_config(autoguidance_config_path)

            # Update checkpoint path if iteration number is specified
            if ag_ckpt_iter is not None:
                autoguidance_config = update_autoguidance_ckpt_path(autoguidance_config, ag_ckpt_iter)
                if accelerator.process_index == 0:
                    print_with_prefix(f'Updated autoguidance checkpoint to {ag_ckpt_iter}k iteration')

            # Get autoguidance model checkpoint path
            autoguidance_ckpt_path = autoguidance_config['ckpt_path']

            # Create autoguidance model with same latent size
            autoguidance_model = LightningDiT_models[autoguidance_config['model']['model_type']](
                input_size=latent_size,
                class_dropout_prob=autoguidance_config['model']['class_dropout_prob'] if 'class_dropout_prob' in autoguidance_config['model'] else 0.1,
                num_classes=autoguidance_config['data']['num_classes'],
                use_qknorm=autoguidance_config['model']['use_qknorm'],
                use_swiglu=autoguidance_config['model']['use_swiglu'] if 'use_swiglu' in autoguidance_config['model'] else False,
                use_rope=autoguidance_config['model']['use_rope'] if 'use_rope' in autoguidance_config['model'] else False,
                use_rmsnorm=autoguidance_config['model']['use_rmsnorm'] if 'use_rmsnorm' in autoguidance_config['model'] else False,
                wo_shift=autoguidance_config['model']['wo_shift'] if 'wo_shift' in autoguidance_config['model'] else False,
                in_channels=autoguidance_config['model']['in_chans'] if 'in_chans' in autoguidance_config['model'] else 4,
                learn_sigma=autoguidance_config['model']['learn_sigma'] if 'learn_sigma' in autoguidance_config['model'] else False,
                use_repa=autoguidance_config['model']['use_repa'] if 'use_repa' in autoguidance_config['model'] else False,
                repa_dino_version=autoguidance_config['model']['repa_dino_version'] if 'repa_dino_version' in autoguidance_config['model'] else None,
                repa_depth=autoguidance_config['model']['repa_feat_depth'] if 'repa_feat_depth' in autoguidance_config['model'] else None,
                semantic_chans=autoguidance_config['model']['semantic_chans'] if 'semantic_chans' in autoguidance_config['model'] else 0,
                semfirst_delta_t=autoguidance_config['model']['semfirst_delta_t'] if 'semfirst_delta_t' in autoguidance_config['model'] else 0.0,
                semfirst_infer_interval_mode=autoguidance_config['model']['semfirst_infer_interval_mode'] if 'semfirst_infer_interval_mode' in autoguidance_config['model'] else 'both'
            )

            # Load autoguidance checkpoint
            autoguidance_checkpoint = torch.load(autoguidance_ckpt_path, map_location=lambda storage, loc: storage)
            if "ema" in autoguidance_checkpoint:
                autoguidance_checkpoint = autoguidance_checkpoint["ema"]
            autoguidance_model.load_state_dict(autoguidance_checkpoint)
            autoguidance_model.eval()
            autoguidance_model.to(device)

            if accelerator.process_index == 0:
                print_with_prefix(f'Loaded autoguidance model from: {autoguidance_ckpt_path}')
        else:
            if accelerator.process_index == 0:
                print_with_prefix('Warning: autoguidance enabled but autoguidance_config not specified')

    transport = create_transport(
        train_config['transport']['path_type'],
        train_config['transport']['prediction'],
        train_config['transport']['loss_weight'],
        train_config['transport']['train_eps'],
        train_config['transport']['sample_eps'],
        use_cosine_loss = train_config['transport']['use_cosine_loss'] if 'use_cosine_loss' in train_config['transport'] else False,
        use_lognorm = train_config['transport']['use_lognorm'] if 'use_lognorm' in train_config['transport'] else False,
        semantic_weight = train_config['model']['semantic_weight'] if 'semantic_weight' in train_config['model'] else 1.0,
        semantic_chans = train_config['model']['semantic_chans'] if 'semantic_chans' in train_config['model'] else 0,
        semfirst_delta_t = train_config['model']['semfirst_delta_t'] if 'semfirst_delta_t' in train_config['model'] else 0.0,
        repa_weight = train_config['model']['repa_weight'] if 'repa_weight' in train_config['model'] else 1.0,
        repa_mode = train_config['model']['repa_mode'] if 'repa_mode' in train_config['model'] else 'cos',
    )  # default: velocity;
    sampler = Sampler(transport)
    mode = train_config['sample']['mode']

    # Check if semantic first mode is enabled
    semantic_chans = train_config['model']['semantic_chans'] if 'semantic_chans' in train_config['model'] else 0
    semfirst_delta_t = train_config['model']['semfirst_delta_t'] if 'semfirst_delta_t' in train_config['model'] else 0.0
    use_semantic_first = semfirst_delta_t > 0 and semantic_chans > 0

    # Validate semantic first configuration
    if semfirst_delta_t > 0 and semantic_chans == 0:
        if accelerator.process_index == 0:
            print_with_prefix(f'Warning: semfirst_delta_t={semfirst_delta_t} but semantic_chans=0, semantic first disabled')
        use_semantic_first = False
    elif semfirst_delta_t == 0 and semantic_chans > 0:
        if accelerator.process_index == 0:
            print_with_prefix(f'Info: semantic_chans={semantic_chans} but semfirst_delta_t=0, using standard sampling')

    if use_semantic_first:
        model_in_channels = train_config['model']['in_chans']
        expected_texture_chans = model_in_channels - semantic_chans
        if accelerator.process_index == 0:
            print_with_prefix(f'Semantic First enabled: {expected_texture_chans} texture + {semantic_chans} semantic = {model_in_channels} total channels')

    if mode == "ODE":
        if use_semantic_first:
            # Use semantic first ODE sampling
            sample_fn = sampler.sample_ode_semantic_first(
                sampling_method=train_config['sample']['sampling_method'],
                num_steps=train_config['sample']['num_sampling_steps'],
                atol=train_config['sample']['atol'],
                rtol=train_config['sample']['rtol'],
                reverse=train_config['sample']['reverse'],
                timestep_shift=timestep_shift,
                semfirst_delta_t=semfirst_delta_t,
                semantic_chans=semantic_chans,
            )
            if accelerator.process_index == 0:
                print_with_prefix(f'Using Semantic First ODE sampling with delta_t={semfirst_delta_t}, semantic_chans={semantic_chans}')
                print_with_prefix(f'Semantic first phases: [0, {semfirst_delta_t:.1f}]=semantic-only, [{semfirst_delta_t:.1f}, 1.0]=both, [1.0, {1.0+semfirst_delta_t:.1f}]=texture-only')
        else:
            # Use standard ODE sampling
            sample_fn = sampler.sample_ode(
                sampling_method=train_config['sample']['sampling_method'],
                num_steps=train_config['sample']['num_sampling_steps'],
                atol=train_config['sample']['atol'],
                rtol=train_config['sample']['rtol'],
                reverse=train_config['sample']['reverse'],
                timestep_shift=timestep_shift,
            )
    else:
        raise NotImplementedError(f"Sampling mode {mode} is not supported.")

    if vae is None:
        vae = VA_VAE(
            f'tokenizer/configs/{train_config["vae"]["model_name"]}.yaml',
        )
        if accelerator.process_index == 0:
            print_with_prefix('Loaded VAE model')

    using_cfg = cfg_scale > 1.0
    if using_cfg:
        if accelerator.process_index == 0:
            print_with_prefix('Using cfg:', using_cfg)

    if not demo_sample_mode:
        if rank == 0:
            os.makedirs(sample_folder_dir, exist_ok=True)
            if accelerator.process_index == 0:
                print_with_prefix(f"Saving .png samples at {sample_folder_dir}")
        accelerator.wait_for_everyone()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = train_config['sample']['per_proc_batch_size']
    global_batch_size = n * accelerator.num_processes
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    if demo_sample_mode:
        num_samples = 0
    else:
        num_samples = len([name for name in os.listdir(sample_folder_dir) if (os.path.isfile(os.path.join(sample_folder_dir, name)) and ".png" in name)])
    total_samples = int(math.ceil(train_config['sample']['fid_num'] / global_batch_size) * global_batch_size)
    if rank == 0:
        if accelerator.process_index == 0 and not demo_sample_mode:
            print_with_prefix(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % accelerator.num_processes == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // accelerator.num_processes)
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    done_iterations = int( int(num_samples // accelerator.num_processes) // n)
    pbar = range(iterations)
    if not demo_sample_mode:
        pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0

    if accelerator.process_index == 0:
        print_with_prefix("Using latent normalization")
    dataset = ImgLatentDataset(
        data_dir=train_config['data']['data_path'],
        latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
        latent_sv_norm=train_config['data']['latent_sv_norm'] if 'latent_sv_norm' in train_config['data'] else False,
        latent_multiplier=train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215,
    )
    latent_mean, latent_std = dataset.get_latent_stats()
    latent_multiplier = train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215
    # move to device
    latent_mean = latent_mean.clone().detach().to(device)
    latent_std = latent_std.clone().detach().to(device)


    # Check if class_label is specified for fixed class generation
    fixed_class_label = train_config['sample'].get('class_label', None)
    if fixed_class_label is not None:
        assert 0 <= fixed_class_label <= 1000, "class_label must be in range [0, 1000]"
        if accelerator.process_index == 0:
            if fixed_class_label == 1000:
                print_with_prefix(f'Using fixed class_label={fixed_class_label} (unconditional generation)')
            else:
                print_with_prefix(f'Using fixed class_label={fixed_class_label} (conditional generation)')

    # Check if balanced sampling is enabled
    use_balanced_sampling = train_config['sample'].get('balanced_sampling', False)
    balanced_labels = None
    fid_num = train_config['sample']['fid_num']

    if use_balanced_sampling and not demo_sample_mode:
        # Generate balanced labels only for fid_num samples (not total_samples which may be larger)
        num_classes = train_config['data']['num_classes']
        samples_per_class = fid_num // num_classes

        # Create balanced label array for fid_num samples
        balanced_labels_list = []
        for class_id in range(num_classes):
            balanced_labels_list.extend([class_id] * samples_per_class)

        # If fid_num is not divisible by num_classes, add remaining samples
        remaining = fid_num - len(balanced_labels_list)
        if remaining > 0:
            balanced_labels_list.extend(list(range(remaining)))

        balanced_labels = torch.tensor(balanced_labels_list, device=device)

        if accelerator.process_index == 0:
            print_with_prefix(f'Using balanced sampling: {samples_per_class} samples per class, {remaining} remaining samples')
            print_with_prefix(f'Note: total_samples={total_samples}, fid_num={fid_num}, samples beyond fid_num will use random sampling')

            # for check, can be deleted
            print_with_prefix(balanced_labels[:200])

    if demo_sample_mode:
        if accelerator.process_index == 0:
            images = []
            for label in tqdm([106, 22, 279, 975, 388, 15, 36, 979], desc="Generating Demo Samples"):
                z = torch.randn(1, model.in_channels, latent_size, latent_size, device=device)
                y = torch.tensor([label], device=device)

                if autoguidance_model is not None:
                    # AutoGuidance mode: use small model for unconditional
                    model_kwargs = dict(y=y, cfg_scale=cfg_scale, cfg_interval=False, cfg_interval_start=cfg_interval_start,
                                      autoguidance_model=autoguidance_model,
                                      cfg_scale_sem=effective_cfg_sem, cfg_scale_tex=effective_cfg_tex)
                    model_fn = model.forward_with_autoguidance
                else:
                    # Standard CFG mode
                    z = torch.cat([z, z], 0)
                    y_null = torch.tensor([1000] * 1, device=device)
                    y = torch.cat([y, y_null], 0)
                    model_kwargs = dict(y=y, cfg_scale=cfg_scale, cfg_interval=False, cfg_interval_start=cfg_interval_start)
                    model_fn = model.forward_with_cfg

                samples = sample_fn(z, model_fn, **model_kwargs)[-1]
                samples = drop_semantic_chansels(train_config, samples)
                samples = (samples * latent_std) / latent_multiplier + latent_mean
                samples = vae.decode_to_images(samples)
                images.append(samples)
            # Combine 8 images into a 2x4 grid
            os.makedirs('demo_images', exist_ok=True)
            # Stack all images into a large numpy array
            all_images = np.stack([img[0] for img in images])  # Take first image from each batch
            # Rearrange into 2x4 grid
            h, w = all_images.shape[1:3]
            grid = np.zeros((2 * h, 4 * w, 3), dtype=np.uint8)
            for idx, image in enumerate(all_images):
                i, j = divmod(idx, 4)  # Calculate position in 2x4 grid
                grid[i*h:(i+1)*h, j*w:(j+1)*w] = image

            # Save the combined image
            Image.fromarray(grid).save('demo_images/demo_samples.png')
            print_with_prefix(f"Demo samples saved to demo_images/demo_samples.png")

            return None
    else:
        for iter_idx in pbar:
            # Sample inputs:
            z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)

            # Get labels (fixed, balanced, or random)
            if fixed_class_label is not None:
                # Use fixed class label for all samples
                y = torch.tensor([fixed_class_label] * n, device=device)
            elif use_balanced_sampling:
                # Use the same index calculation as saving images
                # to ensure label-image correspondence
                y_list = []
                for sample_idx in range(n):
                    global_index = sample_idx * accelerator.num_processes + accelerator.process_index + total

                    if global_index >= fid_num:
                        # Beyond fid_num, use random sampling
                        y_list.append(torch.randint(0, train_config['data']['num_classes'], (1,), device=device))
                    else:
                        # Within fid_num, use balanced sampling
                        y_list.append(balanced_labels[global_index:global_index+1])

                y = torch.cat(y_list, dim=0)
            else:
                y = torch.randint(0, train_config['data']['num_classes'], (n,), device=device)

            # Setup classifier-free guidance:
            if using_cfg:
                if autoguidance_model is not None:
                    # AutoGuidance mode: use small model for unconditional
                    model_kwargs = dict(y=y, cfg_scale=cfg_scale, cfg_interval=True, cfg_interval_start=cfg_interval_start,
                                      autoguidance_model=autoguidance_model,
                                      cfg_scale_sem=effective_cfg_sem, cfg_scale_tex=effective_cfg_tex)
                    model_fn = model.forward_with_autoguidance
                else:
                    # Standard CFG mode
                    z = torch.cat([z, z], 0)
                    y_null = torch.tensor([1000] * n, device=device)
                    y = torch.cat([y, y_null], 0)
                    model_kwargs = dict(y=y, cfg_scale=cfg_scale, cfg_interval=True, cfg_interval_start=cfg_interval_start)
                    model_fn = model.forward_with_cfg
            else:
                model_kwargs = dict(y=y)
                model_fn = model.forward

            samples = sample_fn(z, model_fn, **model_kwargs)[-1]
            if using_cfg and autoguidance_model is None:
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples (only for standard CFG)
            samples = drop_semantic_chansels(train_config, samples)
            samples = (samples * latent_std) / latent_multiplier + latent_mean
            samples = vae.decode_to_images(samples)

            # Log progress to swanlab
            if accelerator.process_index == 0 and enable_swandb:
                print_with_prefix(f"current_sample_iter / iterations: {iter_idx} / {iterations}")
            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i * accelerator.num_processes + accelerator.process_index + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
            total += global_batch_size
            accelerator.wait_for_everyone()

    return [sample_folder_dir]

# some utils
def print_with_prefix(*messages):
    prefix = f"\033[34m[LightningDiT-Sampling {strftime('%Y-%m-%d %H:%M:%S')}]\033[0m"
    combined_message = ' '.join(map(str, messages))
    print(f"{prefix}: {combined_message}")

# def load_config(config_path):
#     with open(config_path, "r") as file:
#         config = yaml.safe_load(file)
#     return config
def load_config(config_path):
    config = OmegaConf.load(config_path)
    return config

if __name__ == "__main__":

    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/lightningdit_b_ldmvae_f16d16.yaml')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--calculate-fid', action='store_true', default=False, help='Calculate FID after inference (default: False)')
    parser.add_argument('--cfg_scale', nargs='*', type=float, default=None)
    parser.add_argument('--cfg_interval_start', nargs='*', type=float, default=None)
    parser.add_argument('--timestep_shift', nargs='*', type=float, default=None)
    parser.add_argument('--autoguidance_model_size', nargs='*', type=str, default=None, help='AutoGuidance model size: "s" or "b" (default: from config)')
    parser.add_argument('--autoguidance_ckpt_iter', nargs='*', type=int, default=None, help='AutoGuidance checkpoint iteration in k (e.g., 100 for 100k, default: from config)')
    parser.add_argument('--cfg_scale_sem', nargs='*', type=float, default=None, help='CFG scale for semantic (AutoGuidance only, default: use cfg_scale)')
    parser.add_argument('--cfg_scale_tex', nargs='*', type=float, default=None, help='CFG scale for texture (AutoGuidance only, default: use cfg_scale)')
    parser.add_argument('--fid_num', type=int, default=None, help='Number of samples to generate (default: from config)')
    parser.add_argument('--num_sampling_steps', nargs='*', type=int, default=None, help='Number of sampling steps (default: from config)')
    args = parser.parse_args()
    accelerator = Accelerator()
    train_config = load_config(args.config)

    # Initialize swanlab
    if accelerator.is_main_process and enable_swandb:
        swanlab.init(
            project="LightningDiT-Inference",
            experiment_name=train_config['train']['exp_name'],
            config=train_config
        )

    # get ckpt_dir
    assert 'ckpt_path' in train_config, "ckpt_path must be specified in config"
    if accelerator.process_index == 0:
        print_with_prefix('Using ckpt:', train_config['ckpt_path'])
    ckpt_dir = train_config['ckpt_path']

    if 'downsample_ratio' in train_config['vae']:
        latent_size = train_config['data']['image_size'] // train_config['vae']['downsample_ratio']
    else:
        latent_size = train_config['data']['image_size'] // 16

    # get model
    model = LightningDiT_models[train_config['model']['model_type']](
        input_size=latent_size,
        class_dropout_prob=train_config['model']['class_dropout_prob'] if 'class_dropout_prob' in train_config['model'] else 0.1,
        num_classes=train_config['data']['num_classes'],
        use_qknorm=train_config['model']['use_qknorm'],
        use_swiglu=train_config['model']['use_swiglu'] if 'use_swiglu' in train_config['model'] else False,
        use_rope=train_config['model']['use_rope'] if 'use_rope' in train_config['model'] else False,
        use_rmsnorm=train_config['model']['use_rmsnorm'] if 'use_rmsnorm' in train_config['model'] else False,
        wo_shift=train_config['model']['wo_shift'] if 'wo_shift' in train_config['model'] else False,
        in_channels=train_config['model']['in_chans'] if 'in_chans' in train_config['model'] else 4,
        learn_sigma=train_config['model']['learn_sigma'] if 'learn_sigma' in train_config['model'] else False,
        use_repa=train_config['model']['use_repa'] if 'use_repa' in train_config['model'] else False,
        repa_dino_version=train_config['model']['repa_dino_version'] if 'repa_dino_version' in train_config['model'] else None,
        repa_depth=train_config['model']['repa_feat_depth'] if 'repa_feat_depth' in train_config['model'] else None,
        semantic_chans=train_config['model']['semantic_chans'] if 'semantic_chans' in train_config['model'] else 0,
        semfirst_delta_t=train_config['model']['semfirst_delta_t'] if 'semfirst_delta_t' in train_config['model'] else 0.0,
        semfirst_infer_interval_mode=train_config['model']['semfirst_infer_interval_mode'] if 'semfirst_infer_interval_mode' in train_config['model'] else 'both'
    )

    # naive sample
    sample_folder_dirs = do_sample(
        train_config,
        accelerator,
        ckpt_path=ckpt_dir,
        cfg_scale=args.cfg_scale,
        cfg_interval_start=args.cfg_interval_start,
        timestep_shift=args.timestep_shift,
        autoguidance_model_size=args.autoguidance_model_size,
        autoguidance_ckpt_iter=args.autoguidance_ckpt_iter,
        cfg_scale_sem=args.cfg_scale_sem,
        cfg_scale_tex=args.cfg_scale_tex,
        fid_num=args.fid_num,
        num_sampling_steps=args.num_sampling_steps,
        model=model,
        demo_sample_mode=args.demo)

    if not args.demo and args.calculate_fid:
        # calculate FID
        # Important: FID is only for reference, please use ADM evaluation for paper reporting
        if accelerator.process_index == 0:
            for sample_folder_dir in sample_folder_dirs:
                print_with_prefix(f'Calculating FID for {sample_folder_dir}')
                from tools.calculate_fid import calculate_fid_given_paths
                print_with_prefix('Calculating FID with {} number of samples'.format(train_config['sample']['fid_num']))
                assert 'fid_reference_file' in train_config['data'], "fid_reference_file must be specified in config"
                fid_reference_file = train_config['data']['fid_reference_file']
                fid = calculate_fid_given_paths(
                    [fid_reference_file, sample_folder_dir],
                    batch_size=50,
                    dims=2048,
                    device='cuda',
                    num_workers=8,
                    sp_len = train_config['sample']['fid_num']
                )
                print_with_prefix('fid=',fid)
                # Log FID to swanlab
                if enable_swandb:
                    swanlab.log({'FID': fid})
        accelerator.wait_for_everyone()
