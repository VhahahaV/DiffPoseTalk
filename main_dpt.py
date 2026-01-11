import argparse
import math
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import warnings
# Fix numpy compatibility for chumpy (compatible with both numpy 1.x and 2.x)
# numpy 2.0 removed unicode_ and some other aliases
# Suppress FutureWarnings during compatibility setup
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    try:
        if not hasattr(np, 'bool'):
            np.bool = np.bool_
        if not hasattr(np, 'int'):
            np.int = np.int_
        if not hasattr(np, 'float'):
            np.float = np.float64 if hasattr(np, 'float64') else getattr(np, 'float_', np.float64)
        if not hasattr(np, 'complex'):
            np.complex = np.complex128 if hasattr(np, 'complex128') else getattr(np, 'complex_', np.complex128)
        if not hasattr(np, 'object'):
            np.object = np.object_
        # numpy 2.0: unicode_ was removed, use str_ instead
        if not hasattr(np, 'unicode'):
            # Check if unicode_ exists (numpy 1.x) or use str_ (numpy 2.x)
            if hasattr(np, 'unicode_'):
                np.unicode = np.unicode_
            else:
                np.unicode = np.str_  # numpy 2.x fallback
        if not hasattr(np, 'str'):
            np.str = np.str_
    except (AttributeError, TypeError):
        # numpy 2.x - use alternative approach
        try:
            import numpy._core.multiarray as _multiarray
            if not hasattr(_multiarray, 'bool'):
                _multiarray.bool = np.bool_
            if not hasattr(_multiarray, 'int'):
                _multiarray.int = np.int_
            if not hasattr(_multiarray, 'float'):
                _multiarray.float = np.float64
            if not hasattr(_multiarray, 'complex'):
                _multiarray.complex = np.complex128
            if not hasattr(_multiarray, 'object'):
                _multiarray.object = np.object_
            if not hasattr(_multiarray, 'unicode'):
                # Check if unicode_ exists (numpy 1.x) or use str_ (numpy 2.x)
                if hasattr(np, 'unicode_'):
                    _multiarray.unicode = np.unicode_
                else:
                    _multiarray.unicode = np.str_  # numpy 2.x fallback
            if not hasattr(_multiarray, 'str'):
                _multiarray.str = np.str_
        except (ImportError, AttributeError):
            # If all else fails, just suppress the error - chumpy might work anyway
            pass

import torch
import torch.nn.functional as F
import torch.optim as optim
from colorama import Fore, Back, Style
from tensorboardX import SummaryWriter
from torch.utils import data
from tqdm import tqdm

import options.dpt as options
import utils
from data import MotionInferenceDataset, MotionJsonDataset, infinite_data_loader
from datasets import build_style_id
from models import DiffTalkingHead, StyleEncoder
from models.flame import FLAME, FLAMEConfig


def train(args, model: DiffTalkingHead, style_enc: Optional[StyleEncoder], train_loader, val_loader, optimizer,
          save_dir, scheduler=None, writer=None, flame=None):
    device = model.device
    save_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    data_loader = infinite_data_loader(train_loader)

    coef_stats = train_loader.dataset.coef_stats
    if coef_stats is not None:
        coef_stats = {x: coef_stats[x].to(device) for x in coef_stats}
    audio_unit = train_loader.dataset.audio_unit
    predict_head_pose = not args.no_head_pose

    loss_log = defaultdict(lambda: deque(maxlen=args.log_smooth_win))
    pbar = tqdm(range(args.max_iter + 1), dynamic_ncols=True)
    optimizer.zero_grad()
    for it in pbar:
        # Load data
        audio_pair, coef_pair, audio_stats = next(data_loader)
        audio_pair = [audio.to(device) for audio in audio_pair]
        coef_pair = [{x: coef_pair[i][x].to(device) for x in coef_pair[i]} for i in range(2)]
        motion_coef_pair = [
            utils.get_motion_coef(coef_pair[i], args.rot_repr, predict_head_pose) for i in range(2)
        ]  # (N, L, 51) when no_head_pose=True

        # Verify motion_coef dimensions match style encoder expectations
        if style_enc is not None:
            expected_dim = style_enc.motion_coef_dim
            actual_dim = motion_coef_pair[0].shape[-1]
            if actual_dim != expected_dim:
                raise ValueError(f'Motion coefficient dimension mismatch: expected {expected_dim} (from style encoder), got {actual_dim}')

        # Use the shape coefficients from the first frame of the first clip as the condition
        if coef_pair[0]['shape'].ndim == 2:  # (N, 100)
            shape_coef = coef_pair[0]['shape'].clone().to(device)
        else:  # (N, L, 100)
            shape_coef = coef_pair[0]['shape'][:, 0].clone().to(device)

        # Extract style features
        if style_enc is not None:
            with torch.no_grad():
                style_pair = [style_enc(motion_coef_pair[i]) for i in range(2)]
                # Verify style feature dimensions
                if style_pair[0].shape[-1] != args.d_style:
                    raise ValueError(f'Style feature dimension mismatch: expected {args.d_style}, got {style_pair[0].shape[-1]}')

        if args.use_context_audio_feat:
            # Extract audio features
            audio_feat = model.extract_audio_feature(torch.cat(audio_pair, dim=1), args.n_motions * 2)  # (N, 2L, :)

        loss_noise = 0
        loss_vert = 0
        loss_vel = torch.tensor(0, device=device)
        loss_smooth = torch.tensor(0, device=device)
        loss_head_angle = 0
        loss_head_vel = torch.tensor(0, device=device)
        loss_head_smooth = torch.tensor(0, device=device)
        loss_head_trans = 0
        for i in range(2):
            audio = audio_pair[i]  # (N, L_a)
            motion_coef = motion_coef_pair[i]  # (N, L, 51) when no_head_pose=True
            style = style_pair[1 - i] if style_enc is not None else None
            batch_size = audio.shape[0]

            # truncate input audio and motion according to trunc_prob
            if (i == 0 and np.random.rand() < args.trunc_prob1) or (i != 0 and np.random.rand() < args.trunc_prob2):
                audio_in, motion_coef_in, end_idx = utils.truncate_motion_coef_and_audio(
                    audio, motion_coef, args.n_motions, audio_unit, args.pad_mode)
                if args.use_context_audio_feat and i != 0:
                    # use contextualized audio feature for the second clip
                    audio_in = model.extract_audio_feature(torch.cat([audio_pair[i - 1], audio_in], dim=1),
                                                           args.n_motions * 2)[:, -args.n_motions:]
            else:
                if args.use_context_audio_feat:
                    audio_in = audio_feat[:, i * args.n_motions:(i + 1) * args.n_motions]
                else:
                    audio_in = audio
                motion_coef_in, end_idx = motion_coef, None

            if args.use_indicator:
                if end_idx is not None:
                    indicator = torch.arange(args.n_motions, device=device).expand(batch_size, -1) < end_idx.unsqueeze(
                        1)
                else:
                    indicator = torch.ones(batch_size, args.n_motions, device=device)
            else:
                indicator = None

            # Inference
            if i == 0:
                noise, target, prev_motion_coef, prev_audio_feat = model(
                    motion_coef_in, audio_in, shape_coef, style, indicator=indicator)
                if end_idx is not None:  # was truncated, needs to use the complete feature
                    prev_motion_coef = motion_coef[:, -args.n_prev_motions:]
                    if args.use_context_audio_feat:
                        prev_audio_feat = audio_feat[:, args.n_motions - args.n_prev_motions:args.n_motions].detach()
                    else:
                        with torch.no_grad():
                            prev_audio_feat = model.extract_audio_feature(audio)[:, -args.n_prev_motions:]
                else:
                    prev_motion_coef = prev_motion_coef[:, -args.n_prev_motions:]
                    prev_audio_feat = prev_audio_feat[:, -args.n_prev_motions:]
            else:
                noise, target, _, _ = model(motion_coef_in, audio_in, shape_coef, style,
                                            prev_motion_coef, prev_audio_feat, indicator=indicator)

            loss_n, loss_v, loss_c, loss_s, loss_ha, loss_hc, loss_hs, loss_ht = utils.compute_loss(
                args, i == 0, shape_coef, motion_coef_in, noise, target, prev_motion_coef, coef_stats, flame, end_idx)
            loss_noise = loss_noise + loss_n / 2
            if args.target == 'sample' and args.l_vert > 0:
                loss_vert = loss_vert + loss_v / 2
            if args.target == 'sample' and args.l_vel > 0 and loss_c is not None:
                loss_vel = loss_vel + loss_c / 2
            if args.target == 'sample' and args.l_smooth > 0 and loss_s is not None:
                loss_smooth = loss_smooth + loss_s / 2
            if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
                loss_head_angle = loss_head_angle + loss_ha / 2
            if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0 and loss_hc is not None:
                loss_head_vel = loss_head_vel + loss_hc / 2
            if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0 and loss_hs is not None:
                loss_head_smooth = loss_head_smooth + loss_hs / 2
            if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0 and loss_ht is not None:
                # no need to divide by 2 because it only applies to the second clip
                loss_head_trans = loss_head_trans + loss_ht

        loss_log['noise'].append(loss_noise.item())
        loss = loss_noise
        if args.target == 'sample' and args.l_vert > 0:
            loss_log['vert'].append(loss_vert.item())
            loss = loss + args.l_vert * loss_vert
        if args.target == 'sample' and args.l_vel > 0:
            loss_log['vel'].append(loss_vel.item())
            loss = loss + args.l_vel * loss_vel
        if args.target == 'sample' and args.l_smooth > 0:
            loss_log['smooth'].append(loss_smooth.item())
            loss = loss + args.l_smooth * loss_smooth
        if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
            loss_log['head_angle'].append(loss_head_angle.item())
            loss = loss + args.l_head_angle * loss_head_angle
        if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0:
            loss_log['head_vel'].append(loss_head_vel.item())
            loss = loss + args.l_head_vel * loss_head_vel
        if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0:
            loss_log['head_smooth'].append(loss_head_smooth.item())
            loss = loss + args.l_head_smooth * loss_head_smooth
        if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0:
            loss_log['head_trans'].append(loss_head_trans.item())
            loss = loss + args.l_head_trans * loss_head_trans
        loss.backward()

        if it % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Logging
        loss_log['loss'].append(loss.item())
        description = f'Train loss: [N: {np.mean(loss_log["noise"]):.3e}'
        if args.target == 'sample' and args.l_vert > 0:
            description += f', V: {np.mean(loss_log["vert"]):.3e}'
        if args.target == 'sample' and args.l_vel > 0:
            description += f', C: {np.mean(loss_log["vel"]):.3e}'
        if args.target == 'sample' and args.l_smooth > 0:
            description += f', S: {np.mean(loss_log["smooth"]):.3e}'
        if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
            description += f', HA: {np.mean(loss_log["head_angle"]):.3e}'
        if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0:
            description += f', HC: {np.mean(loss_log["head_vel"]):.3e}'
        if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0:
            description += f', HS: {np.mean(loss_log["head_smooth"]):.3e}'
        if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0:
            description += f', HT: {np.mean(loss_log["head_trans"]):.3e}'
        description += ']'
        pbar.set_description(description)

        if it % args.log_iter == 0 and writer is not None:
            # write to tensorboard
            writer.add_scalar('train/loss', np.mean(loss_log['loss']), it)
            writer.add_scalar('train/noise', np.mean(loss_log['noise']), it)
            if args.target == 'sample' and args.l_vert > 0:
                writer.add_scalar('train/vert', np.mean(loss_log['vert']), it)
            if args.target == 'sample' and args.l_vel > 0:
                writer.add_scalar('train/vel', np.mean(loss_log['vel']), it)
            if args.target == 'sample' and args.l_smooth > 0:
                writer.add_scalar('train/smooth', np.mean(loss_log['smooth']), it)
            if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
                writer.add_scalar('train/head_angle', np.mean(loss_log['head_angle']), it)
            if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0:
                writer.add_scalar('train/head_vel', np.mean(loss_log['head_vel']), it)
            if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0:
                writer.add_scalar('train/head_smooth', np.mean(loss_log['head_smooth']), it)
            if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0:
                writer.add_scalar('train/head_trans', np.mean(loss_log['head_trans']), it)
            writer.add_scalar('opt/lr', optimizer.param_groups[0]['lr'], it)

        # update learning rate
        if scheduler is not None:
            if args.scheduler != 'WarmupThenDecay' or (args.scheduler == 'WarmupThenDecay' and it < args.cos_max_iter):
                scheduler.step()

        # save model
        if (it % args.save_iter == 0 and it != 0) or it == args.max_iter:
            torch.save({
                'args': args,
                'model': model.state_dict(),
                'iter': it,
            }, save_dir / f'iter_{it:07}.pt')

        # validation
        if (it % args.val_iter == 0 and it != 0) or it == args.max_iter:
            test(args, model, style_enc, val_loader, it, 10, 'val', writer, flame)


@torch.no_grad()
def test(args, model: DiffTalkingHead, style_enc: Optional[StyleEncoder], test_loader, current_iter, n_rounds=10,
         mode='val', writer=None, flame=None):
    is_training = model.training
    device = model.device
    model.eval()

    coef_stats = test_loader.dataset.coef_stats
    if coef_stats is not None:
        coef_stats = {x: coef_stats[x].to(device) for x in coef_stats}
    audio_unit = test_loader.dataset.audio_unit
    predict_head_pose = not args.no_head_pose

    loss_log = defaultdict(list)
    for test_round in range(n_rounds):
        for audio_pair, coef_pair, audio_stats in test_loader:
            audio_pair = [audio.to(device) for audio in audio_pair]
            coef_pair = [{x: coef_pair[i][x].to(device) for x in coef_pair[i]} for i in range(2)]
            motion_coef_pair = [
                utils.get_motion_coef(coef_pair[i], args.rot_repr, predict_head_pose) for i in range(2)
            ]  # (N, L, 51) when no_head_pose=True

            # Verify motion_coef dimensions match style encoder expectations
            if style_enc is not None:
                expected_dim = style_enc.motion_coef_dim
                actual_dim = motion_coef_pair[0].shape[-1]
                if actual_dim != expected_dim:
                    raise ValueError(f'Motion coefficient dimension mismatch: expected {expected_dim} (from style encoder), got {actual_dim}')

            # Use the shape coefficients from the first frame of the first clip as the condition
            if coef_pair[0]['shape'].ndim == 2:  # (N, 100)
                shape_coef = coef_pair[0]['shape'].clone().to(device)
            else:  # (N, L, 100)
                shape_coef = coef_pair[0]['shape'][:, 0].clone().to(device)

            # Extract style features
            if style_enc is not None:
                with torch.no_grad():
                    style_pair = [style_enc(motion_coef_pair[i]) for i in range(2)]
                    # Verify style feature dimensions
                    if style_pair[0].shape[-1] != args.d_style:
                        raise ValueError(f'Style feature dimension mismatch: expected {args.d_style}, got {style_pair[0].shape[-1]}')

            if args.use_context_audio_feat:
                # Extract audio features
                audio_feat = model.extract_audio_feature(torch.cat(audio_pair, dim=1), args.n_motions * 2)  # (N, 2L, :)

            loss_noise = 0
            loss_vert = 0
            loss_vel = torch.tensor(0, device=device)
            loss_smooth = torch.tensor(0, device=device)
            loss_head_angle = 0
            loss_head_vel = torch.tensor(0, device=device)
            loss_head_smooth = torch.tensor(0, device=device)
            loss_head_trans = 0
            for i in range(2):
                audio = audio_pair[i]  # (N, L_a)
                motion_coef = motion_coef_pair[i]  # (N, L, 51) when no_head_pose=True
                style = style_pair[1 - i] if style_enc is not None else None
                batch_size = audio.shape[0]

                # truncate input audio and motion according to trunc_prob
                if (i == 0 and np.random.rand() < args.trunc_prob1) or (i != 0 and np.random.rand() < args.trunc_prob2):
                    audio_in, motion_coef_in, end_idx = utils.truncate_motion_coef_and_audio(
                        audio, motion_coef, args.n_motions, audio_unit, args.pad_mode)
                    if args.use_context_audio_feat and i != 0:
                        # use contextualized audio feature for the second clip
                        audio_in = model.extract_audio_feature(torch.cat([audio_pair[i - 1], audio_in], dim=1),
                                                               args.n_motions * 2)[:, -args.n_motions:]

                else:
                    if args.use_context_audio_feat:
                        audio_in = audio_feat[:, i * args.n_motions:(i + 1) * args.n_motions]
                    else:
                        audio_in = audio
                    motion_coef_in, end_idx = motion_coef, None

                if args.use_indicator:
                    if end_idx is not None:
                        indicator = torch.arange(args.n_motions, device=device).expand(batch_size,
                                                                                       -1) < end_idx.unsqueeze(1)
                    else:
                        indicator = torch.ones(batch_size, args.n_motions, device=device)
                else:
                    indicator = None

                # Inference
                if i == 0:
                    noise, target, prev_motion_coef, prev_audio_feat = model(
                        motion_coef_in, audio_in, shape_coef, style, indicator=indicator)
                    if end_idx is not None:  # was truncated, needs to use the complete feature
                        prev_motion_coef = motion_coef[:, -args.n_prev_motions:]
                        if args.use_context_audio_feat:
                            prev_audio_feat = audio_feat[:, args.n_motions - args.n_prev_motions:args.n_motions]
                        else:
                            with torch.no_grad():
                                prev_audio_feat = model.extract_audio_feature(audio)[:, -args.n_prev_motions:]
                    else:
                        prev_motion_coef = prev_motion_coef[:, -args.n_prev_motions:]
                        prev_audio_feat = prev_audio_feat[:, -args.n_prev_motions:]
                else:
                    noise, target, _, _ = model(motion_coef_in, audio, shape_coef, style,
                                                prev_motion_coef, prev_audio_feat, indicator=indicator)

                loss_n, loss_v, loss_c, loss_s, loss_ha, loss_hc, loss_hs, loss_ht = utils.compute_loss(
                    args, i == 0, shape_coef, motion_coef_in, noise, target, prev_motion_coef, coef_stats, flame,
                    end_idx
                )
                loss_noise = loss_noise + loss_n / 2
                if args.target == 'sample' and args.l_vert > 0:
                    loss_vert = loss_vert + loss_v / 2
                if args.target == 'sample' and args.l_vel > 0 and loss_c is not None:
                    loss_vel = loss_vel + loss_c / 2
                if args.target == 'sample' and args.l_smooth > 0 and loss_s is not None:
                    loss_smooth = loss_smooth + loss_s / 2
                if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
                    loss_head_angle = loss_head_angle + loss_ha / 2
                if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0 and loss_hc is not None:
                    loss_head_vel = loss_head_vel + loss_hc / 2
                if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0 and loss_hs is not None:
                    loss_head_smooth = loss_head_smooth + loss_hs / 2
                if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0 and loss_ht is not None:
                    # no need to divide by 2 because it only applies to the second clip
                    loss_head_trans = loss_head_trans + loss_ht

            loss_log['noise'].append(loss_noise.item())
            loss = loss_noise
            if args.target == 'sample' and args.l_vert > 0:
                loss_log['vert'].append(loss_vert.item())
                loss = loss + args.l_vert * loss_vert
            if args.target == 'sample' and args.l_vel > 0:
                loss_log['vel'].append(loss_vel.item())
                loss = loss + args.l_vel * loss_vel
            if args.target == 'sample' and args.l_smooth > 0:
                loss_log['smooth'].append(loss_smooth.item())
                loss = loss + args.l_smooth * loss_smooth
            if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
                loss_log['head_angle'].append(loss_head_angle.item())
                loss = loss + args.l_head_angle * loss_head_angle
            if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0:
                loss_log['head_vel'].append(loss_head_vel.item())
                loss = loss + args.l_head_vel * loss_head_vel
            if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0:
                loss_log['head_smooth'].append(loss_head_smooth.item())
                loss = loss + args.l_head_smooth * loss_head_smooth
            if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0:
                loss_log['head_trans'].append(loss_head_trans.item())
                loss = loss + args.l_head_trans * loss_head_trans
            loss_log['loss'].append(loss.item())

    description = f'(Iter {current_iter:>6}) {mode} loss: [N: {np.mean(loss_log["noise"]):.3e}'
    if args.target == 'sample' and args.l_vert > 0:
        description += f', V: {np.mean(loss_log["vert"]):.3e}'
    if args.target == 'sample' and args.l_vel > 0:
        description += f', C: {np.mean(loss_log["vel"]):.3e}'
    if args.target == 'sample' and args.l_smooth > 0:
        description += f', S: {np.mean(loss_log["smooth"]):.3e}'
    if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
        description += f', HA: {np.mean(loss_log["head_angle"]):.3e}'
    if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0:
        description += f', HC: {np.mean(loss_log["head_vel"]):.3e}'
    if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0:
        description += f', HS: {np.mean(loss_log["head_smooth"]):.3e}'
    if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0:
        description += f', HT: {np.mean(loss_log["head_trans"]):.3e}'
    description += ']'
    print(description)

    if writer is not None:
        # write to tensorboard
        writer.add_scalar(f'{mode}/loss', np.mean(loss_log['loss']), current_iter)
        writer.add_scalar(f'{mode}/noise', np.mean(loss_log['noise']), current_iter)
        if args.target == 'sample' and args.l_vert > 0:
            writer.add_scalar(f'{mode}/vert', np.mean(loss_log['vert']), current_iter)
        if args.target == 'sample' and args.l_vel > 0:
            writer.add_scalar(f'{mode}/vel', np.mean(loss_log['vel']), current_iter)
        if args.target == 'sample' and args.l_smooth > 0:
            writer.add_scalar(f'{mode}/smooth', np.mean(loss_log['smooth']), current_iter)
        if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
            writer.add_scalar(f'{mode}/head_angle', np.mean(loss_log['head_angle']), current_iter)
        if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0:
            writer.add_scalar(f'{mode}/head_vel', np.mean(loss_log['head_vel']), current_iter)
        if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0:
            writer.add_scalar(f'{mode}/head_smooth', np.mean(loss_log['head_smooth']), current_iter)
        if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0:
            writer.add_scalar(f'{mode}/head_trans', np.mean(loss_log['head_trans']), current_iter)

    if is_training:
        model.train()


@torch.no_grad()
def generate_motion_sequence(args, model: DiffTalkingHead, sample: dict, coef_stats=None, style_enc=None):
    device = model.device
    batch_size = args.n_repetitions

    audio = sample['audio'].to(device)
    shape_vec = sample['shape'].to(device)
    shape_batch = shape_vec.unsqueeze(0).expand(batch_size, -1)

    total_frames = int(sample['seq_len'])
    stride = args.n_motions
    audio_unit = 16000. / args.fps
    n_audio_samples = round(audio_unit * stride)

    motion_chunks = []
    prev_motion = None
    for start in range(0, total_frames, stride):
        end = min(start + stride, total_frames)
        frame_len = end - start
        start_sample = round(start * audio_unit)
        end_sample = start_sample + n_audio_samples
        audio_chunk = audio[start_sample:end_sample]
        if audio_chunk.numel() < n_audio_samples:
            if args.pad_mode == 'replicate' and audio_chunk.numel() > 0:
                pad_value = audio_chunk[-1]
            else:
                pad_value = 0
            audio_chunk = F.pad(audio_chunk, (0, n_audio_samples - audio_chunk.numel()), value=pad_value)
        audio_batch = audio_chunk.unsqueeze(0).expand(batch_size, -1)

        indicator = torch.ones((batch_size, args.n_motions), device=device) if model.use_indicator else None
        if indicator is not None and frame_len < args.n_motions:
            indicator[:, frame_len:] = 0

        motion_pred, _, _ = model.sample(
            audio_batch, shape_batch, prev_motion_feat=prev_motion, indicator=indicator,
            cfg_mode=None, cfg_cond=['audio'], cfg_scale=1.15, ret_traj=False)
        motion_chunks.append(motion_pred[:, :frame_len].detach())
        prev_motion = motion_pred[:, -args.n_prev_motions:].detach()

    motion_seq = torch.cat(motion_chunks, dim=1)
    if coef_stats is not None:
        coef_stats = {k: v.to(device) for k, v in coef_stats.items()}
    coef_dict = utils.get_coef_dict(
        motion_seq, shape_batch, coef_stats, with_global_pose=False, rot_repr=args.rot_repr)
    return motion_seq, coef_dict


@torch.no_grad()
def run_inference(args, model: DiffTalkingHead, flame, dataset, coef_stats=None, style_enc=None):
    device = model.device
    data_loader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                  collate_fn=lambda batch: batch[0])
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    def resample_vertices(verts, src_fps, target_fps):
        """Resample vertex sequence from src_fps to target_fps.
        verts shape: (batch_size, seq_len, verts_num, 3)
        """
        if src_fps == target_fps or verts.shape[1] == 0:  # Check seq_len dimension
            return verts

        seq_len = verts.shape[1]  # Time dimension is axis 1
        target_len = int(np.round(seq_len * target_fps / float(src_fps)))
        target_len = max(target_len, 1)
        src_idx = np.linspace(0, seq_len - 1, seq_len)
        tgt_idx = np.linspace(0, seq_len - 1, target_len)

        # Resample each batch, vertex coordinate
        # verts shape: (batch_size, seq_len, verts_num, 3)
        # resampled shape: (batch_size, target_len, verts_num, 3)
        resampled = np.zeros((verts.shape[0], target_len, verts.shape[2], verts.shape[3]), dtype=verts.dtype)

        for batch_idx in range(verts.shape[0]):  # For each batch
            for vert_idx in range(verts.shape[2]):  # For each vertex
                for coord_idx in range(verts.shape[3]):  # For each coordinate (x, y, z)
                    resampled[batch_idx, :, vert_idx, coord_idx] = np.interp(
                        tgt_idx, src_idx, verts[batch_idx, :, vert_idx, coord_idx]
                    )

        return resampled

    for sample in data_loader:
        motion_seq, coef_dict = generate_motion_sequence(args, model, sample, coef_stats, style_enc)
        verts = utils.coef_dict_to_vertices(coef_dict, flame, rot_repr=args.rot_repr,
                                            ignore_global_rot=False).detach().cpu().numpy()

        # Map dataset names to match baseline requirements
        dataset_name = sample['dataset_name']
        if dataset_name == 'MEAD_VHAP_test':
            dataset_name = 'MEAD_VHAP'
        elif dataset_name == 'MultiModal200_test':
            dataset_name = 'MultiModal200'

        # For MultiModal200, resample from 25fps to 20fps before saving
        if dataset_name == 'MultiModal200':
            verts = resample_vertices(verts, src_fps=25, target_fps=20)

        dataset_dir = output_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        style_id = sample.get('style_id') or build_style_id(sample.get('meta', {}))
        if style_id is None:
            style_id = str(sample['sample_id'])
            print(f'Warning: style_id missing for sample {sample["sample_id"]}, fallback to {style_id}')
        style_dir = dataset_dir / style_id
        style_dir.mkdir(parents=True, exist_ok=True)

        for rep in range(verts.shape[0]):
            filename = f'{rep}.npy' if args.n_repetitions > 1 else '0.npy'
            np.save(style_dir / filename, verts[rep])

        print(f'Saved predictions to {style_dir}')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args, option_text=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Enforce 51-dim motion (50 expr + 1 jaw)
    args.no_head_pose = True
    args.l_head_angle = 0
    args.l_head_vel = 0
    args.l_head_smooth = 0
    args.l_head_trans = 0

    if args.fps != 25:
        print('Baseline requires 25 fps; overriding fps to 25.')
        args.fps = 25

    if not args.data_roots:
        raise ValueError('data_roots must be provided (see baseline_prompt.md).')
    if not args.data_jsons:
        raise ValueError('data_jsons must list at least one JSON index file.')
    data_roots = [Path(r) for r in args.data_roots]

    # 如果没有style encoder，确保guiding_conditions不包含'style'
    if not hasattr(args, 'style_enc_ckpt') or args.style_enc_ckpt is None:
        if hasattr(args, 'guiding_conditions') and args.guiding_conditions:
            original_guiding = args.guiding_conditions
            guiding_conditions = args.guiding_conditions.split(',') if isinstance(args.guiding_conditions, str) else args.guiding_conditions
            guiding_conditions = [cond.strip() for cond in guiding_conditions if cond.strip() != 'style']
            args.guiding_conditions = ','.join(guiding_conditions) if guiding_conditions else 'audio'
            if 'style' in original_guiding:
                print(f'Warning: No style encoder provided. Removing "style" from guiding_conditions. Using: {args.guiding_conditions}')

    coef_stats_file: Optional[Path] = args.stats_file
    if coef_stats_file is not None and not coef_stats_file.is_absolute():
        coef_stats_file = data_roots[0] / coef_stats_file

    # Loss / FLAME forward
    flame = FLAME(FLAMEConfig).to(device)

    if args.mode == 'train':
        # Style Encoder
        if args.style_enc_ckpt:
            # Build model
            # Use weights_only=False for PyTorch 2.6+ compatibility (checkpoint contains argparse.Namespace)
            enc_model_data = torch.load(args.style_enc_ckpt, map_location=device, weights_only=False)
            enc_model_args = utils.NullableArgs(enc_model_data['args'])
            # Ensure style encoder args match training args
            enc_model_args.no_head_pose = args.no_head_pose
            enc_model_args.rot_repr = args.rot_repr
            style_enc = StyleEncoder(enc_model_args).to(device)
            style_enc.encoder.load_state_dict(enc_model_data['encoder'], strict=False)
            style_enc.eval()
            print(f'✓ Style Encoder loaded: input_dim={style_enc.motion_coef_dim}, output_dim={style_enc.feature_dim}')
            # Verify d_style matches style encoder output dimension
            if args.d_style != style_enc.feature_dim:
                print(f'Warning: d_style ({args.d_style}) != style encoder feature_dim ({style_enc.feature_dim}). Updating d_style.')
                args.d_style = style_enc.feature_dim
        else:
            style_enc = None

        # Build model
        model = DiffTalkingHead(args, device=device)

        # Dataset
        train_dataset = MotionJsonDataset(
            data_roots, args.data_jsons, n_motions=args.n_motions, crop_strategy='random',
            target_fps=args.fps, stats_file=coef_stats_file, split='train', rot_repr=args.rot_repr,
            pad_mode=args.pad_mode)
        val_jsons = args.val_jsons if args.val_jsons is not None else args.data_jsons
        val_dataset = MotionJsonDataset(
            data_roots, val_jsons, n_motions=args.n_motions, crop_strategy='begin',
            target_fps=args.fps, stats_file=coef_stats_file, split='val', rot_repr=args.rot_repr,
            pad_mode=args.pad_mode)
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers, pin_memory=True)
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers)

        # Logging
        exp_dir = Path('experiments/DPT') / f'{args.exp_name}-{datetime.now().strftime("%y%m%d_%H%M%S")}'
        log_dir = exp_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))
        if option_text is not None:
            with open(log_dir / 'options.log', 'w') as f:
                f.write(option_text)
            writer.add_text('options', option_text)

        print(Back.RED + Fore.YELLOW + Style.BRIGHT + exp_dir.name + Style.RESET_ALL)
        print('model parameters: ', count_parameters(model))

        # Train the model
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        # scheduler
        if args.scheduler == 'Warmup':
            from scheduler import GradualWarmupScheduler
            scheduler = GradualWarmupScheduler(optimizer, 1, args.warm_iter)
        elif args.scheduler == 'WarmupThenDecay':
            from scheduler import GradualWarmupScheduler
            after_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.cos_max_iter - args.warm_iter,
                                                                   args.lr * args.min_lr_ratio)
            scheduler = GradualWarmupScheduler(optimizer, 1, args.warm_iter, after_scheduler)
        else:
            scheduler = None

        train(args, model, style_enc, train_loader, val_loader, optimizer, exp_dir / 'checkpoints', scheduler, writer,
              flame)
    else:
        # Load model
        checkpoint_path, exp_name = utils.get_model_path(args.exp_name, args.iter)
        model_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_args = utils.NullableArgs(model_data['args'])
        model_args.no_head_pose = True
        # Allow overriding style encoder checkpoint via CLI
        override_style_ckpt = getattr(args, 'style_enc_ckpt', None)
        if override_style_ckpt:
            model_args.style_enc_ckpt = override_style_ckpt

        # Style Encoder
        if model_args.style_enc_ckpt:
            enc_model_data = torch.load(model_args.style_enc_ckpt, map_location=device, weights_only=False)
            enc_model_args = utils.NullableArgs(enc_model_data['args'])
            enc_model_args.no_head_pose = True
            style_enc = StyleEncoder(enc_model_args).to(device)
            style_enc.encoder.load_state_dict(enc_model_data['encoder'], strict=False)
            style_enc.eval()
        else:
            style_enc = None
            # If no style encoder, drop style from guiding_conditions to avoid mismatch
            if hasattr(model_args, 'guiding_conditions') and model_args.guiding_conditions:
                guiding_conditions = model_args.guiding_conditions.split(',') if isinstance(model_args.guiding_conditions, str) else model_args.guiding_conditions
                guiding_conditions = [c.strip() for c in guiding_conditions if c.strip() and c.strip() != 'style']
                model_args.guiding_conditions = ','.join(guiding_conditions) if guiding_conditions else 'audio'

        # Build model
        model = DiffTalkingHead(model_args, device=device)
        model.load_state_dict(model_data['model'])
        # Re-set attributes that are not saved in state_dict
        model.use_indicator = getattr(model_args, 'use_indicator', False)
        model.eval()

        infer_args = model_args
        infer_args.data_roots = data_roots
        infer_args.data_jsons = args.data_jsons
        infer_args.num_workers = args.num_workers
        infer_args.output_dir = args.output_dir
        infer_args.n_repetitions = args.n_repetitions
        infer_args.rot_repr = model_args.rot_repr
        infer_args.fps = model_args.fps
        infer_args.n_motions = model_args.n_motions
        infer_args.n_prev_motions = model_args.n_prev_motions
        infer_args.use_indicator = getattr(model_args, 'use_indicator', False)
        infer_args.pad_mode = getattr(model_args, 'pad_mode', 'zero') or 'zero'
        infer_args.no_head_pose = True

        coef_stats = None
        if coef_stats_file is not None and coef_stats_file.exists():
            stats_np = dict(np.load(coef_stats_file))
            coef_stats = {k: torch.tensor(v, dtype=torch.float32) for k, v in stats_np.items()}

        infer_dataset = MotionInferenceDataset(data_roots, args.data_jsons, target_fps=infer_args.fps)
        run_inference(infer_args, model, flame, infer_dataset, coef_stats, style_enc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DiffTalkingHead: Speech-Driven 3D Facial Animation')

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer', 'test'])
    parser.add_argument('--iter', type=int, default=100000, help='iteration to test')

    # Dataset
    options.add_data_options(parser)

    # Model
    options.add_model_options(parser)

    # Training
    options.add_training_options(parser)

    # Additional options depending on previous options
    options.add_additional_options(parser)

    args = parser.parse_args()
    if args.mode == 'train':
        option_text = utils.get_option_text(args, parser)
    else:
        option_text = None

    main(args, option_text)
