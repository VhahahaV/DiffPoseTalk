import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

TARGET_FPS = 25


def detect_source_fps(dataset_name: str, frame_count: int) -> int:
    """Infer source fps using dataset name hints or frame count fallback."""
    name = (dataset_name or '').lower()
    if 'multimodal' in name:
        return 20
    if 'mead' in name:
        return 25
    return 20 if frame_count < 3000 else 25


def resample_sequence(seq: Optional[np.ndarray], src_fps: int, target_fps: int = TARGET_FPS) -> Optional[np.ndarray]:
    """Linearly resample a temporal sequence to target fps."""
    if seq is None:
        return None
    seq = np.asarray(seq)
    if seq.ndim == 1:
        seq = seq[:, None]
        squeeze_back = True
    else:
        squeeze_back = False

    if src_fps == target_fps or seq.shape[0] == 0:
        return seq.squeeze(-1) if squeeze_back else seq

    target_len = int(np.round(seq.shape[0] * target_fps / float(src_fps)))
    target_len = max(target_len, 1)
    src_idx = np.linspace(0, seq.shape[0] - 1, seq.shape[0])
    tgt_idx = np.linspace(0, seq.shape[0] - 1, target_len)
    resampled = np.stack([np.interp(tgt_idx, src_idx, seq[..., i]) for i in range(seq.shape[-1])], axis=-1)
    resampled = resampled.astype(np.float32)
    return resampled.squeeze(-1) if squeeze_back else resampled


def _select_first_available(data: Dict[str, Any], keys) -> Optional[np.ndarray]:
    for key in keys:
        if key in data:
            return data[key]
    return None


def load_flame_coefficients(npz_path: Path, dataset_name: str, target_fps: int = TARGET_FPS) -> Dict[str, Any]:
    """Load FLAME coefficients with unified schema (exp, jaw, neck, pose, shape)."""
    data = dict(np.load(npz_path, allow_pickle=True))
    is_digital_human = 'expcode' in data

    if is_digital_human:
        exp = data['expcode']
        posecode = data['posecode']
        shape = data.get('shapecode')
        jaw = posecode[:, 3:4]
        neck_raw = None
    else:
        exp = _select_first_available(data, ['expr', 'exp'])
        jaw_raw = _select_first_available(data, ['jaw_pose', 'jaw'])
        neck_raw = _select_first_available(data, ['neck_pose', 'neck'])
        shape = _select_first_available(data, ['shape', 'shapecode', 'shape_params'])
        if jaw_raw is None and 'pose' in data:
            jaw_raw = data['pose'][:, 3:4]
        if jaw_raw is None and 'posecode' in data:
            jaw_raw = data['posecode'][:, 3:4]
        jaw = jaw_raw[:, :1] if jaw_raw is not None else None

    if exp is None:
        raise KeyError(f'Cannot find expression coefficients in {npz_path}')

    frame_count = exp.shape[0]
    src_fps = detect_source_fps(dataset_name, frame_count)

    exp = resample_sequence(exp, src_fps, target_fps)
    jaw = resample_sequence(jaw, src_fps, target_fps)
    neck = resample_sequence(neck_raw, src_fps, target_fps)
    if jaw is None:
        jaw = np.zeros((exp.shape[0], 1), dtype=np.float32)
    else:
        jaw = jaw.astype(np.float32)
    if neck is None:
        neck = np.zeros((exp.shape[0], 3), dtype=np.float32)
    else:
        neck = neck.astype(np.float32)

    pose = np.zeros((exp.shape[0], 6), dtype=np.float32)
    pose[:, 3:4] = jaw[:, :1]

    if shape is None:
        shape_vec = np.zeros((100,), dtype=np.float32)
    else:
        shape = np.asarray(shape)
        if shape.ndim > 1:
            shape_vec = shape[0]
        else:
            shape_vec = shape
        shape_vec = shape_vec.astype(np.float32)

    return {
        'exp': exp.astype(np.float32),
        'jaw': jaw,
        'neck': neck,
        'pose': pose,
        'shape': shape_vec,
        'fps': target_fps,
        'dataset': dataset_name,
    }


def build_style_id(meta: Dict[str, Any]) -> Optional[str]:
    """Construct STYLE_ID string if speaker/emotion are available."""
    if 'style_id' in meta and meta['style_id']:
        return meta['style_id']
    speaker = meta.get('speaker_id') or meta.get('speaker') or meta.get('spk_id')
    emotion = meta.get('emotion') or meta.get('emo') or meta.get('emotion_id')

    # Handle case where speaker_id is a list like ['W024', 'happy']
    if isinstance(speaker, list) and len(speaker) >= 2:
        speaker, emotion = speaker[0], speaker[1]

    if speaker is None or emotion is None:
        return None
    return f'["{speaker}", "{emotion}"]_passionate'
