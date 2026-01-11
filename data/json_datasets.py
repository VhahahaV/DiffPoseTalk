import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchaudio
from torch.utils import data

from datasets import build_style_id, load_flame_coefficients, TARGET_FPS

# Set soundfile backend for torchaudio to avoid torchcodec dependency
try:
    torchaudio.set_audio_backend('soundfile')
except Exception:
    # If soundfile backend fails, try to use default
    pass


def _resolve_path_with_dataset_prefix(rel_path: str, dataset_name: str, data_roots: List[Path]) -> Optional[Path]:
    """Resolve a relative path that may include dataset name prefix.
    
    Args:
        rel_path: Relative path from JSON (e.g., 'MultiModal200/m_ch009/...')
        dataset_name: Dataset name (e.g., 'MultiModal200')
        data_roots: List of data root directories
    
    Returns:
        Resolved Path if found, None otherwise
    """
    if rel_path is None:
        return None
    
    # Find matching data_root by dataset name
    for root in data_roots:
        root_name = root.name
        if root_name.lower() != dataset_name.lower():
            continue
        
        # Try direct path
        candidate = root / rel_path
        if candidate.exists():
            return root
        
        # Try removing dataset name prefix if present
        if rel_path.startswith(f'{dataset_name}/'):
            rel_path_stripped = rel_path[len(dataset_name)+1:]
            candidate = root / rel_path_stripped
            if candidate.exists():
                return root
    
    # Fallback: try all roots without strict dataset matching
    for root in data_roots:
        # Try direct path
        if (root / rel_path).exists():
            return root
        # Try removing common dataset prefixes
        for prefix in ['MultiModal200/', 'MEAD_VHAP/', 'digital_human/']:
            if rel_path.startswith(prefix):
                rel_path_stripped = rel_path[len(prefix):]
                if (root / rel_path_stripped).exists():
                    return root
    
    return None


def _strip_dataset_prefix(rel_path: str, root: Path) -> str:
    """Strip dataset name prefix from a relative path if root matches.

    Args:
        rel_path: Relative path that may include dataset prefix
        root: Data root directory

    Returns:
        Path with dataset prefix removed if applicable
    """
    root_name = root.name
    # Try removing root name prefix (e.g., 'MultiModal200/...' -> '...')
    if rel_path.startswith(f'{root_name}/'):
        return rel_path[len(root_name)+1:]  # +1 to skip the '/'
    # For single root configuration (baseline_prompt.md), don't strip dataset prefixes
    # The paths in JSON are already relative to the single data root
    return rel_path


def _load_stats(stats_file: Optional[Path]) -> Optional[Dict[str, torch.Tensor]]:
    if stats_file is None:
        return None
    if not stats_file.exists():
        print(f'Warning: stats file {stats_file} not found. Skipping normalization.')
        return None
    coef_stats = dict(np.load(stats_file))
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in coef_stats.items()}


class MotionJsonDataset(data.Dataset):
    """Dataset that reads multiple JSON index files and crops two clips."""

    def __init__(
            self,
            data_roots: List[Path],
            data_jsons: List[str],
            n_motions: int = 100,
            crop_strategy: str = 'random',
            target_fps: int = TARGET_FPS,
            stats_file: Optional[Path] = None,
            split: str = 'train',
            rot_repr: str = 'aa',
            pad_mode: str = 'zero'):
        super().__init__()
        self.data_roots = [Path(r) for r in data_roots]
        if len(self.data_roots) == 0:
            raise ValueError('data_roots cannot be empty')
        self.data_jsons = data_jsons
        self.n_motions = n_motions
        self.crop_strategy = crop_strategy if split == 'train' else 'begin'
        self.target_fps = target_fps
        self.rot_representation = rot_repr
        self.pad_mode = pad_mode

        self.coef_fps = target_fps
        self.audio_unit = 16000. / self.coef_fps
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.coef_total_len = self.n_motions * 2
        self.audio_total_len = round(self.audio_unit * self.coef_total_len)

        self.stats_file = stats_file
        self.coef_stats = _load_stats(stats_file)
        self.entries = self._load_entries()

    def _resolve_json_path(self, json_rel: str) -> Path:
        json_path = Path(json_rel)
        if json_path.is_absolute() and json_path.exists():
            return json_path
        for root in self.data_roots:
            candidate = root / json_rel
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f'Cannot locate json file {json_rel}')

    @staticmethod
    def _select_dataset_name(meta: Dict[str, Any], json_path: Path) -> str:
        return meta.get('dataset') or meta.get('dataset_name') or json_path.stem

    def _resolve_sample_root(self, meta: Dict[str, Any], json_path: Path) -> Path:
        audio_rel = meta.get('audio_path') or meta.get('audio')
        flame_rel = meta.get('flame_coeff_save_path') or meta.get('flame_path') or meta.get('flame_coeff_path')
        dataset_name = self._select_dataset_name(meta, json_path)
        
        # Try to resolve using dataset-aware path resolution
        for rel_path in [audio_rel, flame_rel]:
            root = _resolve_path_with_dataset_prefix(rel_path, dataset_name, self.data_roots)
            if root is not None:
                return root
        
        # Fallback
        return self.data_roots[0] if self.data_roots else json_path.parent

    def _load_entries(self) -> List[Dict[str, Any]]:
        entries = []
        for json_rel in self.data_jsons:
            json_path = self._resolve_json_path(json_rel)
            with open(json_path, 'r') as f:
                content = json.load(f)
            for sample_id, meta in content.items():
                dataset_name = self._select_dataset_name(meta, json_path)
                root = self._resolve_sample_root(meta, json_path)
                entries.append({
                    'json_path': json_path,
                    'sample_id': sample_id,
                    'meta': meta,
                    'dataset_name': dataset_name,
                    'root': root,
                })
        if len(entries) == 0:
            raise RuntimeError('No samples found in provided json files.')
        return entries

    def __len__(self):
        return len(self.entries)

    def _load_audio(self, path: Path) -> torch.Tensor:
        # Use soundfile backend (should be set at module level)
        try:
            audio, sr = torchaudio.load(str(path))
        except Exception as e:
            raise RuntimeError(f'Failed to load audio {path}: {e}')

        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        audio = audio.squeeze(0)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        return audio

    def _apply_coef_stats(self, coef_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.coef_stats is None:
            return coef_dict
        out = {}
        for key in ['exp', 'pose', 'shape']:
            if key not in coef_dict:
                continue
            mean_key = f'{key}_mean'
            std_key = f'{key}_std'
            if mean_key in self.coef_stats and std_key in self.coef_stats:
                mean = self.coef_stats[mean_key].to(coef_dict[key].device)
                std = self.coef_stats[std_key].to(coef_dict[key].device)
                out[key] = (coef_dict[key] - mean) / (std + 1e-9)
            else:
                out[key] = coef_dict[key]
        return out

    def __getitem__(self, index):
        entry = self.entries[index]
        meta = entry['meta']
        dataset_name = entry['dataset_name']
        root = entry['root']

        flame_rel = meta.get('flame_coeff_save_path') or meta.get('flame_path') or meta.get('flame_coeff_path')
        if flame_rel is None:
            raise KeyError(f'Missing flame path for sample {entry["sample_id"]}')
        # Strip dataset prefix if present
        flame_rel_stripped = _strip_dataset_prefix(flame_rel, root)
        flame_path = root / flame_rel_stripped
        if not flame_path.exists():
            raise FileNotFoundError(f'FLAME coefficients not found: {flame_path}')

        coef_np = load_flame_coefficients(flame_path, dataset_name, self.target_fps)
        exp = coef_np['exp']
        pose = coef_np['pose']
        shape_vec = coef_np['shape']

        seq_len = exp.shape[0]
        if seq_len < self.coef_total_len:
            pad_len = self.coef_total_len - seq_len
            exp = np.pad(exp, ((0, pad_len), (0, 0)), mode='edge')
            pose = np.pad(pose, ((0, pad_len), (0, 0)), mode='edge')
            seq_len = exp.shape[0]

        if self.crop_strategy == 'random':
            start_frame = random.randint(0, seq_len - self.coef_total_len)
        elif self.crop_strategy == 'begin':
            start_frame = 0
        elif self.crop_strategy == 'center':
            start_frame = max((seq_len - self.coef_total_len) // 2, 0)
        else:
            raise ValueError(f'Unknown crop strategy: {self.crop_strategy}')

        end_frame = start_frame + self.coef_total_len
        exp_clip = torch.tensor(exp[start_frame:end_frame], dtype=torch.float32)
        pose_clip = torch.tensor(pose[start_frame:end_frame], dtype=torch.float32)
        shape_clip = torch.tensor(
            np.broadcast_to(shape_vec[None, :], (self.coef_total_len, shape_vec.shape[0])),
            dtype=torch.float32
        )

        coef_clip = {'exp': exp_clip, 'pose': pose_clip, 'shape': shape_clip}
        coef_clip = self._apply_coef_stats(coef_clip)

        audio_rel = meta.get('audio_path') or meta.get('audio')
        if audio_rel is None:
            raise KeyError(f'Missing audio_path for sample {entry["sample_id"]}')
        # Strip dataset prefix if present
        audio_rel_stripped = _strip_dataset_prefix(audio_rel, root)
        audio_path = root / audio_rel_stripped
        if not audio_path.exists():
            raise FileNotFoundError(f'Audio not found: {audio_path}')
        audio = self._load_audio(audio_path)

        start_sample = round(start_frame * self.audio_unit)
        end_sample = start_sample + self.audio_total_len
        if end_sample > audio.shape[0]:
            if self.pad_mode == 'replicate' and audio.numel() > 0:
                pad_value = audio[-1]
            else:
                pad_value = 0
            audio = torch.nn.functional.pad(audio, (0, end_sample - audio.shape[0]), value=pad_value)
        audio_clip = audio[start_sample:end_sample]

        audio_mean = torch.mean(audio_clip)
        audio_std = torch.std(audio_clip)
        audio_clip = (audio_clip - audio_mean) / (audio_std + 1e-5)

        audio_pair = [audio_clip[:self.n_audio_samples].clone(), audio_clip[-self.n_audio_samples:].clone()]
        coef_pair = [
            {k: v[:self.n_motions].clone() for k, v in coef_clip.items()},
            {k: v[-self.n_motions:].clone() for k, v in coef_clip.items()},
        ]

        return audio_pair, coef_pair, (float(audio_mean), float(audio_std))


class MotionInferenceDataset(data.Dataset):
    """Dataset for full-sequence inference outputs."""

    def __init__(self, data_roots: List[Path], data_jsons: List[str], target_fps: int = TARGET_FPS):
        super().__init__()
        self.data_roots = [Path(r) for r in data_roots]
        if len(self.data_roots) == 0:
            raise ValueError('data_roots cannot be empty')
        self.data_jsons = data_jsons
        self.target_fps = target_fps
        self.entries = self._load_entries()
        self.audio_unit = 16000. / self.target_fps

    def _resolve_json_path(self, json_rel: str) -> Path:
        json_path = Path(json_rel)
        if json_path.is_absolute() and json_path.exists():
            return json_path
        for root in self.data_roots:
            candidate = root / json_rel
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f'Cannot locate json file {json_rel}')

    @staticmethod
    def _select_dataset_name(meta: Dict[str, Any], json_path: Path) -> str:
        return meta.get('dataset') or meta.get('dataset_name') or json_path.stem

    def _resolve_sample_root(self, meta: Dict[str, Any], json_path: Path) -> Path:
        audio_rel = meta.get('audio_path') or meta.get('audio')
        flame_rel = meta.get('flame_coeff_save_path') or meta.get('flame_path') or meta.get('flame_coeff_path')
        dataset_name = self._select_dataset_name(meta, json_path)
        
        # Try to resolve using dataset-aware path resolution
        for rel_path in [audio_rel, flame_rel]:
            root = _resolve_path_with_dataset_prefix(rel_path, dataset_name, self.data_roots)
            if root is not None:
                return root
        
        # Fallback
        return self.data_roots[0] if self.data_roots else json_path.parent

    def _load_entries(self) -> List[Dict[str, Any]]:
        entries = []
        for json_rel in self.data_jsons:
            json_path = self._resolve_json_path(json_rel)
            with open(json_path, 'r') as f:
                content = json.load(f)
            for sample_id, meta in content.items():
                dataset_name = self._select_dataset_name(meta, json_path)
                root = self._resolve_sample_root(meta, json_path)
                style_id = build_style_id(meta)
                entries.append({
                    'json_path': json_path,
                    'sample_id': sample_id,
                    'meta': meta,
                    'dataset_name': dataset_name,
                    'root': root,
                    'style_id': style_id,
                })
        if len(entries) == 0:
            raise RuntimeError('No samples found in provided json files.')
        return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        meta = entry['meta']
        dataset_name = entry['dataset_name']
        root = entry['root']

        flame_rel = meta.get('flame_coeff_save_path') or meta.get('flame_path') or meta.get('flame_coeff_path')
        if flame_rel is None:
            raise KeyError(f'Missing flame path for sample {entry["sample_id"]}')
        # Strip dataset prefix if present
        flame_rel_stripped = _strip_dataset_prefix(flame_rel, root)
        flame_path = root / flame_rel_stripped
        if not flame_path.exists():
            raise FileNotFoundError(f'FLAME coefficients not found: {flame_path}')
        coef_np = load_flame_coefficients(flame_path, dataset_name, self.target_fps)
        exp = torch.tensor(coef_np['exp'], dtype=torch.float32)
        jaw = torch.tensor(coef_np['jaw'], dtype=torch.float32)
        pose = torch.tensor(coef_np['pose'], dtype=torch.float32)
        shape_vec = torch.tensor(coef_np['shape'], dtype=torch.float32)

        motion = torch.cat([exp, jaw], dim=-1)
        seq_len = motion.shape[0]

        audio_rel = meta.get('audio_path') or meta.get('audio')
        if audio_rel is None:
            raise KeyError(f'Missing audio_path for sample {entry["sample_id"]}')
        # Strip dataset prefix if present
        audio_rel_stripped = _strip_dataset_prefix(audio_rel, root)
        audio_path = root / audio_rel_stripped
        if not audio_path.exists():
            raise FileNotFoundError(f'Audio not found: {audio_path}')
        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        audio = audio.squeeze(0)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)

        audio_mean = torch.mean(audio)
        audio_std = torch.std(audio)
        audio = (audio - audio_mean) / (audio_std + 1e-5)

        return {
            'audio': audio,
            'audio_mean': float(audio_mean),
            'audio_std': float(audio_std),
            'motion': motion,
            'pose': pose,
            'shape': shape_vec,
            'seq_len': seq_len,
            'dataset_name': dataset_name,
            'style_id': entry.get('style_id'),
            'sample_id': entry['sample_id'],
            'meta': meta,
        }


class MotionJsonDatasetForSE(data.Dataset):
    """Dataset for Style Encoder training: returns two consecutive motion clips (51-dim: 50 expr + 1 jaw)."""

    def __init__(
            self,
            data_roots: List[Path],
            data_jsons: List[str],
            n_motions: int = 100,
            crop_strategy: str = 'random',
            target_fps: int = TARGET_FPS,
            stats_file: Optional[Path] = None,
            rot_repr: str = 'aa',
            no_head_pose: bool = True):
        super().__init__()
        self.data_roots = [Path(r) for r in data_roots]
        if len(self.data_roots) == 0:
            raise ValueError('data_roots cannot be empty')
        self.data_jsons = data_jsons
        self.n_motions = n_motions
        self.crop_strategy = crop_strategy
        self.target_fps = target_fps
        self.rot_representation = rot_repr
        self.no_head_pose = no_head_pose

        self.coef_fps = target_fps
        # Total length for two consecutive clips (with some overlap)
        self.coef_total_len = int(self.n_motions * 2.1)

        self.stats_file = stats_file
        self.coef_stats = _load_stats(stats_file)
        self.entries = self._load_entries()

    def _resolve_json_path(self, json_rel: str) -> Path:
        json_path = Path(json_rel)
        if json_path.is_absolute() and json_path.exists():
            return json_path
        for root in self.data_roots:
            candidate = root / json_rel
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f'Cannot locate json file {json_rel}')

    @staticmethod
    def _select_dataset_name(meta: Dict[str, Any], json_path: Path) -> str:
        return meta.get('dataset') or meta.get('dataset_name') or json_path.stem

    def _resolve_sample_root(self, meta: Dict[str, Any], json_path: Path) -> Path:
        audio_rel = meta.get('audio_path') or meta.get('audio')
        flame_rel = meta.get('flame_coeff_save_path') or meta.get('flame_path') or meta.get('flame_coeff_path')
        dataset_name = self._select_dataset_name(meta, json_path)
        
        # Try to resolve using dataset-aware path resolution
        for rel_path in [audio_rel, flame_rel]:
            root = _resolve_path_with_dataset_prefix(rel_path, dataset_name, self.data_roots)
            if root is not None:
                return root
        
        # Fallback
        return self.data_roots[0] if self.data_roots else json_path.parent

    def _load_entries(self) -> List[Dict[str, Any]]:
        entries = []
        for json_rel in self.data_jsons:
            json_path = self._resolve_json_path(json_rel)
            with open(json_path, 'r') as f:
                content = json.load(f)
            for sample_id, meta in content.items():
                dataset_name = self._select_dataset_name(meta, json_path)
                root = self._resolve_sample_root(meta, json_path)
                entries.append({
                    'json_path': json_path,
                    'sample_id': sample_id,
                    'meta': meta,
                    'dataset_name': dataset_name,
                    'root': root,
                })
        if len(entries) == 0:
            raise RuntimeError('No samples found in provided json files.')
        return entries

    def __len__(self):
        return len(self.entries)

    def _apply_coef_stats(self, coef_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.coef_stats is None:
            return coef_dict
        out = {}
        # For style encoder, we normalize exp and jaw separately
        # Handle exp normalization
        if 'exp' in coef_dict:
            if 'exp_mean' in self.coef_stats and 'exp_std' in self.coef_stats:
                mean = self.coef_stats['exp_mean'].to(coef_dict['exp'].device)
                std = self.coef_stats['exp_std'].to(coef_dict['exp'].device)
                out['exp'] = (coef_dict['exp'] - mean) / (std + 1e-9)
            else:
                out['exp'] = coef_dict['exp']
        
        # Handle jaw normalization (may be in jaw_mean/jaw_std or pose_mean/pose_std)
        if 'jaw' in coef_dict:
            if 'jaw_mean' in self.coef_stats and 'jaw_std' in self.coef_stats:
                mean = self.coef_stats['jaw_mean'].to(coef_dict['jaw'].device)
                std = self.coef_stats['jaw_std'].to(coef_dict['jaw'].device)
                out['jaw'] = (coef_dict['jaw'] - mean) / (std + 1e-9)
            elif 'pose_mean' in self.coef_stats and 'pose_std' in self.coef_stats:
                # Extract jaw part from pose stats (pose[:, 3:4])
                pose_mean = self.coef_stats['pose_mean'].to(coef_dict['jaw'].device)
                pose_std = self.coef_stats['pose_std'].to(coef_dict['jaw'].device)
                if pose_mean.ndim == 1 and pose_mean.shape[0] >= 4:
                    jaw_mean = pose_mean[3:4]
                    jaw_std = pose_std[3:4]
                    out['jaw'] = (coef_dict['jaw'] - jaw_mean) / (jaw_std + 1e-9)
                else:
                    out['jaw'] = coef_dict['jaw']
            else:
                out['jaw'] = coef_dict['jaw']
        
        return out

    def __getitem__(self, index):
        entry = self.entries[index]
        meta = entry['meta']
        dataset_name = entry['dataset_name']
        root = entry['root']

        flame_rel = meta.get('flame_coeff_save_path') or meta.get('flame_path') or meta.get('flame_coeff_path')
        if flame_rel is None:
            raise KeyError(f'Missing flame path for sample {entry["sample_id"]}')
        # Strip dataset prefix if present
        flame_rel_stripped = _strip_dataset_prefix(flame_rel, root)
        flame_path = root / flame_rel_stripped
        if not flame_path.exists():
            raise FileNotFoundError(f'FLAME coefficients not found: {flame_path}')

        coef_np = load_flame_coefficients(flame_path, dataset_name, self.target_fps)
        exp = coef_np['exp']
        jaw = coef_np['jaw']

        seq_len = exp.shape[0]
        if seq_len < self.coef_total_len:
            pad_len = self.coef_total_len - seq_len
            exp = np.pad(exp, ((0, pad_len), (0, 0)), mode='edge')
            jaw = np.pad(jaw, ((0, pad_len), (0, 0)), mode='edge')
            seq_len = exp.shape[0]

        if self.crop_strategy == 'random':
            start_frame = random.randint(0, max(0, seq_len - self.coef_total_len))
        elif self.crop_strategy == 'begin':
            start_frame = 0
        elif self.crop_strategy == 'end':
            start_frame = max(0, seq_len - self.coef_total_len)
        else:
            raise ValueError(f'Unknown crop strategy: {self.crop_strategy}')

        end_frame = start_frame + self.coef_total_len
        exp_clip = torch.tensor(exp[start_frame:end_frame], dtype=torch.float32)
        jaw_clip = torch.tensor(jaw[start_frame:end_frame], dtype=torch.float32)

        coef_clip = {'exp': exp_clip, 'jaw': jaw_clip}
        coef_clip = self._apply_coef_stats(coef_clip)

        # Extract two consecutive motion clips (51-dim: 50 expr + 1 jaw)
        motion_coef = torch.cat([coef_clip['exp'], coef_clip['jaw']], dim=-1)  # (total_len, 51)
        
        # Extract two consecutive clips
        coef_pair = [
            motion_coef[:self.n_motions].clone(),  # First clip
            motion_coef[-self.n_motions:].clone(),  # Second clip
        ]

        return coef_pair
