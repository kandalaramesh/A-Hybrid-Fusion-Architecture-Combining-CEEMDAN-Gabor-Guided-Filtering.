#reading in between images

import os
import glob
import time
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ============================================================================
# I/O FUNCTIONS
# ============================================================================

def imread_rgb01(path: str, size=None):
    img = Image.open(path).convert('RGB')
    if size:
        img = img.resize(size, Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0

def imread_gray01(path: str, size=None):
    img = Image.open(path).convert('L')
    if size:
        img = img.resize(size, Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0

def imsave01(path: str, arr01: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.clip(arr01, 0, 1)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(path, quality=95)

def rgb01_to_yuv01(rgb01: np.ndarray):
    rgb_u8 = (np.clip(rgb01, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    yuv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2YUV).astype(np.float32) / 255.0
    return yuv[..., 0], yuv[..., 1], yuv[..., 2]

def yuv01_to_rgb01(Y, U, V):
    Y = np.clip(np.squeeze(Y), 0, 1).astype(np.float32)
    U = np.clip(np.squeeze(U), 0, 1).astype(np.float32)
    V = np.clip(np.squeeze(V), 0, 1).astype(np.float32)
    Y = np.nan_to_num(Y, nan=0.5, posinf=1.0, neginf=0.0)
    U = np.nan_to_num(U, nan=0.5, posinf=1.0, neginf=0.0)
    V = np.nan_to_num(V, nan=0.5, posinf=1.0, neginf=0.0)
    yuv = np.dstack([Y, U, V]).astype(np.float32)
    yuv = np.clip(yuv, 0, 1)
    yuv_u8 = np.clip(yuv * 255.0 + 0.5, 0, 255).astype(np.uint8)
    try:
        rgb_u8 = cv2.cvtColor(yuv_u8, cv2.COLOR_YUV2RGB)
        return rgb_u8.astype(np.float32) / 255.0
    except Exception:
        return yuv

def ensure_hw_match(a: np.ndarray, b: np.ndarray):
    hb, wb = b.shape[:2]
    if a.shape[:2] == (hb, wb):
        return a, b
    pil = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    pil = pil.resize((wb, hb), Image.LANCZOS)
    a2 = np.asarray(pil, dtype=np.float32) / 255.0
    return a2, b

# ============================================================================
# DECOMPOSITION FUNCTIONS
# ============================================================================

try:
    from PyEMD import CEEMDAN
    _HAS_CEEMDAN = True
    print("✓ CEEMDAN imported successfully!")
except ImportError as e:
    _HAS_CEEMDAN = False
    print(f"✗ CEEMDAN NOT installed: {e}")
    print("  Fix: pip install PyEMD")

def gaussian_blur(gray01: np.ndarray, sigma: float):
    k = int(max(3, (int(6 * sigma) // 2) * 2 + 1))
    return cv2.GaussianBlur(gray01.astype(np.float32), (k, k), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)

def proper_ceemdan_decompose(gray01: np.ndarray, n_imfs_for_base=2, num_realizations=15, std_ratio=0.2):
    if not _HAS_CEEMDAN:
        base = gaussian_blur(gray01, sigma=5.0)
        detail = gray01 - base
        return base.astype(np.float32), detail.astype(np.float32)

    h, w = gray01.shape
    base = np.zeros((h, w), dtype=np.float32)
    detail = np.zeros((h, w), dtype=np.float32)

    import warnings
    for i in range(h):
        row = gray01[i, :]
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                ceemdan = CEEMDAN(trials=num_realizations, std=std_ratio, seed=42)
                imfs = ceemdan(row)
                n_imfs = imfs.shape[0]

                if n_imfs >= n_imfs_for_base:
                    base_row = np.sum(imfs[-n_imfs_for_base:], axis=0)
                    detail_row = np.sum(imfs[:-n_imfs_for_base], axis=0) if n_imfs > n_imfs_for_base else np.zeros_like(row)
                else:
                    base_row = imfs[-1] if n_imfs > 0 else row.copy()
                    detail_row = np.zeros_like(row)

                base_row = np.nan_to_num(base_row, nan=0.5, posinf=1.0, neginf=0.0)
                detail_row = np.nan_to_num(detail_row, nan=0.0, posinf=1.0, neginf=0.0)
                base[i, :] = np.clip(base_row[:w], 0, 1)
                detail[i, :] = np.clip(detail_row[:w], -1, 1)
        except Exception:
            blurred = gaussian_blur(row[np.newaxis, :].astype(np.float32), sigma=5.0).squeeze()
            base[i, :] = np.clip(blurred, 0, 1)
            detail[i, :] = row - base[i, :]

    reconstructed = base + detail
    residual = gray01 - reconstructed
    base = np.clip(base + residual / 2.0, 0, 1)
    detail = np.clip(detail + residual / 2.0, -1, 1)

    return base.astype(np.float32), detail.astype(np.float32)

def _boxfilter(x: np.ndarray, r: int):
    k = 2 * r + 1
    return cv2.blur(x.astype(np.float32), (k, k))

def guided_filter(I: np.ndarray, p: np.ndarray, r=8, eps=1e-4):
    mean_I = _boxfilter(I, r)
    mean_p = _boxfilter(p, r)
    corr_I = _boxfilter(I * I, r)
    corr_Ip = _boxfilter(I * p, r)
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = _boxfilter(a, r)
    mean_b = _boxfilter(b, r)
    return (mean_a * I + mean_b).astype(np.float32)

def gabor_bank_response(gray01: np.ndarray):
    sigmas = (1.0, 2.0, 3.0)
    thetas = (0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6)
    energy = np.zeros_like(gray01, dtype=np.float32)

    for s in sigmas:
        ksize = int(max(9, round(6 * s) | 1))
        lambd = max(4.0, 4 * s)
        for th in thetas:
            kern = cv2.getGaborKernel((ksize,ksize), s, th, lambd, 0.5, 0, ktype=cv2.CV_32F)
            resp = cv2.filter2D(gray01.astype(np.float32), cv2.CV_32F, kern, borderType=cv2.BORDER_REFLECT)
            energy += np.abs(resp).astype(np.float32)

    p1, p99 = np.percentile(energy, [1, 99])
    if p99 > p1:
        energy = np.clip((energy - p1) / (p99 - p1 + 1e-8), 0, 1)
    else:
        energy = energy / (energy.max() + 1e-8)

    return energy

def nsst_like_decompose_Y(gray01: np.ndarray):
    base = guided_filter(gray01, gray01, r=8, eps=1e-4)
    detail = (gray01 - base).astype(np.float32)
    energy = gabor_bank_response(gray01)
    weight = np.power(energy, 0.8).astype(np.float32)
    return base.astype(np.float32), (detail * weight).astype(np.float32)

# ============================================================================
# PYRAMID FUNCTIONS - FIXED VERSION
# ============================================================================

def build_laplacian_pyramid(img, levels=3):
    """Build Laplacian pyramid with proper size handling"""
    img = img.astype(np.float32)
    gaussian_pyr = [img]
    
    for i in range(levels):
        h, w = gaussian_pyr[-1].shape[:2]
        if h < 4 or w < 4:
            break
        next_level = cv2.pyrDown(gaussian_pyr[-1])
        gaussian_pyr.append(next_level)
    
    laplacian_pyr = []
    actual_levels = len(gaussian_pyr) - 1
    
    for i in range(actual_levels):
        h, w = gaussian_pyr[i].shape[:2]
        size = (w, h)
        up = cv2.pyrUp(gaussian_pyr[i+1], dstsize=size)
        diff = gaussian_pyr[i] - up
        laplacian_pyr.append(diff)
    
    laplacian_pyr.append(gaussian_pyr[-1])
    return gaussian_pyr, laplacian_pyr

def reconstruct_from_laplacian(laplacian_pyr):
    """Reconstruct image from Laplacian pyramid with proper error handling"""
    if not laplacian_pyr or len(laplacian_pyr) == 0:
        raise ValueError("Empty Laplacian pyramid")
    
    img = laplacian_pyr[-1].astype(np.float32)
    
    for lev in range(len(laplacian_pyr) - 2, -1, -1):
        h, w = laplacian_pyr[lev].shape[:2]
        target_size = (w, h)
        
        img_up = cv2.pyrUp(img, dstsize=target_size)
        
        if img_up.shape != laplacian_pyr[lev].shape:
            img_up = cv2.resize(img_up, (w, h), interpolation=cv2.INTER_LINEAR)
        
        img = img_up + laplacian_pyr[lev].astype(np.float32)
    
    return np.clip(img, 0, 1).astype(np.float32)

# ============================================================================
# VGG19 SALIENCY NETWORK
# ============================================================================

try:
    from torchvision.models import VGG19_Weights
    _HAS_NEW_WEIGHTS = True
except Exception:
    _HAS_NEW_WEIGHTS = False

class VGG19FeatVariable(nn.Module):
    def __init__(self, num_layers=16, pretrained=True, device='cpu'):
        super().__init__()
        if pretrained and _HAS_NEW_WEIGHTS:
            vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        else:
            vgg = models.vgg19(pretrained=pretrained).features
        
        maxlen = len(vgg)
        num_layers = int(num_layers)
        if num_layers < 1:
            num_layers = 1
        if num_layers > maxlen:
            num_layers = maxlen
        
        self.slice = nn.Sequential(*[vgg[i] for i in range(num_layers)]).to(device).eval()
        for p in self.slice.parameters():
            p.requires_grad_(False)
    
    def forward(self, x: torch.Tensor):
        x3 = x.repeat(1, 3, 1, 1)
        feats = self.slice(x3)
        sal = torch.sqrt(torch.clamp((feats ** 2).sum(dim=1, keepdim=True), 1e-12))
        return F.interpolate(sal, size=x.shape[-2:], mode='bilinear', align_corners=False)

# ============================================================================
# ADAPTIVE FUSION CONFIG
# ============================================================================

class AdaptiveFusionConfig(nn.Module):
    def __init__(self,
                 init_scale_hf=0.80, init_scale_range_hf=0.20, init_threshold_hf=0.5,
                 init_scale_lf=0.85, init_scale_range_lf=0.25, init_threshold_lf=0.5):
        super().__init__()
        self.scale_hf = nn.Parameter(torch.tensor(init_scale_hf, dtype=torch.float32))
        self.scale_range_hf = nn.Parameter(torch.tensor(init_scale_range_hf, dtype=torch.float32))
        self.threshold_hf = nn.Parameter(torch.tensor(init_threshold_hf, dtype=torch.float32))
        self.scale_lf = nn.Parameter(torch.tensor(init_scale_lf, dtype=torch.float32))
        self.scale_range_lf = nn.Parameter(torch.tensor(init_scale_range_lf, dtype=torch.float32))
        self.threshold_lf = nn.Parameter(torch.tensor(init_threshold_lf, dtype=torch.float32))
    
    def forward(self):
        scale_hf = torch.clamp(self.scale_hf, 0.5, 1.0)
        scale_range_hf = torch.clamp(self.scale_range_hf, 0.05, 0.30)
        threshold_hf = torch.clamp(self.threshold_hf, 0.3, 0.7)
        scale_lf = torch.clamp(self.scale_lf, 0.6, 1.0)
        scale_range_lf = torch.clamp(self.scale_range_lf, 0.1, 0.40)
        threshold_lf = torch.clamp(self.threshold_lf, 0.3, 0.7)
        
        return {
            'scale_hf': float(scale_hf.detach().cpu().numpy()),
            'scale_range_hf': float(scale_range_hf.detach().cpu().numpy()),
            'threshold_hf': float(threshold_hf.detach().cpu().numpy()),
            'scale_lf': float(scale_lf.detach().cpu().numpy()),
            'scale_range_lf': float(scale_range_lf.detach().cpu().numpy()),
            'threshold_lf': float(threshold_lf.detach().cpu().numpy())
        }

# ============================================================================
# FUSION FUNCTIONS
# ============================================================================

def _auto_gamma_for_y(y: np.ndarray, target_mean: float=0.42):
    mu = float(np.clip(y.mean(), 0.05, 0.95))
    g = np.log(target_mean) / np.log(mu)
    return float(np.clip(g, 0.9, 1.7))

def vgg19_hf_fusion_adaptive(det_m: np.ndarray, hf_pet: np.ndarray,
                             net: VGG19FeatVariable, device: str, fusion_params: dict):
    with torch.no_grad():
        sm_hf = net(torch.from_numpy(det_m[None, None, ...].astype(np.float32)).to(device)).squeeze().cpu().numpy().astype(np.float32)
        sp_hf = net(torch.from_numpy(hf_pet[None, None, ...].astype(np.float32)).to(device)).squeeze().cpu().numpy().astype(np.float32)
    
    w_pet_hf = sp_hf / (sm_hf + sp_hf + 1e-8)
    r_global_hf = float(sp_hf.mean() / (sp_hf.mean() + sm_hf.mean() + 1e-8))
    
    scale_hf = fusion_params['scale_hf']
    scale_range_hf = fusion_params['scale_range_hf']
    threshold_hf = fusion_params['threshold_hf']
    
    scale_hf_adjusted = scale_hf - scale_range_hf * max(0.0, r_global_hf - threshold_hf) / 0.5
    w_pet_hf = np.clip(w_pet_hf * scale_hf_adjusted, 0.0, 1.0)
    
    return ((1.0 - w_pet_hf) * det_m + w_pet_hf * hf_pet).astype(np.float32)

def fuse_luminance_LPAS(Ym: np.ndarray, Yp: np.ndarray,
                        net: VGG19FeatVariable, device: str, fusion_params: dict):
    """
    LPAS (Laplacian Pyramid Adaptive Selection) - CORRECTED VERSION
    
    Key Fixes:
    1. HF: Weighted blend (0.60 MRI, 0.40 PET)
    2. LF: Adaptive ratio based on saliency (55-65% MRI)
    3. Removed: Arbitrary scaling factors
    """
    
    # ===== Step 1: PET Tone Mapping =====
    gamma = _auto_gamma_for_y(Yp)
    Yp_tone = np.power(np.clip(Yp, 0, 1), gamma).astype(np.float32)
    
    # ===== Step 2: Build Laplacian Pyramids =====
    _, lp_m = build_laplacian_pyramid(Ym, levels=3)
    _, lp_p = build_laplacian_pyramid(Yp_tone, levels=3)
    
    # ===== Step 3: Compute Saliency for Adaptation =====
    with torch.no_grad():
        sm = net(torch.from_numpy(Ym[None, None, ...].astype(np.float32)).to(device)).squeeze().cpu().numpy().astype(np.float32)
        sp = net(torch.from_numpy(Yp_tone[None, None, ...].astype(np.float32)).to(device)).squeeze().cpu().numpy().astype(np.float32)
    
    sm_global = float(sm.mean())
    sp_global = float(sp.mean())
    r_global = sp_global / (sm_global + sp_global + 1e-8)
    
    # ===== Step 4: Initialize Fusion =====
    fused_lp = []
    num_levels = len(lp_m)
    
    # ===== High-Frequency Fusion =====
    for lev in range(num_levels - 1):
        hf_fused = 0.60 * lp_m[lev] + 0.40 * lp_p[lev]
        fused_lp.append(hf_fused.astype(np.float32))
    
    # ===== Low-Frequency Fusion =====
    if r_global > 0.55:
        alpha_lf = 0.55
    elif r_global < 0.45:
        alpha_lf = 0.65
    else:
        alpha_lf = 0.60
    
    lf_fused = (alpha_lf * lp_m[-1] + (1.0 - alpha_lf) * lp_p[-1]).astype(np.float32)
    fused_lp.append(lf_fused)
    
    # ===== Step 5: Reconstruct from Pyramid =====
    Yf = reconstruct_from_laplacian(fused_lp)
    
    # ===== Step 6: Final Clipping =====
    return np.clip(Yf, 0, 1).astype(np.float32)

def _desaturate_uv(U, V, sat=0.88):
    U2 = 0.5 + sat * (np.squeeze(U).astype(np.float32) - 0.5)
    V2 = 0.5 + sat * (np.squeeze(V).astype(np.float32) - 0.5)
    return np.clip(U2, 0, 1), np.clip(V2, 0, 1)

def chroma_neutralize_if_dark(Y: np.ndarray, U: np.ndarray, V: np.ndarray, thr: float):
    Y = np.squeeze(Y).astype(np.float32)
    U = np.squeeze(U).astype(np.float32)
    V = np.squeeze(V).astype(np.float32)
    mask = (Y < thr).astype(np.float32)
    return mask*0.5 + (1-mask)*U, mask*0.5 + (1-mask)*V

def _auto_rim_thr(Yf: np.ndarray):
    return float(np.clip(np.percentile(np.clip(Yf, 0, 1), 5), 0.04, 0.12))

def fuse_one_pair_adaptive_ceemdan(mri_path: str, pet_path: str,
                                    net: VGG19FeatVariable, device: str, fusion_config: AdaptiveFusionConfig):
    Ym, Yp, U, V = mri2, *rgb01_to_yuv01(pet)
    fusion_params = fusion_config()

    # ==============================
    # 🔹 CEEMDAN + Laplacian Fusion Logic (Hybrid)
    # ==============================
    # ✅ CEEMDAN decomposition (once!)
    # ✅ Laplacian fusion logic (saliency-weighted)
    # ✅ No double decomposition
    # ✅ Clean reconstruction

    # Decompose MRI
    base_m, hf_m = proper_ceemdan_decompose(Ym)

    # Decompose PET
    base_p, hf_p = nsst_like_decompose_Y(Yp)

        # ==============================
    # 🔹 Laplacian-based fusion (safe reconstruction)
    # ==============================

    # Use your existing LPAS fusion for low-frequency part
    # (this internally uses Laplacian pyramid and reconstructs safely)
    Yf_low = fuse_luminance_LPAS(base_m, base_p, net, device, fusion_params)

    # Simple high-frequency blending (MRI structure preserved more)
    Yf_high = 0.60 * hf_m + 0.40 * hf_p

    # Final fused luminance
    Yf = np.clip(Yf_low + Yf_high, 0, 1).astype(np.float32)


    # Reconstruct from CEEMDAN
    Yf = np.clip(Yf_low + Yf_high, 0, 1).astype(np.float32)


    # ==============================
    # 🔹 Chroma fusion (unchanged)
    # ==============================

    U_s, V_s = _desaturate_uv(U, V)
    U_N, V_N = chroma_neutralize_if_dark(Yf, U_s, V_s, _auto_rim_thr(Yf))

    return yuv01_to_rgb01(Yf, U_N, V_N)


def collect_images(dir_path: str):
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(dir_path, e)))
    return {os.path.splitext(os.path.basename(p))[0].lower(): p for p in paths}

# ============================================================================
# SALIENCY-BASED LAYER SELECTION
# ============================================================================

def saliency_ratio_for_L(Ym, Yp_tone, net: VGG19FeatVariable, device: str):
    with torch.no_grad():
        sm = net(torch.from_numpy(Ym[None, None, ...].astype(np.float32)).to(device)).squeeze().cpu().numpy().astype(np.float32)
        sp = net(torch.from_numpy(Yp_tone[None, None, ...].astype(np.float32)).to(device)).squeeze().cpu().numpy().astype(np.float32)
    
    Em = float(sm.mean())
    Ep = float(sp.mean())
    ratio = Ep / (Ep + Em + 1e-8)
    return ratio

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    def get_range_for_fusion(tag):
        """
        Returns:
        None -> process all
        (from, to) -> process range (1-based, inclusive)
        "SKIP" -> skip this fusion
        """
        try:
            _inp = input(
                f"Enter image index range for {tag} fusion "
                f"(FROM TO), press Enter for ALL, or enter 0 to SKIP: "
            ).strip()
            
            if _inp == "0":
                return "SKIP"
            if _inp == "":
                return None
            
            parts = _inp.split()
            if len(parts) != 2:
                raise ValueError("Please enter exactly two numbers: FROM TO")
            
            idx_from = int(parts[0])
            idx_to = int(parts[1])
            
            if idx_from < 1 or idx_to < idx_from:
                raise ValueError("Invalid range values")
            
            return (idx_from, idx_to)
        
        except Exception as e:
            print("Invalid input. Processing ALL images. Reason:", e)
            return None
    
    # =========================
    # Device
    # =========================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    
    # =========================
    # Paths
    # =========================
    MRI_ROOT_T1 = r"D:\codes\brain_atlas\MR_T1"
    MRI_ROOT_T2 = r"D:\codes\brain_atlas\MR_T2"
    PET_ROOT = r"D:\codes\brain_atlas\PET"
    OUT_ROOT_BASE = r"D:\codes\work2\New folder\EMD-NSST-Fusion\combinations\vggfull\SampleLPASadaptive_saliencyT15_fixed_newLs"
    
    # =========================
    # VGG parameters
    # =========================
    # candidate_layers = [12, 14, 16, 18, 21, 23, 25, 27]
    candidate_layers = [11, 13, 15, 17, 20, 22, 24, 26]
    # candidate_layers = [35]
    target_ratio = 0.55
    
    nets = {}
    for L in candidate_layers:
        print(f"Creating VGG19 net with num_layers={L}")
        nets[L] = VGG19FeatVariable(
            num_layers=L, pretrained=True, device=device
        ).to(device).eval()
    
    fusion_config = AdaptiveFusionConfig().to(device)
    
    # =========================
    # Separate prompts
    # =========================
    range_T1 = get_range_for_fusion("T1–PET")
    range_T2 = get_range_for_fusion("T2–PET")
    
    fusion_jobs = []
    if range_T1 != "SKIP":
        fusion_jobs.append(("MR_T1", MRI_ROOT_T1, "T1-PET", range_T1))
    if range_T2 != "SKIP":
        fusion_jobs.append(("MR_T2", MRI_ROOT_T2, "T2-PET", range_T2))
    
    if not fusion_jobs:
        print("No fusion selected. Exiting.")
        exit(0)
    
    # =========================
    # Processing loop
    # =========================
    for MRI_FOLDER, MRI_DIR, OUT_TAG, IDX_RANGE in fusion_jobs:
        OUT_DIR = os.path.join(OUT_ROOT_BASE, OUT_TAG)
        PET_DIR = PET_ROOT
        os.makedirs(OUT_DIR, exist_ok=True)
        
        import csv
        csv_log_path = os.path.join(OUT_DIR, "log_layers.csv")
        txt_log_path = os.path.join(OUT_DIR, "runtime.txt")
        
        csv_file = open(csv_log_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["MRI_Type", "Image", "Chosen_L", "Saliency_Cost", "Time(sec)"])
        
        txt_log = open(txt_log_path, "w")
        txt_log.write(f"Execution Log\nDevice: {device}\n\n")
        
        print("\n" + "="*70)
        print(f"Processing set: {OUT_TAG}")
        print("="*70)
        
        mri_paths = collect_images(MRI_DIR)
        pet_paths = collect_images(PET_DIR)
        
        keys_all = sorted(set(mri_paths) & set(pet_paths))
        
        if IDX_RANGE is None:
            keys = keys_all
            print(f"Selected ALL {len(keys)} matching pairs.")
        else:
            idx_from, idx_to = IDX_RANGE
            start = idx_from - 1
            end = min(idx_to, len(keys_all))
            keys = keys_all[start:end]
            print(f"Selected images {idx_from} to {end} out of {len(keys_all)} pairs.")
        
        total_start = time.time()
        success_count = 0
        fail_count = 0
        
        for idx, k in enumerate(keys):
            try:
                mri_path = mri_paths[k]
                pet_path = pet_paths[k]
                
                mri = imread_gray01(mri_path)
                pet = imread_rgb01(pet_path)
                mri2, pet = ensure_hw_match(mri, pet)
                
                Ym, Yp, U, V = mri2, *rgb01_to_yuv01(pet)
                
                gamma = _auto_gamma_for_y(Yp)
                Yp_tone = np.power(np.clip(Yp, 0, 1), gamma).astype(np.float32)
                
                best_L = None
                best_cost = 1e9
                
                for L in candidate_layers:
                    rL = saliency_ratio_for_L(Ym, Yp_tone, nets[L], device)
                    cost = (rL - target_ratio) ** 2
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_L = L
                
                fused_img = fuse_one_pair_adaptive_ceemdan(
                    mri_path, pet_path, nets[best_L], device, fusion_config
                )
                
                out_path = os.path.join(OUT_DIR, f"{k}.png")
                imsave01(out_path, fused_img)
                
                elapsed = time.time() - total_start
                
                print(f"[{idx+1}/{len(keys)}] {k} → L={best_L}, cost={best_cost:.2e} ✓")
                csv_writer.writerow([OUT_TAG, k, best_L, f"{best_cost:.4e}", f"{elapsed:.3f}"])
                txt_log.write(f"{OUT_TAG}/{k} L={best_L}, Cost={best_cost:.4e}, {elapsed:.2f}s\n")
                
                success_count += 1
                
            except Exception as e:
                print(f"[{idx+1}/{len(keys)}] {k} FAILED: {str(e)[:80]}")
                txt_log.write(f"ERROR {OUT_TAG}/{k}: {str(e)}\n")
                fail_count += 1
        
        total_time = time.time() - total_start
        csv_file.close()
        txt_log.close()
        
        print("\nDone:", OUT_TAG)
        print("Success:", success_count, "Failed:", fail_count)
        print("Total Time(s):", total_time)
        if success_count:
            print("Avg per image:", total_time / success_count)
