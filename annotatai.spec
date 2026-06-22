# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec para AnnotatAI (build one-folder, Windows, torch CUDA).

Construir:  pyinstaller annotatai.spec --noconfirm
Salida:     dist/AnnotatAI/AnnotatAI.exe  (repartir la carpeta dist/AnnotatAI completa)

Los modelos (DINOv3, SigLIP2, SAM2, YOLO) NO se incluyen: se descargan en el
primer uso a la cache de HuggingFace/ultralytics del usuario.
"""

from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

# Paquetes ML que cargan submódulos/configs dinámicamente y PyInstaller no rastrea solo.
_HEAVY = [
    "ultralytics",
    "transformers",
    "tokenizers",
    "open_clip",
    "timm",
    "umap",
    "numba",
    "llvmlite",
    "hdbscan",
    "sklearn",
    "scipy",
    "safetensors",
    "huggingface_hub",
    "PIL",
]
for _pkg in _HEAVY:
    try:
        d, b, h = collect_all(_pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:  # noqa: BLE001
        print(f"[spec] collect_all({_pkg!r}) falló: {exc}")

# torch + torchvision (incluye runtime CUDA empaquetado en los wheels cu121)
for _pkg in ["torch", "torchvision"]:
    try:
        d, b, h = collect_all(_pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:  # noqa: BLE001
        print(f"[spec] collect_all({_pkg!r}) falló: {exc}")

# Paquetes nvidia cuda (cublas, cudnn, etc.) que vienen como wheels separados.
for _nv in [
    "nvidia.cublas", "nvidia.cuda_runtime", "nvidia.cuda_nvrtc", "nvidia.cudnn",
    "nvidia.cufft", "nvidia.curand", "nvidia.cusolver", "nvidia.cusparse",
    "nvidia.nccl", "nvidia.nvtx", "nvidia.cuda_cupti",
]:
    try:
        d, b, h = collect_all(_nv)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:  # noqa: BLE001
        pass

hiddenimports += [
    "sklearn.utils._typedefs",
    "sklearn.neighbors._partition_nodes",
    "scipy._lib.array_api_compat.numpy.fft",
]

block_cipher = None

a = Analysis(
    ["annotatai.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib.tests", "test", "tests"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AnnotatAI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,      # GUI app (sin consola)
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AnnotatAI",
)
