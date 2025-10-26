import torch, os, glob
print("torch:", torch.__version__)
try:
    import torchtext
    print("torchtext:", torchtext.__version__)
except Exception as e:
    print("torchtext import failed:", e)
# 列出 torchtext 包内的 pyd/dll
import pkgutil
try:
    import torchtext as tt
    d = os.path.dirname(tt.__file__)
    print("torchtext dir:", d)
    print("pyd/dll:", glob.glob(os.path.join(d,"*.*")))
except Exception:
    pass