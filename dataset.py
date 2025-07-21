import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapDataset(Dataset):
    """
    .npy（memmap）を直接読む軽量 Dataset
    __getitem__ は極力軽く保ち、CPU ⇆ RAM コピーを避ける
    """

    def __init__(self, x_mm: np.memmap, y_mm: np.memmap):
        assert len(x_mm) == len(y_mm), "x と y の長さが一致しません"
        # np.memmap は Pickle 可能なので DataLoader worker に複製しても容量増は僅少
        self.x, self.y = x_mm, y_mm

        # x の dtype が uint8 なら後段で astype 不要
        self._needs_cast = self.x.dtype != np.uint8

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x_np = self.x[idx]
        if self._needs_cast:
            # 型が違う場合だけ 1 回限りのコピー + キャスト
            x_np = x_np.astype(np.uint8, copy=False)

        # torch.from_numpy(...) はゼロコピー
        x = torch.from_numpy(x_np).long()  # (T,) → long
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y
