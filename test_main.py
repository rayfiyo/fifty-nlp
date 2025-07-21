"""tests/test_main.py FiFTy 学習スクリプトのユニットテスト
==========================================================

目的
----

- `load_memmap` が期待どおり `np.memmap` を返すかどうか
- `FiFTyModel` の順伝播が (B, T) → (B, C) の形状を維持するか
- `eval_model` が推論結果と真値を突き合わせて正しい精度を返すか

すべて最小サイズのダミーデータを tmp_path 配下に生成して実行する。
PEP8 準拠日本語コメント。
"""

import numpy as np
import torch
import pytest
from pathlib import Path
from torch.utils.data import DataLoader

import main  # load_memmap, FiFTyModel, eval_model_dl, MemmapDataset を含む


# ヘルパ関数／フィクスチャ
def _create_dataset(
    base_dir: Path, split: str, x_array: np.ndarray, y_array: np.ndarray
) -> None:
    """``base_dir/split`` 直下に ``x.npy`` と ``y.npy`` を生成する。"""
    split_dir = base_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "x.npy", x_array)
    np.save(split_dir / "y.npy", y_array)


@pytest.fixture(autouse=True)
def patch_base_dir(tmp_path: Path, monkeypatch):
    """各テストで ``config['data']['base_dir']`` を一時ディレクトリに差し替え。"""
    monkeypatch.setitem(main.config["data"], "base_dir", str(tmp_path))
    return tmp_path


# テスト１：load_memmap
def test_load_memmap_returns_memmap_and_correct_shape(tmp_path: Path):
    """``load_memmap`` が memmap を返し、形状を保持することを検証。"""
    x_dummy = np.arange(20, dtype=np.uint8).reshape(4, 5)  # 4×5
    y_dummy = np.array([0, 1, 0, 1], dtype=np.uint8)
    _create_dataset(tmp_path, "train", x_dummy, y_dummy)

    x_mm, y_mm = main.load_memmap("train")

    assert isinstance(x_mm, np.memmap)
    assert isinstance(y_mm, np.memmap)
    assert x_mm.shape == (4, 5)
    assert y_mm.shape == (4,)


# テスト２：FiFTyModel forward
def test_fiftymodel_forward_keeps_shape():
    """``FiFTyModel`` の順伝播が (B, T) → (B, C) を維持することを検証。"""
    B, T, C = 3, 10, 5
    model = main.FiFTyModel(
        n_classes=C,
        embed_dim=8,
        conv_channels=16,
        hidden=32,
        kernel_size=3,
        pool_size=2,
        dropout=0.0,
        num_blocks=2,
    )
    input_tensor = torch.randint(0, 256, (B, T), dtype=torch.int64)
    output = model(input_tensor)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (B, C)


# テスト３：eval_model_dl
def test_eval_model_dl_computes_accuracy(tmp_path: Path):
    """``eval_model_dl`` が正しい精度を返すことを検証。"""
    # ダミーデータ（4 サンプル、時系列長 1）
    x_dummy = np.zeros((4, 1), dtype=np.uint8)
    y_dummy = np.array([0, 1, 0, 1], dtype=np.uint8)
    _create_dataset(tmp_path, "dummy", x_dummy, y_dummy)

    # memmap → Dataset → DataLoader
    x_mm, y_mm = main.load_memmap("dummy")
    dataset = main.MemmapDataset(x_mm, y_mm)
    loader = DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=False
    )

    # 常にクラス 0 を出力するダミーモデル
    class DummyModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
            return torch.zeros(x.size(0), 2)

    dummy = DummyModel()
    accuracy = main.eval_model_dl(dummy, loader, device="cpu")

    # 真値は [0,1,0,1]、常に 0 を予測 → 精度は 0.5
    assert pytest.approx(accuracy, rel=1e-6) == 0.5
