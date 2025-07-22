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
import main  # load_memmap, FiFTyModel, eval_model を含むスクリプトを指す


def create_dataset(
    base_dir: Path, split: str, x_array: np.ndarray, y_array: np.ndarray
) -> None:
    """
    ヘルパ関数：指定ディレクトリ直下に split/x.npy と split/y.npy を生成
    """
    split_dir = base_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "x.npy", x_array)
    np.save(split_dir / "y.npy", y_array)


@pytest.fixture(autouse=True)
def patch_base_dir(tmp_path: Path, monkeypatch):
    """
    各テストで config['data']['base_dir'] を tmp_path に差し替え
    """
    # main.config["data"]["base_dir"] は文字列か Path を受け入れる想定
    monkeypatch.setitem(main.config["data"], "base_dir", str(tmp_path))
    return tmp_path


def test_load_memmap_returns_memmap_and_correct_shape(tmp_path: Path):
    """
    load_memmap が np.memmap を返し、保存時の形状を維持していることを検証
    """
    # ダミーデータ作成 (4サンプル × 時系列長5)
    x_dummy = np.arange(4 * 5, dtype=np.uint8).reshape(4, 5)
    y_dummy = np.array([0, 1, 0, 1], dtype=np.uint8)
    create_dataset(tmp_path, "train", x_dummy, y_dummy)

    x_memmap, y_memmap = main.load_memmap("train")

    # np.memmap 型であること
    assert isinstance(x_memmap, np.memmap)
    assert isinstance(y_memmap, np.memmap)

    # 形状が正しいこと
    assert x_memmap.shape == (4, 5)
    assert y_memmap.shape == (4,)


def test_fiftymodel_forward_keeps_shape():
    """
    FiFTyModel の順伝播が (B, T) → (B, C) の形状を維持することを検証
    """
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
    # ランダム整数列を入力 (0〜255)
    input_tensor = torch.randint(0, 256, (B, T), dtype=torch.int64)
    output = model(input_tensor)

    # 出力の型と形状をチェック
    assert isinstance(output, torch.Tensor)
    assert output.shape == (B, C)


def test_eval_model_computes_accuracy(tmp_path: Path):
    """
    eval_model が予測結果と真値を突き合わせて正しい精度を返すことを検証
    """
    # ダミーデータ作成 (4サンプル × 時系列長1)
    x_dummy = np.zeros((4, 1), dtype=np.uint8)
    # 真値は [0,1,0,1] とする
    y_dummy = np.array([0, 1, 0, 1], dtype=np.uint8)
    split_dir = tmp_path / "dummy"
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "x.npy", x_dummy)
    np.save(split_dir / "y.npy", y_dummy)

    # メモリマップとしてロード
    x_memmap = np.load(split_dir / "x.npy", mmap_mode="r")
    y_memmap = np.load(split_dir / "y.npy", mmap_mode="r")

    # 常にクラス0を予測するダミーモデル
    class DummyModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch = x.shape[0]
            # 2クラス分のロジットをゼロで返し、argmax=0に固定
            return torch.zeros(batch, 2)

    dummy = DummyModel()
    accuracy, batches_run, total_samples = main.eval_model(
        dummy, x_memmap, y_memmap, batch_size=4, device="cpu"
    )

    # バッチ数とサンプル数も期待どおりかチェックしておくと安心
    assert batches_run == 1  # 4サンプル／バッチ4 → バッチ数は1
    assert total_samples == 4  # 総サンプル数は4

    # 正解ラベルは 0 が2つ、1 が2つ → 精度は 2/4 = 0.5
    assert pytest.approx(accuracy, rel=1e-6) == 0.5
