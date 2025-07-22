"""main.py FiFTy 学習スクリプト
FiFty の 512-byte Scenario #1 (All Classes 75) の最終モデル TABLE II を参考
https://arxiv.org/abs/1908.06148v2
https://arxiv.org/pdf/1908.06148v2
https://www.alphaxiv.org/overview/1908.06148v2
https://github.com/mittalgovind/fifty
===============================

result/
└── YYYY‑MM‑DD_HH-MM-SS_<tag>/
        ├── log.txt         : 実行ログ（stdout）
        ├── torchview.svg   : レイヤー構造図（画像）
        └── torchinfo.txt   : 形状・パラメータ統計（プレーンテキスト）

PEP8 ＆ 日本語コメ
"""

from __future__ import annotations


# 機械学習に関与
from models import FiFTyModel, FiFTyLSTMModel, FiFTyGRUModel
from torch import amp
from torch.utils.data import Dataset, DataLoader
import intel_extension_for_pytorch as ipex
import numpy as np
import torch
import torch.nn as nn

# ログなどの表示調整に関与
from logging import (
    getLogger,
    FileHandler,
    StreamHandler,
    Formatter,
    INFO,
)
from math import ceil
from pathlib import Path
from torchinfo import summary
from torchview import draw_graph
import datetime as _dt
import os
import subprocess
import sys
import yaml


#  oneDNN（AutocastCPU） の C++ 層からの警告を抑制する
os.environ["DNNL_VERBOSE"] = "0"  # oneDNN の verbose ログを完全抑制
os.environ["IPEX_LOG_LEVEL"] = "ERROR"  # IPEX Python ロガーを ERROR 以上に

# config.yml の読み込み設定
PWD = Path(__file__).parent
config = yaml.safe_load((PWD / "config.yml").read_text(encoding="utf-8"))
run_type = config.get("type", "cnn")

# ロギング周りの設定
logger = getLogger(__name__)  # __name__: 個別モジュール用ロガーが得られる
logger.propagate = False  # ルートにバブリングさせない


def load_memmap(split: str) -> tuple[np.memmap, np.memmap]:
    """
    指定したデータセット(split)に対応する .npy ファイルを
    メモリマップとして読み込み、メモリ使用量を抑制する関数
    """
    base_dir = Path(config["data"]["base_dir"]).expanduser()  # ~ の展開
    base = base_dir / split

    # 遅延ロードで RAM 最小化
    x = np.load(base / "x.npy", mmap_mode="r")
    y = np.load(base / "y.npy", mmap_mode="r")
    return x, y


class MemmapDataset(Dataset):
    """
    NumPy memmap を直接読む Dataset（バイト列分類用）
    """

    # np.memmap を保持し、複製による RAM 消費を避ける
    def __init__(self, x_memmap, y_memmap):
        self.x, self.y = x_memmap, y_memmap

    # サンプル総数を返す（DataLoader がバッチ分割に利用）
    def __len__(self):
        return len(self.y)

    # idx 番目のサンプルを (Tensor, int) で返す
    def __getitem__(self, idx):
        # - x: 1D バイト系列 → long Tensor
        x_tensor = torch.from_numpy(self.x[idx]).long()

        # - y: 正解ラベル (int)
        y_scalar = int(self.y[idx])

        return x_tensor, y_scalar


def configure_logging(run_dir: Path) -> None:
    """
    ルートロガーを指定し、標準出力とファイル出力に同じフォーマットで出力する（二重化）。
    一部を標準出力にし、他はデフォルトの標準エラーに出したい場合は、フィルタなどが必要。
    """
    # 1. ログレベルの設定: DEBUG, INFO, WARNING, ERROR, CRITICAL がある
    # import しているモジュールも出力されるので DEBUG は注意
    log_level = INFO

    # 2. ルートロガーを log_level 以上にする
    root_logger = getLogger()
    root_logger.setLevel(log_level)

    # 3. フォーマッタ定義
    fmt = "%(asctime)s %(name)s %(levelname)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = Formatter(fmt, datefmt=datefmt)

    # 4. ファイルハンドラ: log.txt へ log_level 以上を
    fh = FileHandler(run_dir / "log.txt", encoding="utf-8")
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)

    # 5. コンソールハンドラ: 標準出力へ log_level 以上を
    ch = StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)


def eval_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> tuple[float, int, int]:
    """
    評価用関数：モデルを評価モードに切り替え、
    検証データをバッチ単位で処理して精度 (accuracy) を計算する。

    Dropout や BatchNorm を無効化するとOOMエラーになる。 # preds, trues = [], []
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad(), (
        torch.cuda.amp.autocast()
        if device.startswith("cuda")
        else torch.cpu.amp.autocast()
    ):
        for inputs, true_y in loader:
            # GPU利用時は non_blocking=True で転送を非同期化できる
            inputs = inputs.to(device, non_blocking=False)
            logits = model(inputs)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            true_np = true_y.numpy()
            correct += (pred == true_np).sum()
            total += len(true_np)

    # ストリーミング集計した正解率を返す
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, len(loader), total


def save_model_visuals(
    model: nn.Module,
    run_dir: Path,
    input_length: int,
    batch_size: int,
) -> None:
    """
    可視化ユーティリティ
    torchview・torchinfo の結果を run_dir に保存する。
    """
    # torchview: SVG 形式でレイヤー構造を保存
    graph = draw_graph(
        model,
        input_size=(1, input_length),  # バッチ 1 で OK（描画用）
        expand_nested=True,
        save_graph=False,
    )
    graph.visual_graph.render(
        filename=run_dir / "torchview", format="svg", cleanup=True
    )

    # torchinfo: 形状とパラメータ数の表をプレーンテキストで保存
    info = summary(
        model,
        input_size=(batch_size, input_length),
        col_names=("input_size", "output_size", "num_params"),
        verbose=0,
    )
    (run_dir / "torchinfo.txt").write_text(str(info), encoding="utf-8")


def main(device: str = "cpu") -> None:  # noqa: C901 (関数長は許容)
    """
    メイン関数:
    - データ読み込み
    - モデル初期化
    - 学習ループ
    - 評価
    - 可視化ファイル保存とログ出力
    """

    # 1. タグのデフォルト値として直近コミットの先頭7桁を取得
    try:
        repo_dir = Path(__file__).parent
        default_tag = (
            subprocess.check_output(
                ["git", "rev-parse", "--short=7", "HEAD"], cwd=repo_dir
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        default_tag = ""

    # 2. 実行用ディレクトリを作成
    ts = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tag = sys.argv[1] if len(sys.argv) > 1 else default_tag

    run_dir = PWD / Path(config["output"]["result_dir"]) / f"{ts}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 3. ログ二重化
    configure_logging(run_dir)

    # 4. データ読み込み
    splits = config["data"]["splits"]
    train_x, train_y = load_memmap(splits[0])
    val_x, val_y = load_memmap(splits[1])
    test_x, test_y = load_memmap(splits[2])

    # 5. ハイパーパラメータ定義
    # モデル設定を切り替え
    mcfg = config["model"][run_type]
    # 学習設定を切り替え
    tcfg = config["training"][run_type]
    #
    batch_size = tcfg["batch_size"]
    logger.info(f"batch_size: {batch_size}")
    #
    epochs = tcfg["epochs"]
    logger.info(f"epochs: {epochs}")
    #
    lr = tcfg["lr"]
    logger.info(f"lr: {lr}")
    #
    n_classes = int(train_y.max()) + 1  # クラスへの出力の最終全結合 F(75) を動的に
    logger.info(f"n_classes: {n_classes}")
    #
    n_subset = tcfg["n_subset"]
    logger.info(f"n_subset: {n_subset}")
    #
    eta_min = tcfg.get("eta_min", 1e-5)
    logger.info(f"eta_min: {eta_min}")
    #
    num_workers = tcfg.get("num_workers", os.cpu_count() or 1)
    logger.info(f"num_workers: {num_workers}")

    # 6. 小規模な開発用サブセットを使用
    if len(train_y) > n_subset:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(train_y), size=n_subset, replace=False)
        # .copy() を末尾につけると RAM に複製される
        # つけないと memmap のままインデックスなので RAM 消費ゼロ
        train_x = train_x[idx]
        train_y = train_y[idx]

    # 7. DataLoader 生成
    pin_memory = device != "cpu"  # ピンメモリは GPU 時に有効化
    if len(train_y) > n_subset:
        subset_idx = np.random.default_rng(42).choice(
            len(train_y), n_subset, replace=False
        )
        sampler = torch.utils.data.SubsetRandomSampler(subset_idx)
    else:
        sampler = None
    train_dl = DataLoader(
        MemmapDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=(sampler is None),  # データ順序をエポック毎にシャッフル
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dl = DataLoader(
        MemmapDataset(val_x, val_y),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_dl = DataLoader(
        MemmapDataset(test_x, test_y),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # 8. モデル構築
    if run_type == "gru":
        model = FiFTyGRUModel(
            n_classes=n_classes,
            embed_dim=mcfg["embed_dim"],
            hidden=mcfg["hidden_dim"],
            num_layers=mcfg["num_layers"],
            bidirectional=mcfg["bidirectional"],
            dropout=mcfg["dropout"],
        ).to(device)
    elif run_type == "lstm":
        model = FiFTyLSTMModel(
            n_classes=n_classes,
            embed_dim=mcfg["embed_dim"],
            hidden=mcfg["hidden_dim"],
            num_layers=mcfg["num_layers"],
            bidirectional=mcfg["bidirectional"],
            dropout=mcfg["dropout"],
        ).to(device)
    elif run_type == "cnn":
        model = FiFTyModel(
            n_classes=n_classes,
            embed_dim=mcfg["embed_dim"],
            conv_channels=mcfg["conv_channels"],
            hidden=mcfg["hidden_dim"],
            kernel_size=mcfg["kernel_size"],
            pool_size=mcfg["pool_size"],
            dropout=mcfg["dropout"],
            num_blocks=mcfg.get("num_blocks", 2),
        ).to(device)
    # 確率的勾配降下法
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
    )

    # 9. モデルの最適化
    # モデル可視化ファイルをコンパイル
    save_model_visuals(
        model,
        run_dir,
        input_length=train_x.shape[1],
        batch_size=batch_size,
    )

    if device == "cpu":
        model, optimizer = ipex.optimize(model, optimizer=optimizer, level="O1")
        torch.set_float32_matmul_precision("medium")  # oneDNN 最適化
    else:
        # GPU: IPEX不要
        torch.backends.cuda.matmul.allow_tf32 = True
        # PyTorch 2.0 実行最適化
        model = torch.compile(model, mode="reduce-overhead")

    # 10. 学習のための値設定
    total_batches = ceil(len(train_dl) / batch_size)  # 総バッチ数（切り上げ）
    progress_step = max(1, total_batches // 10)  # 10% ごとに進捗表示

    # 11. 学習ループ
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,  # 周期 (= 総エポック数)；Cosine なので 1 期で eta_min まで下がる
        eta_min=eta_min,  # 最終学習率 (最小値)
    )

    # 学習ループ内の AMP
    print()
    for epoch in range(epochs):
        # 学習モード
        model.train()

        # バッチループ
        for batch_idx, (inputs, labels) in enumerate(train_dl):
            # 1. バッチ取り出し（memmap → Tensor、入力テンソルとラベルの作成）
            #   non_blocking=True で転送を非同期化（GPU 利用時向け）
            inputs = inputs.to(device, non_blocking=False)
            labels = labels.to(device, non_blocking=False)

            # 2. 勾配初期化
            optimizer.zero_grad()

            # 3. 自動混合精度 (AMP) での損失計算
            # CPU／GPU に合わせて autocast を適用
            with amp.autocast(
                device_type=("cuda" if device == "cuda" else "cpu"),
                dtype=(None if device == "cuda" else torch.bfloat16),
            ):
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)

            # 4. 後処理
            loss.backward()  # 逆伝播（勾配計算）
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=0.5
            )  # 勾配クリッピング
            optimizer.step()  # パラメータ更新

            # 最初のエポックだけ 10 % ごとに進捗表示
            if epoch == 0 and batch_idx % progress_step == 0:
                percent = int(batch_idx / total_batches * 100)
                logger.info(
                    f"Epoch {epoch + 1}:"
                    + f"progress {percent}% (batch {batch_idx + 1}/{total_batches})"
                )

        # 学習率を 1 段階更新
        scheduler.step()

        # 検証精度の評価（スモークテスト用バリデーション）
        if epoch == 0 or (epoch + 1) % 6 == 0:
            val_acc, val_batches, val_samples = eval_model(model, val_dl, device)
            logger.info(
                f"Epoch {epoch + 1}: Validation:"
                + f" batches={val_batches}, samples={val_samples}, acc={val_acc:.3f}"
            )

        # エポック終了のログ
        logger.info(f"Epoch {epoch + 1}: done!")

    # 12. テストデータでの最終評価
    full_test_acc, full_test_batches, full_test_samples = eval_model(
        model, test_dl, device
    )
    logger.info(
        f"Full test accuracy: {full_test_acc:.3f} "
        f"(batches={full_test_batches}, samples={full_test_samples})"
    )


if __name__ == "__main__":
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        main(device=device_str)
    except (Exception, MemoryError) as err:
        # Python MemoryError / RuntimeError: CUDA OOM などもここでキャッチ
        logger.error(f"Fatal error occurred: {err}", exc_info=True)
        sys.exit(1)
