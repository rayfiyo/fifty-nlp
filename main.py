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
import numpy as np
import random
import torch
import torch.nn
import os

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
import subprocess
import sys
import yaml
from typing import Any


# config.yml の読み込み設定
PWD = Path(__file__).parent


def load_config() -> dict[str, Any]:
    """config.yml を読み込み、環境変数で上書き。"""

    default_path = PWD / "config.yml"
    config_data = (
        yaml.safe_load(
            default_path.read_text(encoding="utf-8"),
        )
        or {}
    )

    env_base_dir = os.environ.get("FIFTY_DATA_BASE_DIR")
    if env_base_dir:
        config_data.setdefault("data", {})["base_dir"] = env_base_dir

    env_result_dir = os.environ.get("FIFTY_RESULT_DIR")
    if env_result_dir:
        config_data.setdefault("experiment", {})["result_dir"] = env_result_dir

    env_run_type = os.environ.get("FIFTY_RUN_TYPE")
    if env_run_type:
        config_data["type"] = env_run_type

    return config_data


config = load_config()
run_type = config.get("type", "cnn")

# ロギング周りの設定
logger = getLogger(__name__)  # __name__: 個別モジュール用ロガーが得られる
# import しているモジュールも出力されるので DEBUG は注意
log_level = INFO  # ログレベル: DEBUG, INFO, WARNING, ERROR, CRITICAL


def set_seed(seed=42):
    """
    乱数シードを固定し、実験の再現性を確保する関数
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def configure_logging(run_dir: Path) -> None:
    """
    ルートロガーを指定し、標準出力とファイル出力に同じフォーマットで出力する（二重化）。
    一部を標準出力にし、他はデフォルトの標準エラーに出したい場合は、フィルタなどが必要。
    """

    # 1. ルートロガーを log_level 以上にする
    root_logger = getLogger()
    root_logger.setLevel(log_level)

    # 2. 既存のハンドラを削除
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

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
    x_memmap: np.memmap,
    y_memmap: np.memmap,
    batch_size: int,
    device: str = "cpu",
    *,
    max_batches: int | None = None,
) -> tuple[float, int, int]:
    """
    評価用関数：モデルを評価モードに切り替え、
    検証データをバッチ単位で処理して精度 (accuracy) を計算する。

    - memmap を丸ごとスライス → 一度のバッチ Tensor 化
    - GPU/CUDA 時は autocast、CPU でも autocast（PyTorch 2.0+）を利用可能
    """
    model.eval()
    correct = 0
    total = 0

    # 総バッチ数を算出し，max_batches で制限
    total_batches = ceil(len(y_memmap) / batch_size)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    # 推論ループ
    with torch.no_grad():
        for batch_idx in range(total_batches):
            # 入力データのバッチを作成
            start = batch_idx * batch_size
            slab = x_memmap[start : start + batch_size].astype(np.uint8)
            logits = model(torch.from_numpy(slab).long().to(device))

            # バッチごとに正解数を加算（ストリーミング）
            pred_np = torch.argmax(logits, dim=1).cpu().numpy()
            true_np = y_memmap[start : start + batch_size]
            correct += (pred_np == true_np).sum()
            total += true_np.shape[0]

    # ストリーミング集計した正解率を返す
    accuracy = float(correct) / float(total) if total > 0 else 0.0
    return accuracy, total_batches, total


def save_model_visuals(
    model: torch.nn.Module,
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
    seed = config["experiment"]["seed"]
    set_seed(seed)

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

    run_dir = PWD / Path(config["experiment"]["result_dir"]) / f"{ts}_{tag}"
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
    eta_min = tcfg.get("eta_min", 1e-7)
    logger.info(f"eta_min: {eta_min}")

    # 6. 小規模な開発用サブセットを使用
    logger.info(f"n_subset: {n_subset}")
    if n_subset and (n_subset < len(train_y)):
        rng = np.random.default_rng(seed=seed)
        idx = rng.choice(len(train_y), size=n_subset, replace=False)
        # .copy() を末尾につけると RAM に複製される
        # つけないと memmap のままで RAM 消費ゼロ
        train_x = train_x[idx]
        train_y = train_y[idx]
    else:
        pass  # フルデータを memmap のまま利用、n_subset = 0 用

    # 7. モデル構築
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
        eps=1e-7,  # Keras 既定
        amsgrad=False,  # Keras 既定
    )

    # 8. モデルの最適化
    # モデル可視化ファイルをコンパイル
    save_model_visuals(
        model,
        run_dir,
        input_length=train_x.shape[1],
        batch_size=batch_size,
    )
    # PyTorch 2.0 実行最適化
    model = torch.compile(model, mode="reduce-overhead")

    # 9. 学習のための値設定
    total_batches = ceil(len(train_y) / batch_size)  # １エポックあたりの総ステップ数
    progress_step = max(1, total_batches // 10)  # 学習進捗 10% 毎の表示用

    # 10. 学習ループ
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,  # 周期 (= 総エポック数)；Cosine なので 1 期で eta_min まで下がる
        eta_min=eta_min,  # 最終学習率 (最小値)
    )
    for epoch in range(epochs):
        # 学習モード
        model.train()

        # 各エポックのシャッフル順を作成（再現性のために epoch 固有シード）
        rng = np.random.default_rng(seed=seed + epoch)
        order = rng.permutation(len(train_y))  # int64 配列（約 49MB @ 6,144,000 件）

        # バッチループ
        for batch_idx in range(total_batches):
            # 1. インデックス経由でバッチ取得
            start = batch_idx * batch_size
            idx = order[start : start + batch_size]
            inputs = (
                torch.from_numpy(
                    train_x[idx].astype(np.uint8),
                )
                .long()
                .to(device)
            )
            labels = torch.from_numpy(train_y[idx]).long().to(device)

            # 2. 勾配初期化
            optimizer.zero_grad()

            # 3. 自動混合精度 (AMP) での損失計算
            if device == "cuda":
                # GPU なら autocast を利用
                with amp.autocast(device_type="cuda"):
                    outputs = model(inputs)
            else:
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
                    + f"progress {percent}% "
                    + f"(batch {batch_idx + 1}/{total_batches})"
                )

        # 学習率を 1 段階更新
        scheduler.step()

        # 検証精度の評価（スモークテスト用バリデーション）
        if epoch == 0 or (epoch + 1) % 12 == 0:
            val_acc, val_batches, val_samples = eval_model(
                model, val_x, val_y, batch_size, device, max_batches=300
            )
            logger.info(
                f"Epoch {epoch + 1}: Validation:"
                + f" batches={val_batches}, "
                + f"samples={val_samples}, acc={val_acc:.3f}"
            )

        # エポック終了のログ
        logger.info(f"Epoch {epoch + 1}: done!")

    # 12. テストデータでの最終評価
    full_test_acc, full_test_batches, full_test_samples = eval_model(
        model, test_x, test_y, batch_size, device
    )
    logger.info(
        f"Full test accuracy: {full_test_acc:.3f} "
        f"(batches={full_test_batches}, samples={full_test_samples})"
    )


if __name__ == "__main__":
    device_str = config["experiment"]["device"]
    if device_str == "":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        main(device=device_str)
    except (Exception, MemoryError) as err:
        # Python MemoryError / RuntimeError: CUDA OOM などもここでキャッチ
        logger.error(f"Fatal error occurred: {err}", exc_info=True)
        sys.exit(1)
