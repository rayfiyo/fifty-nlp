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
from models import FiFTyModel, FiFTyLSTMModel  # models.py
from torchinfo import summary  # 出力形状・パラメータ統計
from torchview import draw_graph  # レイヤー構造図の可視化
import numpy as np  # メモリマップや数値演算に使用
import torch  # テンソル操作と計算グラフに使用
import torch.nn as nn  # ニューラルネットワークモジュールを nn として利用

# ログなどの表示調整に関与
from math import ceil  # 進捗割合の計算に使用
from pathlib import Path  # パス操作に使用
from logging import (  # 学習結果にタイムスタンプがほしいのでロガーを使用
    getLogger,
    FileHandler,
    StreamHandler,
    Formatter,
    INFO,
)
import datetime as _dt  # タイムスタンプ生成に使用
import subprocess  # gitハッシュ取得に使用
import sys  # コマンドライン引数・stdout 差し替えに使用
import yaml  # 設定などを書いた config.yml を読み込むのに使用


# config.yml の読み込み設定
PWD = Path(__file__).parent
config = yaml.safe_load((PWD / "config.yml").read_text(encoding="utf-8"))

# __name__ をキーにすると、個別モジュール用ロガーが得られる
logger = getLogger(__name__)


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
    model: torch.nn.Module,  # 評価対象モデル
    x_memmap: np.memmap,  # 評価データ
    y_memmap: np.memmap,  # 評価データ
    batch_size: int,  # バッチサイズ
    device: str = "cpu",  # 'cpu' or 'cuda'
    *,  # 以降は任意、キー=値 の形式で与える必要通用がある
    max_batches: int | None = None,  # 最大評価バッチ数で。None なら全バッチ
    subset_ratio: float | None = None,  # 評価データの割合 [0.0～1.0]。None なら全データ
) -> tuple[float, int, int]:
    """
    評価用関数：モデルを評価モードに切り替え、
    検証データをバッチ単位で処理して精度 (accuracy) を計算する。

    スモークテスト用の max_batches と subset_ratio は併用可能。
    両方指定時は先に subset_ratio で絞り、さらに max_batches で上限を掛ける。
    例: 動作確認＋大きなバグ検出のため、最初の100バッチだけ評価
    max_batches=100
    例: 学習曲線の山谷をざっくり把握のため、検証データの10%だけを全バッチ評価
    subset_ratio=0.1

    Dropout や BatchNorm を無効化するとOOMエラーになる。 # preds, trues = [], []
    """
    model.eval()
    correct = 0
    total = 0

    # データ量を subset_ratio で絞る
    if subset_ratio is not None:
        N = int(len(y_memmap) * subset_ratio)
        x_memmap = x_memmap[:N]
        y_memmap = y_memmap[:N]

    # 総バッチ数を算出し、max_batches で制限
    total_batches = ceil(len(y_memmap) / batch_size)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    # 勾配計算を無効化
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
    logger.info(f"output duplicated to {run_dir / 'log.txt'}")

    # 4. データ読み込み
    splits = config["data"]["splits"]
    train_x, train_y = load_memmap(splits[0])
    val_x, val_y = load_memmap(splits[1])
    test_x, test_y = load_memmap(splits[2])

    # 5. ハイパーパラメータ定義
    tcfg = config["training"][config["training"].get("type", "cnn")]

    batch_size = tcfg["batch_size"]
    logger.info(f"batch_size: {batch_size}")

    epochs = tcfg["epochs"]
    logger.info(f"epochs: {epochs}")

    lr = tcfg["lr"]
    logger.info(f"lr: {lr}")

    n_classes = int(train_y.max()) + 1  # クラスへの出力の最終全結合 F(75) を動的に
    logger.info(f"n_classes: {n_classes}")

    n_subset = tcfg["n_subset"]
    logger.info(f"n_subset: {n_subset}")

    # 6. 小規模な開発用サブセットを使用
    if len(train_y) > n_subset:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(train_y), size=n_subset, replace=False)
        # .copy() を末尾につけると RAM に複製される
        # つけないと memmap のままで RAM 消費ゼロ
        train_x = train_x[idx]
        train_y = train_y[idx]

    # 7. モデル構築
    mcfg_root = config["model"]
    mtype = mcfg_root.get("type", "cnn")
    mcfg = mcfg_root[mtype]
    if mtype == "lstm":
        model = FiFTyLSTMModel(
            n_classes=n_classes,
            embed_dim=mcfg["embed_dim"],
            hidden=mcfg["hidden_dim"],
            num_layers=mcfg["num_layers"],
            bidirectional=mcfg["bidirectional"],
            dropout=mcfg["dropout"],
        ).to(device)
    elif mtype == "cnn":
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

    # 8. モデル可視化ファイルをコンパイルと学習より前に行い、学習前の形状を記録
    save_model_visuals(
        model,
        run_dir,
        input_length=train_x.shape[1],
        batch_size=batch_size,
    )

    # 9. PyTorch 2.0 実行最適化
    model = torch.compile(model, mode="reduce-overhead")

    # 10. 学習のための値設定
    total_batches = ceil(len(train_y) / batch_size)  # 総バッチ数（切り上げ）
    progress_step = max(1, total_batches // 10)  # 10% ごとに進捗表示

    # 11. 学習ループ
    for epoch in range(epochs):
        model.train()  # 学習モード
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size

            # 入力テンソルとラベルの作成
            slab_x = (
                torch.from_numpy(train_x[start : start + batch_size].astype(np.uint8))
                .long()
                .to(device)
            )
            labels = (
                torch.from_numpy(train_y[start : start + batch_size].copy())
                .long()
                .to(device)
            )

            # 学習ステップ
            optimizer.zero_grad()

            # 損失計算
            loss = nn.functional.cross_entropy(model(slab_x).float(), labels)

            # 勾配計算と勾配爆発対策
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # パラメータ更新
            optimizer.step()

            # 学習スケジューラ
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-5
            )
            scheduler.step()

            # 学習進捗の出力（最初の Epoch のみ）
            if epoch == 0 and batch_idx % progress_step == 0:
                percent = int(batch_idx / total_batches * 100)
                logger.info(
                    f"Epoch {epoch + 1}:"
                    + f"progress {percent}% (batch {batch_idx + 1}/{total_batches})"
                )

        # 検証精度の評価（スモークテスト）
        if epoch == 0 or (epoch + 1) % 6 == 0:
            val_acc, val_batches, val_samples = eval_model(
                model, val_x, val_y, batch_size, device, max_batches=300
            )
            logger.info(
                f"Epoch {epoch + 1}: Validation:"
                + f" batches={val_batches}, samples={val_samples}, acc={val_acc:.3f}"
            )

        # エポック終了のログ
        logger.info(f"Epoch {epoch + 1}: done!")

    # 12. テストデータでの最終評価
    # タプルをアンパックして、accuracy（float）のみ取り出す
    full_test_acc, full_test_batches, full_test_samples = eval_model(
        model, test_x, test_y, batch_size, device
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
