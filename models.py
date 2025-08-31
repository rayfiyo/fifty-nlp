"""models.py  FiFTy とその派生
- FiFTyModel:       元論文の 1D‑CNN (既存)
- FiFTyLSTMModel:   本実装：LSTM バックボーン版
"""

from __future__ import annotations

import torch
import torch.nn as nn


def init_keras_like_(m: nn.Module) -> None:
    """
    FiFTy で使用している Keras 既定に合わせた初期化:
      - Conv1d/Dense: glorot_uniform (Xavier uniform), bias zeros
      - Embedding: uniform(-0.05, 0.05)
    """
    if isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, a=-0.05, b=0.05)

    elif isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)  # glorot_uniform
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # glorot_uniform
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class FiFTyGRUModel(nn.Module):
    """
    FiFTy 派生 GRU モデル（固定長 512byte）
      - 構造: Embedding → (多層)Bi‑GRU → Dropout → 全結合
      - GRU は LSTM よりパラメータと計算量が約 25 % 少なく高速
    """

    def __init__(
        self,
        n_classes: int,
        *,
        embed_dim: int = 64,
        hidden: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # 1. 埋め込み: バイト列 → 実数ベクトル列
        self.embed = nn.Embedding(256, embed_dim)

        # 2. GRU
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,  # (B, T, E) 形式
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 3. Dropout & 全結合
        self.dropout = nn.Dropout(dropout)
        fc_in = hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_in, n_classes)

    # 順伝播
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """入力 (B, T) → ロジット (B, n_classes)"""
        x = self.embed(x.long())  # (B, T, E)
        _, h_n = self.gru(x)  # h_n: (layers*dir, B, H)

        # 双方向の場合は順方向・逆方向を連結
        if self.gru.bidirectional:
            h = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (B, 2H)
        else:
            h = h_n[-1]  # (B, H)

        h = self.dropout(h)
        return self.fc(h)  # (B, n_classes)


class FiFTyLSTMModel(nn.Module):
    """FiFTy 派生 LSTM モデル（固定長 512byte）

    - 構造: Embedding → (多層)Bi‑LSTM → Dropout → 全結合 → Softmax(logits)
    - 入力:
        - バイト系列テンソル (B, T)
        - B=バッチサイズ
        - T=時系列長 (=512 Byte 固定)
    - 出力: クラス分類ロジット (B, n_classes)
    - パラメータ数は元 CNN と同程度 (~49 万) に抑えてある
    - 時系列長が不変なので pack_padded_sequence は省略
    """

    def __init__(
        self,
        n_classes: int,
        embed_dim: int = 64,
        hidden: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # 1. 埋め込み: バイト列 → 実数ベクトル列
        self.embed = nn.Embedding(
            num_embeddings=256,  # バイト値の取り得る範囲 (0–255)
            embedding_dim=embed_dim,
        )

        # 2. LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,  # Embedding 出力次元
            hidden_size=hidden,  # 隠れ状態のユニット数
            num_layers=num_layers,  # スタック数
            batch_first=True,  # (B, T, E) 形式を前提
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 3. 過学習抑制
        self.dropout = nn.Dropout(dropout)

        # 4. LSTM の最終隠れ状態 h_n のチャネル数を計算
        fc_in = hidden * (2 if bidirectional else 1)

        # 5. 全結合: (B, fc_in) → (B, n_classes)
        self.fc = nn.Linear(fc_in, n_classes)

    # 順伝播
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """
        入力テンソル x をロジットに変換する
        """
        # 1. 埋め込み (B, T) → (B, T, E)
        x = self.embed(x.long())

        # 2. LSTM  (B, T, E) → h_n (layers*dir, B, H)
        _, (h_n, _) = self.lstm(x)

        # 3. 双方向の場合は最終層の順方向 & 逆方向を連結
        if self.lstm.bidirectional:
            # h_n[-2] = forward, h_n[-1] = backward
            h = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (B, 2H)
        else:
            h = h_n[-1]  # (B, H)

        # 4. ドロップアウト → 全結合
        h = self.dropout(h)
        logits = self.fc(h)  # (B, n_classes)
        return logits


class FiFTyModel(nn.Module):
    """
    FiFTy 論文で提案された 1D-CNN 畳み込みニューラルネットワークの簡易実装。
    入力: バイト系列 (整数列)
    出力: クラス分類 (softmax logits)
    """

    def __init__(
        self,
        n_classes: int,
        embed_dim: int,  # 埋め込み次元数
        conv_channels: int,  # 1D畳み込み層の出力チャネル数
        hidden: int,  # 全結合中間層のユニット数
        kernel_size: int,  # 畳み込みカーネル幅
        pool_size: int,  # プーリングサイズ
        dropout: float,  # ドロップアウト率
        num_blocks: int,  # 畳み込みブロック数: Conv → ReLU → Pool を繰り返す回数
    ) -> None:
        super().__init__()

        # バイト（0〜255）を embed_dim 次元へ変換（埋め込み）
        self.embed = nn.Embedding(256, embed_dim)

        # 可変深さの畳み込みブロックを ModuleList で持つ
        self.blocks = nn.ModuleList()
        in_channels = embed_dim
        for _ in range(num_blocks):
            conv = nn.Conv1d(
                in_channels,
                conv_channels,
                kernel_size,
                stride=1,  # FiFTy はstride 1 前提の設計
                padding=0,  # FiFTy の Keras 既定の padding="valid" に合わせる
            )
            self.blocks.append(conv)
            in_channels = conv_channels

        # 活性化関数: FiFTy に基づく LeakyReLU(α=0.3) 。inplace は微小な高速化とメモリ節約
        self.activation = nn.LeakyReLU(0.3, inplace=True)

        # 時系列長を pool_size 分縮小
        self.pool = nn.MaxPool1d(pool_size)

        # 時系列長を 1 に平均化(AP)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # 過学習抑制のためのノイズ挿入
        self.dropout = nn.Dropout(dropout)

        # 全結合層: 入力チャネル数（今回は畳み込みブロック後のチャネル数 conv_channels）を
        # hidden 次元の特徴ベクトルに写像する全結合層
        self.fc1 = nn.Linear(conv_channels, hidden)

        # 出力層: hidden 次元ベクトルを最終的に
        #         n_classes クラス分のロジット（未正規化確率）に写像する全結合層
        self.fc2 = nn.Linear(hidden, n_classes)

        # まとめて Keras 既定に近い初期化を適用
        self.apply(init_keras_like_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播処理 (B, T) → (B, C)"""
        # torchview から渡されるダミー入力が float の場合もあるので long にキャスト
        x = x.long()
        x = self.embed(x)  # (B, T, E): 埋め込み
        x = x.permute(0, 2, 1)  # Conv1d 用に次元を入れ替え (B, E, T)
        # ブロック数だけ繰り返す
        for conv in self.blocks:
            x = conv(x)
            x = self.activation(x)  # LeakyReLU(0.3)
            x = self.pool(x)
        x = self.gap(x).squeeze(-1)  # (B, C): 特徴ベクトル。APに対応。
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.activation(x)  # LeakyReLU(0.3)
        return self.fc2(x)  # (B, クラス数 75) のロジット出力
