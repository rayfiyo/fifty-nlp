# fifty-nlp

- A byte‐level file fragment type classifier based on FiFTy, enhanced with NLP models
- FiFTyをベースに、NLPモデルで強化したバイト単位ファイル断片分類器

## Usage

```
cd fifty-nlp && \
```

## Useage: Google Colab + Google Drive

- Drive をマウント
  - Colab セルで `from google.colab import drive; drive.mount("/content/drive")` を実行
  - OAuth で自身の Drive を接続
- リポジトリを配置し、依存関係をインストール
  ```
  ! git clone https://github.com/rayfiyo/fifty-nlp.git /content/fifty-nlp && \
  ! pip install -r requirements.txt
  ```
- 個人設定ファイルを作成
  - `config.local.example.yml` を `config.local.yml` にコピー
  - `experiment.result_dir` や `data.base_dir` を Drive 上の保存先に書き換える
    - 例: `/content/drive/MyDrive/fifty-nlp/...`

### Tips

- `config.local.yml` は `.gitignore` 済み
- 環境変数を使った一時的な上書き: Colab セルで
  `os.environ["FIFTY_DATA_BASE_DIR"] = "/content/drive/..."` のように指定
- 実行: `!python main.py` などで学習を開始すると、結果・データが Drive 側に保存されます。

# Special thanks

## FiFTy

- https://arxiv.org/abs/1908.06148v2
- https://arxiv.org/pdf/1908.06148v2
- https://www.alphaxiv.org/overview/1908.06148v2
- https://github.com/mittalgovind/fifty

# Old information

## lstm-file-classifier

- LSTMによるファイル分類器
- File Classification using LSTM
