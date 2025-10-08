# fifty-nlp

- A byte‐level file fragment type classifier based on FiFTy, enhanced with NLP models
- FiFTyをベースに、NLPモデルで強化したバイト単位ファイル断片分類器

## Usage

```
cd fifty-nlp && \
```

## Useage: Google Colab + Google Drive

1. Drive をマウント
   - Colab セルで次を実行
     ```
     from google.colab import drive;
     drive.mount("/content/drive")
     ```
   - OAuth で自身の Drive を接続
2. リポジトリを配置し、依存関係をインストール
 ```
 ! git clone https://github.com/rayfiyo/fifty-nlp.git /content/fifty-nlp && \
  cd /content/fifty-nlp && \
  pip install -r requirements.txt
  ```
3. 設定ファイルを調整
   - `config.yml` の `experiment.result_dir` や `data.base_dir` を
     Drive 上の保存先に書き換える
     - 例: `/content/drive/MyDrive/fifty-nlp/...`
     - マウントした Google Drive のルートディレクトリは `/content/drive/MyDrive/` である
   - `experiment.device` を `gpu` にすると GPU を利用（CPU に戻すときは `cpu`）
   - 学習再開時は `experiment.resume_from` に `result/<timestamp>_<tag>` または
     `result/<timestamp>_<tag>/checkpoint_latest.pt` を指定
4. GPU の有効化
   - ランタイム > ランタイムのタイプを変更 > ハードウェア アクセラレータ > GPU
5. 実行する
  ```
  !cd /content/fifty-nlp && python main.py
   ```

### Tips

- 環境変数を使った一時的な上書き: Colab セルで
  `os.environ["FIFTY_DATA_BASE_DIR"] = "/content/drive/..."` のように指定
- 実行: `!python main.py` などで学習を開始すると、結果・データが Drive 側に保存されます。
- 途中再開: 各エポック終了時に `checkpoint_latest.pt` が更新され、
  同ディレクトリ配下に `checkpoint_epoch_XXX.pt` も保存されます。
  次回は `config.yml` の `experiment.resume_from` に該当パスを設定して再開できます。

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
