<!--
    README
 -->

# Pytorch Template

[![English](https://img.shields.io/badge/English-018EF5.svg?labelColor=d3d3d3&logo=readme)](./README.md)
[![Japanese](https://img.shields.io/badge/Japanese-018EF5.svg?labelColor=d3d3d3&logo=readme)](./README_JA.md)
[![license](https://img.shields.io/github/license/r-dev95/pytorch-template)](./LICENSE)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

[![Python](https://img.shields.io/badge/Python-3776AB.svg?labelColor=d3d3d3&logo=python)](https://github.com/python)
[![Sphinx](https://img.shields.io/badge/Sphinx-000000.svg?labelColor=d3d3d3&logo=sphinx&logoColor=000000)](https://github.com/sphinx-doc/sphinx)
[![Pytest](https://img.shields.io/badge/Pytest-0A9EDC.svg?labelColor=d3d3d3&logo=pytest)](https://github.com/pytest-dev/pytest)
[![Pydantic](https://img.shields.io/badge/Pydantic-ff0055.svg?labelColor=d3d3d3&logo=pydantic&logoColor=ff0055)](https://github.com/pydantic/pydantic)
[![Numpy](https://img.shields.io/badge/Numpy-blue.svg?labelColor=d3d3d3&logo=numpy&logoColor=blue)](https://github.com/numpy/numpy)
[![Pandas](https://img.shields.io/badge/Pandas-150458.svg?labelColor=d3d3d3&logo=pandas&logoColor=150458)](https://github.com/pandas-dev/pandas)
[![Pytorch](https://img.shields.io/badge/Pytorch-F05F42.svg?labelColor=d3d3d3&logo=pytorch)](https://github.com/pytorch/pytorch)
[![Lightning](https://img.shields.io/badge/Lightning-5500ff.svg?labelColor=d3d3d3&logo=lightning&logoColor=5500ff)](https://github.com/Lightning-AI/pytorch-lightning)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?labelColor=d3d3d3&logo=scikit-learn)](https://github.com/scikit-learn/scikit-learn)

本リポジトリでは、pytorch のテンプレートを定義しています。

パラメータファイルを使用して次の項目を設定することで、様々な組み合わせで学習と評価ができます。
また他の設定をそれぞれの設定クラスに実装することで簡単に使用できます。

- データ (とデータ前処理)
- モデル (とモデルレイヤー)
- 最適化手法
- 誤差関数
- メトリクス
- コールバック

実装されているソースは、Sphinx を使用してドキュメントを作成しています。

本リポジトリをクローンして、ローカルで確認してみてください。

- 英語:

  ```bash
  cd pytorch-templete/docs
  make html -e SPHINXOPTS='-a -E -D language="en"'
  ```

- 日本語:

  ```bash
  cd pytorch-templete/docs
  make html -e SPHINXOPTS='-a -E -D language="ja"'
  ```

## Getting started

### 1. github からインストール

```bash
git clone https://github.com/r-dev95/pythontemplate.git
```

### 2. 仮想環境の構築

`poetry`がインストールされていることが前提です。

python の開発環境がまだ整っていない方は、[こちら](#開発環境の構築)。

```bash
cd pythontemplate/pytorch_template
poetry install
```

### 3. データのダウンロードと shard データ(webdataset)の作成

```bash
poetry shell
python dataset.py --result dataset --data mnist
```

### 4. モデルの学習

```bash
python train.py --param param/tutorial/param_train.yaml
```

### 5. モデルの評価

```bash
python eval.py --param param/tutorial/param_eval.yaml
```

## パラメータファイルの使い方

パラメータファイル(`.yaml`)の使い方について説明します。

パラメータファイルは次のソースコードで使用します。
次のソースコードは一部コマンドライン引数を使用できますが、パラメータファイルで上書きできるため、パラメータファイルにすべてのパラメータを設定する前提です。

- train.py
- eval.py

一部設定はパラメータファイルを使用して設定できません。特に pytorch (keras)の詳細な設定は公式サイトを参照し実装する必要があります。

### 主要なパラメータの設定

コマンドライン引数として実装されている主要なパラメータは、ゼロインデントで設定します。

- 主要なパラメータには`param`を含みますが、コマンドライン引数としてのみ機能するため設定しません。

`train.py`と`eval.py`の共通設定の例:

```yaml
# log handler (idx=0: stream handler, idx=1: file handler)
# (True: set handler, False: not set handler)
# type: list[bool, bool]
handler: [True, True]
# log level (idx=0: stream handler, idx=1: file handler)
# (DEBUG: 10, INFO: 20, WARNING: 30, ERROR: 40, CRITICAL: 50)
# type: list[int, int]
level: [10, 10]
# random seed
# type: int
seed: 0
# directory path (data save)
# type: str
result: result
# number of workers (data loader)
# type: int
num_workers: 4
```

`train.py`のみの設定の例:

```yaml
# directory path (training data)
# type: str
train: data/mnist/train
# directory path (validation data)
# type: str | None
valid: data/mnist/test
# batch size (training data)
# type: int
train_batch: 32
# batch size (validation data)
# type: int | None
valid_batch: 1000
# shuffle size
# type: int | None
shuffle: null
# Number of epochs
# type: int
epochs: 2
```

`eval.py`のみの設定の例:

```yaml
# directory path (evaluation data)
# type: str
eval: data/mnist/test
# batch size (training data)
# type: int
batch: 1000
```

### `データ`と`データ前処理`のパラメータの設定

現在使用可能な`データ`は、`SetupData`クラスの`func`のキーを[参照](pytorch_template/lib/data/setup.py)。

`データ`の設定の例:

```yaml
data:
  kind: mnist
```

現在使用可能な`データ前処理`は、`Processor`クラスの`func`のキーを[参照](pytorch_template/lib/data/processor.py)。

- `データ前処理`の`kind`は、リストとして設定する。

- `kind`に`one_hot`を設定する場合、`one_hot`の設定は次の例のように設定する。
  以降のパラメータについても同様である。

`データ前処理`の設定の例:

```yaml
process:
  kind: [one_hot, rescale]

  one_hot:
    num_classes: &num_classes 10

  rescale:
    scale: 0.00392156862745098
    offset: 0
```

### `モデル`と`モデルレイヤー`のパラメータの設定

現在使用可能な`モデル`は、`SetupModel`クラスの`func`のキーを[参照](pytorch_template/lib/model/setup.py)。

`モデル`の設定の例:

```yaml
model:
  kind: simple
```

現在使用可能な`モデルレイヤー`は、`SetupLayer`クラスの`func`のキーを[参照](pytorch_template/lib/model/layer.py)。

- `モデルレイヤー`の`kind`は、リストとして設定する。

- `kind`の値は、末尾に"\_" + 英数字をつけられる。

`モデルレイヤー`の設定の例:

```yaml
layer:
  kind: [flatten, linear_1, relu, linear_2]

  flatten:
    start_dim: 1
    end_dim: -1

  LINEAR: &LINEAR
    in_features: 0
    out_features: 0
    bias: true
  linear_1:
    <<: *LINEAR
    in_features: 784
    out_features: &out_1 100
  linear_2:
    <<: *LINEAR
    in_features: *out_1
    out_features: *num_classes

  relu:
    inplace: false
```

### `最適化手法`のパラメータの設定

現在使用可能な`最適化手法`は、`SetupOpt`クラスの`func`のキーを[参照](pytorch_template/lib/optimizer/setup.py)。

- `最適化手法`のパラメータは、`train.py`でのみ使用する。

`最適化手法`の設定の例

```yaml
opt:
  kind: adam

  adam:
    lr: 0.001
    betas: [0.9, 0.999]
    eps: 0.00000001
    weight_decay: 0
    amsgrad: false
    foreach: null
    maximize: false
    capturable: false
    differentiable: false
    fused: null
```

### `誤差関数`のパラメータの設定

現在使用可能な`誤差関数`は、`SetupLoss`クラスの`func`のキーを[参照](pytorch_template/lib/loss/setup.py)。

`誤差関数`の設定の例:

```yaml
loss:
  kind: ce

  ce:
    weight: null
    ignore_index: -100
    reduction: mean
    label_smoothing: 0
```

### `メトリクス`のパラメータの設定

現在使用可能な`メトリクス`は、`SetupMetrics`クラスの`func`のキーを[参照](pytorch_template/lib/metrics/setup.py)。

- `メトリクス`の`kind`は、リストとして設定する。

`メトリクス`の設定の例:

```yaml
metrics:
    kind: [mse, bacc]

  mse:
    squared: true
    num_outputs: 1

  bacc:
    threshold: 0.5
    multidim_average: global
    ignore_index: null
    validate_args: true
```

### `コールバック`のパラメータの設定

現在使用可能な`コールバック`は、`SetupCallbacks`クラスの`func`のキーを[参照](pytorch_template/lib/callbacks/setup.py)。

- `コールバック`のパラメータは、`train.py`でのみ使用する。

- `コールバック`の`kind`は、リストとして設定する。

`コールバック`の設定の例:

```yaml
cb:
  kind: [ms, mcp]

  ms:
    max_depth: 3

  mcp:
    monitor: null
    verbose: false
    save_last: null
    save_top_k: -1
    save_weights_only: false
    mode: min
    auto_insert_metric_name: true
    every_n_train_steps: null
    train_time_interval: null
    every_n_epochs: 1
    save_on_train_epoch_end: null
    enable_version_counter: true
```

## 開発環境の構築

python の開発環境がまだ整っていない方は、以下を参照してください。

- [開発環境の構築手順](https://github.com/r-dev95/env-python)

Sphinx ドキュメントのビルドについては、[こちら](docs/sphinx_docs.md)を参照してください。

## ライセンス

本リポジトリは、[BSD 3-Clause](LICENSE)に基づいてライセンスされています。
