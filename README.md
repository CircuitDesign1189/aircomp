# AirComp

従来のテキストベースな AI エージェント間通信と、**セマンティック通信 / 統合情報源・通信路符号化
(JSCC: Joint Source-Channel Coding)** パイプライン ―― LLM の潜在表現を圧縮し、雑音のある無線通信路
のシミュレーション上で伝送する方式 ―― を比較する研究用プロトタイプです。評価タスクには二者間の
交渉ゲームを用います。

アーキテクチャ全体、設計判断の根拠、リポジトリ構成については [CLAUDE.md](CLAUDE.md) を参照してください。

## クイックスタート

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126   # GPU 版。CPU のみで動かす場合はこの行を省略
python scripts/download_model.py --model Qwen/Qwen2.5-1.5B-Instruct

# 従来方式（テキスト + ディジタル通信路）のベースライン
python evaluate.py run-baseline --episodes 50 --snr-db 10 --channel-mode raw

# JSCC 学習データを収集し、セマンティック エンコーダ / デコーダを学習
python train.py collect-dataset --episodes 500 --out data/jscc_dataset.pt
python train.py train-jscc --dataset data/jscc_dataset.pt --out checkpoints/jscc_v1.pt

# SNR スイープで両パイプラインを比較
python evaluate.py snr-sweep --checkpoint checkpoints/jscc_v1.pt --episodes 100 --out results/sweep.json

pytest -m "not slow" -q
```

## 制約事項

- **実際の無線送信もネットワーク伝送も行いません。** 2 つのエージェントは同一マシン上の同一 Python
  プロセス内で動作し、「通信路」はテンソルやビット列に適用される数学的な雑音モデルであって、実際の
  RF やネットワーク伝送ではありません。これはまずアルゴリズム的な仮説を検証するために意図的に設定
  したスコープです（CLAUDE.md 参照）。
- **ビット数とシンボル数は直接比較できません。** スイープでは、生のペイロードサイズの比較
  （ディジタル方式はビット数、セマンティック方式は `k` 個の実数シンボル）に加えて、セマンティック
  通信路についてシャノン容量換算のビット数推定値（`k * 0.5*log2(1+SNR_linear)`）も併せて報告します。
  これは、両者を同一条件の帯域幅比較として誤読されないようにするためです。
- **Phase 1 の JSCC 学習中、LLM は凍結されています**（特徴抽出器としてのみ使用）。LLM 自体の同時
  ファインチューニングは今後の課題であり、CLAUDE.md に記載した微分可能なタスク成果ベースの Phase 2
  ファインチューニングも同様に未実装です。
- 評価は単一の小規模モデル（Qwen2.5-1.5B-Instruct）と、合成された 3 品目の交渉タスクで行っています。
  より大きなモデルやより複雑な交渉ドメインに結果が一般化するとは限りません。
