<!-- translation-of: CLAUDE.md -->
<!-- source-sha256: 16a9c63bc85e5aca662e9db4449dfd418cc1c7fcb69b1cb4244f88725cb65f04 -->

# CLAUDE.md（日本語訳）

**これは [CLAUDE.md](CLAUDE.md) の翻訳であり、正典ではない。** 齟齬があれば英語版が正しい。
上の `source-sha256` は、この訳が作られた時点の CLAUDE.md のハッシュ。
`tests/test_claude_md_translation.py` が両者のずれを検出する。
更新手順は `python scripts/sync_translation.py --update`。

このファイルは、このリポジトリで作業する Claude Code (claude.ai/code) への指示を与える。

## プロジェクトの狙い

AirComp は **AI ネイティブ無線通信**を探る研究プロトタイプである。AI エージェント同士が
人間可読なデジタルプロトコル（テキストをビットにトークン化し、往復遅延を伴う従来型 API で
送る）で情報をやり取りするのではなく、LLM の内部表現を直接圧縮し、実際の無線劣化
（雑音、限られた SNR）をモデル化したチャネル上で交換する。

検証対象の具体的な仮説: **セマンティック / Joint Source-Channel Coding (JSCC)**
パイプライン — LLM の隠れ状態から導いた圧縮済みの連続潜在ベクトルを、シミュレートされた
アナログ AWGN チャネルで送る — は低 SNR で*緩やかに*劣化する。一方、同じ通信内容を
テキスト/JSON にシリアライズしてビットとしてデジタルチャネルで送る**従来型パイプライン**は
*壊滅的に*劣化する（`{`、`"`、`:` の中の 1 ビット反転で JSON 解析が完全に壊れる）。

これは確立した2つの研究の流れに接続する:
- **Over-the-Air Computation (AirComp)**: 復号してから計算するのではなく、無線の重ね合わせを
  利用して送信信号の関数を直接計算する。
- **セマンティック通信 / JSCC**: 正確なビット復元を最適化するのではなく、チャネル雑音下で
  タスクに関係する意味を保つよう端から端まで学習したニューラル符号化器/復号器の対。

`docs/results.md` が「これは機能するのか？」への現時点の答え: **機能する。`compact_fec` に対して
実効 SNR 利得 11〜14 dB。** Phase 3 が冗長な JSON に対して主張していた 20〜25 dB からの下方修正で、
その半分以上はソース符号化と FEC の欠如だった。**数字を引用する前に必ず読むこと。**

`docs/related_work.md` は、上記2つの流れにさらに2つ（3GPP/AI-RAN の「AI ネイティブ・
エアインターフェース」標準化トラック、および 2024-2026 年の LLM 潜在通信の文献）を加えた中に
本プロジェクトを位置づける。新規性を主張する文章を書く前に読むこと。ここには新規で*ない*ものが
2つあり、それを新規と主張すると仕事ごと退けられる: 両側エンコーダ/デコーダの機構
（3GPP Release 20 が CSI を対象としたワークアイテムとして持っている）と、実 SDR での JSCC 検証
（arXiv:2410.17536 がプロトタイプ検証論文）。空いているのは**組み合わせ** — ペイロードとしての
LLM 潜在表現、下流タスク効用による評価、ハードウェアとシミュレーションのエピソード単位での
対応付け。同ドキュメントには、想定される標準的な反論と、それに答えるために本リポジトリが
既に持っているものも列挙してある。

**スコープ**: チャネル実装は現在2つあり、DSP に面する継ぎ目を共有している。

- **シミュレーション** (`airComp/`): 両エージェントが1プロセス内で逐次実行され、「無線チャネル」は
  テンソル演算 (`z + noise`)。これは JSCC/セマンティック通信の文献における標準的な評価手法であり、
  `airComp/eval/` の SNR 掃引が使っているもの。
- **実 RF** (`hwlab/`): 同じ潜在ベクトルをパルス整形し、片方の HackRF One から送信して、
  較正済み減衰器を挟んだ導波路（同軸）経路でもう一方が受信する。`hwlab/README.md` を参照。
  両方の掃引が同じ式からエピソードシードを導くので、曲線はエピソード単位で対応し重ね描きできる。
  両者の間の系統的なズレは物理現象ではなくバグである。**アナログ経路とデジタル経路の両方を
  ベンチで測定済み** — 16 ビットの `compact_fec` フレームは 16 実数次元、すなわち k=16 の潜在
  ベクトルが占めるのと同じ 8 複素シンボルなので、同一のバーストに同一の送信電力で載る
  (`hwlab/channel/sdr_digital.py`)。**どちらも実効 SNR 利得の数字を再現するものではない**:
  ベンチの到達範囲は −11.6..+25.1 dB なので、正規化に必要な −60 dB の床も +40 dB の天井も測れない。

## アーキテクチャ

同一タスク・同一チャネル条件で2つのパイプラインを比較する:

**ベースライン（従来型）パイプライン** — `airComp/baseline/`、`airComp/agents/baseline_agent.py`、
`airComp/channel/digital.py`:
1. LLM エージェントが構造化 JSON の提案をテキストとして生成する。
2. テキスト -> UTF-8 ビット -> BPSK -> AWGN(`SNR_dB`) -> 硬判定復調 -> ビット -> UTF-8（破損の
   可能性あり）-> 正規表現/JSON 抽出 -> pydantic 検証。
3. チャネルモードは2つ: `raw`（FEC なし）と `arq`（CRC-8 で検出して破棄）。**どちらも誤りを
   訂正せず、どちらも LLM の生成全文をワイヤに載せる（6.1 ビットのオファーを伝えるのに約 1000
   ビット）。** これは素朴なデジタルの参照点であって、公平な基準線ではない — 下記参照。

**Compact（公平なデジタル）基準線** — `airComp/agents/compact_agent.py`、
`airComp/baseline/offer_codec.py`: LLM のターンは同一だが、解析済みのオファーを固定 8 ビットの
フレーム（プールの実行可能 count ベクトルへのインデックス + アクション）に源符号化し、同じ
BPSK/AWGN チャネルで送る。`compact_fec` は Hamming(7,4) を加え、**チャネル使用数 16 —
セマンティックパイプラインの k=16 実数チャネル使用と、実数次元あたり同一の SNR で完全に一致する。**
これがリポジトリ内で唯一の apples-to-apples な比較であり、あらゆる主張はこれに対して述べること。

**セマンティック/JSCC パイプライン** — `airComp/agents/semantic_agent.py`、
`airComp/channel/analog.py`、`airComp/jscc/`:
1. LLM の隠れ状態を、オファー JSON に対応するトークンだけで平均プーリングする（私的な推論
   トークンは決して含めない — これが比較の公平性を保つ: 両パイプラインは同じ通信行為を送り、
   異なるのは符号化とチャネルだけ）。
2. `SemanticEncoder`（MLP）がこれを `k` 次元の電力正規化された実ベクトルに圧縮する。
3. `AnalogAWGNChannel` がガウス雑音を加える（微分可能で、端から端までの学習を可能にする）。
4. `SemanticDecoder`（MLP）が構造化オファー（品目数、アクション、補助的な連続「意図」信号）を
   直接再構成する — 独立にインスタンス化されたモデル呼び出しをまたいで注入可能な LLM
   コンテキストを再構成しようとはしない。
5. 学習は教師あり（Phase 1: 凍結した LLM を特徴抽出器として使い、SNR をランダム化）で、
   任意で微分可能なタスク成果によるファインチューニング段階（Phase 2）を持つ。

**タスク/環境** — `airComp/env/negotiation.py`: 簡略化した "Deal or No Deal" 型の二者間交渉ゲーム
（Lewis et al. 2017）。2エージェントが3種類の品目（`book`/`hat`/`ball`）のプールを分割する。
品目ごとの価値は私的かつ独立にランダム化される（プール全体が 100 点になるよう正規化）。
最大 10 回の交互メッセージ（`PROPOSE`/`ACCEPT`/`REJECT`）でやり取りする。
不成立またはラウンド切れは両エージェントとも効用 0。

**評価** — `airComp/eval/snr_sweep.py`: 両パイプライン（ベースラインは両チャネルモード、加えて
セマンティック）を、対応付けた乱数シードで SNR グリッド上に走らせ、合意率、効用、社会厚生、
Pareto 効率、実効ビット/シンボル数を記録する。中心的な結果は、SNR にわたる
「緩やか vs 壊滅的」な劣化の比較である。

### 明示的にスコープ外（現時点では）

- **実ネットワーク転送。** プロセス間・マシン間のソケット転送（TCP/UDP）は行わない。
  実際の LAN/Wi-Fi は物理層で既に誤り訂正しているので、そのスタックより下に降りない限り
  ビット誤りは観測できない。
- **空中への輻射。** HackRF 経路は*導波路*のみ: 同軸と較正済み減衰器で、何も輻射しない。
  HackRF One に日本の技適マークは無いので、シールドボックス外での輻射は電波法に違反する。
  科学的にも最もきれいなチャネルである — マルチパスもフェージングも外来干渉も無く、
  実際の無線機として AWGN モデルに最も近い。
- **2エージェント間の空間的分離。** 両方の無線機は1つのベンチ上にあり、1プロセスが駆動する。

「実 RF 無し」がこのリストから外れたことに注意: `hwlab/` は実際に送信する。
`airComp/channel/base.py` の抽象 `Channel` の継ぎ目があったからこそ、エージェントコードにも
タスクコードにも触れずにこれを追加できた。`hwlab/radio/backend.py` が同じ考えを1段下で担っている。

## リポジトリ構成

```
airComp/
  config.py                 # dataclass: ModelConfig, ChannelConfig, JSCCConfig, NegotiationConfig, TrainConfig
  env/
    negotiation.py          # Pool, Values, Offer, EpisodeState, generate_pool(), generate_values(), utility(), run_episode()
    scoring.py               # social_welfare(), pareto_frontier(), pareto_efficiency()
  agents/
    llm_backend.py           # LocalLLM: chat(), chat_with_hidden()
    prompts.py                # システムプロンプトのテンプレート、JSON スキーマの指示
    parser.py                  # 正規表現+JSON 抽出、pydantic Offer スキーマ、有界リトライのロジック
    baseline_agent.py         # TextAgent: LLM のテキストターン -> DigitalChannel -> 解析済み Offer
    compact_agent.py           # CompactAgent: 同じターン、オファーを 8 ビットに源符号化（公平な基準線）
    semantic_agent.py         # SemanticAgent: 隠れ状態 -> SemanticEncoder -> AnalogAWGNChannel -> SemanticDecoder -> Offer
  channel/
    base.py                    # 抽象 Channel インターフェース（SDR バックエンドがこれを実装する）
    digital.py                  # DigitalChannel: ビット/BPSK/AWGN/復調、"raw"/"arq"/"fec" モード
    analog.py                    # AnalogAWGNChannel(nn.Module): 実ベクトルへの微分可能な AWGN
    fading.py                     # 任意: Rayleigh ブロックフェージング版（発展）
  jscc/
    modules.py                    # SemanticEncoder, SemanticDecoder (nn.Module)
    dataset.py                     # collect_dataset(): 凍結 LLM による自己対戦 -> (hidden_state, offer) の組
    losses.py                       # オファー/アクションの CE 損失、補助 MSE、期待効用の代理目的（Phase 2）
    train_jscc.py                    # 学習ループ、SNR ランダム化、チェックポイント
  baseline/
    run_baseline.py                   # TextAgent<->DigitalChannel<->TextAgent を N エピソード動かす
    offer_codec.py                     # Offer <-> 固定 8 ビットフレーム、共有プールを条件とする
  eval/
    metrics.py                         # agreement_rate, avg_utility, rounds_to_agreement, pareto_efficiency, effective_bits
    normalize.py                        # 床/天井の正規化 + 実効 SNR 利得 -- 見出しの数字
    reconstruction.py                    # デコーダはチャネルを使っているのか、事前分布を出しているのか？
    snr_sweep.py                        # 両パイプライン x 両チャネルモードを SNR グリッド上で、対シードで
    plots.py                             # matplotlib による比較プロット
  utils/
    seeding.py, logging.py, io.py         # RNG シード、JSONL エピソードログ、パス補助
scripts/
  download_model.py                       # ローカル LLM を ./.hf_cache に事前ダウンロード
  build_genai_model.py                    # onnxruntime-genai-directml のモデルディレクトリをビルド
  probe_dual_genai.py                     # genai-directml セッションが GPU 上で2つ同時に動くか確認
configs/
  base.yaml, snr_sweep.yaml
tests/
  test_negotiation_env.py, test_digital_channel.py, test_analog_channel.py,
  test_jscc_modules.py, test_parser.py, test_end_to_end_baseline.py（`slow` マーク付き）
train.py                                   # CLI: collect-dataset / train-jscc
evaluate.py                                 # CLI: run-baseline / snr-sweep
```

## コマンド

```powershell
# 環境構築
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -U pip
pip install torch                     # CPU ビルド。このマシンは AMD GPU -- 下記の環境メモ参照。CUDA wheel は入れないこと。
pip install -r requirements.txt

# モデルのダウンロード（既定は Qwen2.5-1.5B-Instruct — Apache-2.0、ゲート無し）
python scripts/download_model.py --model Qwen/Qwen2.5-1.5B-Instruct

# ベースライン（従来型）パイプライン
python evaluate.py run-baseline --config configs/base.yaml --episodes 50 --snr-db 10 --channel-mode raw

# JSCC のデータ収集 + 学習
python train.py collect-dataset --episodes 500 --out data/jscc_dataset.pt
python train.py train-jscc --dataset data/jscc_dataset.pt --epochs 30 --snr-range -5 20 --k 16 --out checkpoints/jscc_v1.pt

# SNR 掃引一式（両パイプライン、両チャネルモード）
python evaluate.py snr-sweep --checkpoint checkpoints/jscc_v1.pt --episodes 100 --snr-grid -10 -5 0 5 10 15 20 --out results/sweep.json

# テスト
pytest -m "not slow" -q     # 高速: チャネル数学、環境ロジック、パーサの境界条件（モデルのダウンロード不要）
pytest -m slow -q           # 統合: ダウンロード済みモデルが必要
```

## このコードを変更する前に知っておくべき設計判断

- **パイプライン間の公平性は結論を支えており、そして既に一度間違えている。** 隠れ状態は
  オファー JSON のトークン範囲*だけ*でプールしており、掃引は同一のシードによるプール/価値を
  使っている — にもかかわらず Phase 3 は 16 シンボルの潜在ベクトルを約 1000 ビットの散文と
  比較し、その結果の 20〜25 dB を JSCC の成果として報告した。**大半はソース符号化だった。**
  利得を主張する前に4つの予算をすべて確認すること: **ペイロードの情報量、チャネル使用数、
  誤り訂正、ターン数。** `compact_fec` は4つすべてを揃える。`raw`/`arq` は1つも揃えない。
- **パイプライン間で生の合意率を比較してはいけない。** 0 も 1 も共有していない。
  床（−60 dB で実測）: セマンティック **0.48** — デコーダが常に妥当なオファーを出すので、
  2つの事前分布が偶然一致する — に対し compact 系は 0.02〜0.10（復号不能なフレームが暗黙の
  REJECT になるため）。天井（+40 dB で実測）: 0.90〜0.98 で、`CompactAgent` は自前の JSON を
  解析する必要があるが `SemanticAgent` は不要、という違いによる。
  各曲線を自身の床〜天井のレンジで割る `airComp/eval/normalize.py` を使うこと。
- **`lost_message_ends_episode` は公平性のつまみであって、細部ではない。** 既定の `True` は
  Phase 3 を再現するが、最初の復号不能メッセージでエピソードを終了させる — これはデジタル
  パイプラインにしか起こせない。約 2 dB に相当する。公平な数字が要るときは `False`
  （`--survive-lost-messages`）にすること。
- **ビット vs シンボルは apples-to-apples の比較ではない。** セマンティックチャネルについては、
  生のペイロードサイズの比較と Shannon 容量等価ビットの推定（`k * 0.5*log2(1+SNR_linear)`）の
  **両方**を報告すること — ビット数だけを帯域公平性の根拠にしない。この推定にはそれ自身の
  床の問題があることにも注意: −15 dB では 1 エピソードあたり 0.83 ビットが受信側に届くと
  言っているのに合意率は 0.74 で、これは 0.48 が事前分布だからこそ成立している。
- **モデル選択**: Qwen2.5-1.5B-Instruct を Llama-3.2-Instruct より優先したのは、Hugging Face で
  ゲートされておらず（ライセンス同意のクリックが不要）、量子化なしで fp16 に余裕をもって
  収まるため。`bitsandbytes` は意図的に依存関係に入れていない（Windows で不安定、この
  モデルサイズでは不要）。
- **不正な LLM 出力は実在する測定対象の失敗モード**であり、黙ってリトライで消すものではない —
  有界リトライ（最大 2 回）はあるが、使い切った場合は指標上の暗黙の `REJECT` として数える。
- **環境: Python 3.14、GPU は AMD Radeon RX 9060 XT で、このマシンに NVIDIA カードは無い**
  （2026-08-18 検証: `VEN_10DE` デバイス無し）。これは RTX 3060 と動作する `cu126` ビルドを
  記していた本ファイルの以前の記述を上書きする。ハードウェアが交換された。
  **CUDA wheel を入れないこと** — ここで `torch.cuda.is_available()` が `True` になることは
  あり得ないので、2.5 GB のダウンロードは何も買わない。`LocalLLM` は既に CPU にフォールバック
  するので、`configs/*.yaml` の `device: "cuda"` は無害。
- **PyTorch にはここで GPU 経路が無いが、ONNX Runtime にはある — そしてこれは配線済み。**
  `torch-directml` は Python 3.14 用の wheel を出しておらず、ROCm は Linux 専用なので、`torch` は
  CPU のまま。代わりに `onnxruntime-genai-directml` が Radeon を駆動する。
  `model.backend: "onnx-dml"` を設定し（同梱の設定は既にそうなっている）、
  `python scripts/build_genai_model.py` で一度モデルをビルドする。
  `airComp/agents/llm_onnx.py` を参照。
- **汎用 ONNX グラフではなく genai を使うこと。** 250 トークンのプロンプトでの実測:
  CPU torch **345 ms/token**、DirectML 上の汎用 ONNX **189 ms/token**、DirectML 上の genai
  **11 ms/token**。バッチ 1 のデコードは数百個の小さな演算子なので、帯域ではなく演算子ごとの
  ディスパッチで律速される — これは二重に確認済みで、KV キャッシュのトラフィックを 5 倍に
  しても 8% しかコストが増えず、*int4* の汎用グラフは fp16 より**遅かった**（逆量子化が演算子を
  増やす）。genai の融合デコードカーネルがこのオーバーヘッドを取り除いている。GEMV
  マイクロベンチマークは 5.7 倍を予測し、どのボトルネックが効くかについてひどく誤解を招いた。
- **`optimum-onnx` を決して入れないこと**: transformers 5.15 を 4.57 にダウングレードし、
  onnxruntime-directml を CPU ビルドに置き換える。KV キャッシュの扱いは genai のものなので、
  そもそも不要。
- **端から端まで**: データ収集は **1 エピソードあたり 38.9 -> 9.1 秒（4.3 倍）**になり、
  歩留まりも 1.29 -> 2.00 examples/episode に上がった（int4 モデルのほうが整形の良い JSON を
  出すため）。500 エピソードの収集は約 5.4 時間ではなく約 1.3 時間。
- **隠れ状態は設計上 CPU の torch に留める。** `chat_with_hidden` は GPU で生成するが、
  プーリングは torch モデルから行うので、プールされたベクトルは移植前に収集したすべてと
  数値的に同一 — JSCC デコーダに学習/推論のずれが生じない。その prefill（約 2.8 秒）が
  現在ターンあたりの支配的コストであり、次に移すべき明らかな対象。
- **CPU 経路を bfloat16 に「最適化」しないこと。** 正しそうに見える — デコードはメモリ帯域で
  律速され、bf16 は重みのトラフィックを半減する — が、この CPU は AVX2 のみで AVX512-BF16 が
  無いため、bf16 GEMM はエミュレーションになる。デコードは何も得られず、`chat_with_hidden` の
  計算律速な prefill は 2.8 秒から 17 秒に悪化する。1 エピソード全体で **32 秒 -> 60 秒**を実測。
  `torch.set_num_threads` を 8 から 16 に上げても何も変わらず（312 vs 315 ms/token）、
  これが帯域律速であることの裏付けになっている。dtype の変更を信じる前に、tokens/s だけでなく
  prefill を測ること。
- **`onnxruntime-genai-directml` のセッションは、このマシンの GPU 上で2つ同時に動く**
  （2026-08-19 検証、`scripts/probe_dual_genai.py`）: 1つ目の `OnnxDmlLLM` が生きたままでも
  2つ目のロードと生成は問題なく成功し、専有 VRAM は1つあたり約 1.4 GB（合計約 3.4 GB、
  この Radeon の予算に十分収まる）、交互生成でも測定可能な速度低下は無い（単独実行時の
  0.86〜0.96 倍 — これはノイズであって競合ではない）。現在のパイプラインは今も両方の交渉側で
  1つの `OnnxDmlLLM` を共有している（`airComp/agents/factory.py` の `build_llm` は1回の
  実行につき1回しか呼ばれない）— 1組の重みが自己対戦で両役を演じているのであって、独立した
  2つのモデルではない。この結果が示すのは GPU/ドライバが真に独立した2モデル構成を阻んで
  いないということだけであり、実際に配線すること（2セッション、片側に1つずつ、それぞれが
  `hwlab/` の `HardwareSemanticAgent` に供給する）はまだ手つかずの別の作業である。
