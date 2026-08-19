# 関連研究における位置づけ

このプロジェクトが「誰に対して、何を新しいと主張するのか」を確定させるための調査記録
（2026-08-18 時点）。論文・スライドの序論の素材として書いてある。

**結論を先に**: 構成要素はどれも確立した研究・標準化トラックの上にあり、その**交点だけが
空いている**。個々の機構が奇抜なのではなく、組み合わせが未踏。

---

## 全体像 — 4つの層

隣接分野は、運ぶもの（画像 / CSI / 勾配 / LLM 潜在表現）と、チャネルを本気で扱うかどうかで
きれいに分かれる。

```
                             運ぶもの
                  ┌────────────────┬────────────────┐
                  │ 画像・CSI・勾配 │  LLM 潜在表現   │
  ┌───────────────┼────────────────┼────────────────┤
  │ 実チャネル     │   層1・2・3     │ ★ 本プロジェクト│
  │ （ノイズ有り） │   （成熟）      │   （空白）      │
  ├───────────────┼────────────────┼────────────────┤
  │ ノイズ無し     │       —        │   層4           │
  │ （DC 内前提）  │                │  （急成長中）   │
  └───────────────┴────────────────┴────────────────┘
```

### 層1 — AI ネイティブ・エアインターフェース（標準化・事業化が進んでいる層）

**ここが一番お金と組織が動いている。** そして重要なことに、本プロジェクトの機構は
ここで既に産業実証されつつある。

- **3GPP** — Release 18 で AI/ML が NR エアインターフェースに正式導入された
  （CSI フィードバック圧縮、ビーム管理、測位）。さらに **Release 20 で
  "two-sided AI model" が正式なワークアイテム**となり、2027年後半に完了予定。
  端末側にエンコーダ、基地局側にデコーダを置いて両者を同時に学習させるという構造は、
  `airComp/jscc/modules.py` の `SemanticEncoder` / `SemanticDecoder` と**同型**である。
  違いは運ぶものが CSI か LLM 潜在表現かだけ。
- **AI-RAN Alliance** — 2026年2月時点で 132 社、MWC 2026 で 33 のデモ。
  **O-RAN Alliance** は Release 5 を 2026年6月に完了し、AI/ML ワークフローを含む。
- **DeepSig**（2016年創業）— このアプローチの直系の商用化企業。創業者 Tim O'Shea による
  O'Shea & Hoydis 2017 のオートエンコーダ型物理層が本プロジェクトのアーキテクチャの祖先にあたる。
  ニューラル受信機・AI ネイティブ波形をシミュレーションからフィールド試験・実配備まで
  持っていっており、NASA の TDRS 通信系にチャネルオートエンコーダを適用した実績もある。

> **使い方**: 「ニューラルなエンコーダ・デコーダ対を実チャネル越しに end-to-end 学習する」
> という機構そのものは、もはや研究仮説ではなく標準化トラックに乗っている。これは本研究の
> 新規性を削るのではなく、**土台が固いことの証拠**として序論に置ける。

### 層2 — セマンティック通信 / Deep JSCC（学術の本流）

- **Deniz Gündüz（Imperial College London）** — DeepJSCC（Bourtsoulatze, Kurka, Gündüz 2018）。
  画像画素を複素チャネル入力に直接写像し、中央に学習不可能なチャネル層を挟んだ
  オートエンコーダとして訓練する。**低 SNR・低帯域で、JPEG + 容量達成符号のデジタル方式を
  上回る**ことを示した — 本プロジェクトが検証しようとしている「緩やかな劣化 vs 崖」の原型。
- **Xie, Qin, Li, Juang — DeepSC**（2020）— Transformer ベースのテキストのセマンティック通信。
  ビット/シンボル誤りではなく**文の意味の復元**を最大化する。「テキストをニューラルに符号化して
  ノイズチャネルに通す」という本プロジェクトのベースライン対照に最も近い先行研究。
- **Walid Saad（Virginia Tech）、Mehdi Bennis（Oulu）、Petar Popovski（Aalborg）** —
  goal-oriented / task-oriented 通信の理論側。
- **実機検証は既に存在する — ここが方法論上の直接の先行研究。**
  Ding, Jiang, Wen, Jin, *Adaptive Wireless Image Semantic Transmission: Design, Simulation,
  and Prototype Validation*（arXiv:2410.17536）は、ViT ベース JSCC（ASCViT-JSCC）を
  **SDR + 組込み GPU のプロトタイプで over-the-air 試験**し、シミュレーションと実測の
  両方を報告している。タイトルに "Prototype Validation" を掲げている点で、
  `hwlab/` がやっていることと構成が最も近い。
  また知識グラフ駆動のセマンティック通信を USRP 2954R × 4 台 + GNU Radio の SDR
  テストベッドで実装した例もある（arXiv:2303.08546）。
  **つまり「JSCC を実 SDR で検証する」こと自体は新規ではない。** 本プロジェクトの新規性は
  運ぶものが LLM の潜在表現であること、および評価が下流タスクの効用であることに絞られる。
  この2本の存在は序論で先に認めておいたほうが強い。
- **Next G Alliance（ATIS）** が 2026年初夏にセマンティック通信パネルを開催
  （InterDigital、Ericsson、Nokia、Virginia Tech、Ohio State）。標準化団体が扱い始めた段階。

### 層3 — Over-the-Air Computation

ほぼ学術。応用先は**連合学習の勾配集約**（多数端末の重畳で平均を計算する）に収束しており、
2エージェント間の意味伝達としての AirComp はむしろ少数派。近年の関心は
「デジタル AirComp」（低 SNR 域での信頼性改善）に移っている。

### 層4 — LLM 潜在通信（新興、ただし**チャネルが無い**）

**これが調査で一番重要な発見。**

BUPT のサーベイ "Beyond Tokens: A Unified Framework for Latent Communication in
LLM-based Multi-Agent Systems"（arXiv:2606.05711）が、**2024〜2026 年の 18 手法**を
WHAT / WHICH / HOW の3軸で整理している。18 手法中 9 つが KV キャッシュを送っており、
「KV キャッシュが事実上の標準になりつつある」とされる。**Interlat** は最終層の隠れ状態を
そのまま送る方式で、`airComp/agents/llm_onnx.py` の `chat_with_hidden` のプーリングと同発想。

**しかしこのサーベイには、無線チャネル・ノイズ・JSCC・セマンティック通信への言及が一切ない。**
挙げられている未解決課題は「アーキテクチャ間の整合」「潜在チャネルの安全性」
「エッジ展開のための圧縮」であり、物理層が視野に入っていない。彼らの言う「チャネル」は
抽象的なデータ経路であって、電波ではない。

---

## 空白地帯

```
層1・2・3（無線側） : ニューラル符号化を実チャネルで検証   ← 画像・CSI・勾配を運ぶ
層4  （LLM 側）     : LLM 潜在表現を交換                  ← DC 内、ノイズ無し
──────────────────────────────────────────────────────────────────
本プロジェクト       : LLM 潜在表現 × JSCC × 実 RF × 下流タスクでの評価
```

明示的に検索した範囲では、**LLM の潜在表現を SDR で実際に送信している公開研究は
見つからなかった。**

## 本プロジェクトの差別化

上の空白に入っていることに加えて、方法論として2点強い。

1. **下流タスクの効用で測っている。** 層2 の大半は PSNR / BLEU で測る。本プロジェクトは
   交渉ゲーム（Lewis et al. 2017 の Deal or No Deal 系）の**合意率・社会厚生・Pareto 効率**
   で測る（`airComp/eval/metrics.py`、`airComp/env/scoring.py`）。「意味が保たれたか」の
   測り方としてこちらが本質的。
2. **対シードでの実機 vs シミュレーション比較。** 両スイープが `int(snr_db * 10_000) + 1_000_000`
   という同一のシード式を使うため（`airComp/eval/snr_sweep.py` と
   `hwlab/scripts/run_sdr_sweep.py`）、同じ SNR・同じインデックスのエピソードが同一の
   pool / values を見る。先行研究の「実チャネルはシミュレーションと概ね一致した」という
   定性的主張より一段厳密で、**系統的なズレをバグとして検出できる**。

---

## 予想される批判と、それへの答え

### 「実運用の無線は必ず誤り訂正されている。生の AWGN に LLM の潜在ベクトルを晒す状況が現実にあるのか？」

最も来る指摘。`CLAUDE.md` の "Explicitly out of scope" が既に半分認めている論点
（実 LAN / Wi-Fi は物理層で訂正済みなのでビット誤りが観測できない）。答えは3つ。

1. **JSCC の前提そのもの。** 訂正符号・再送・プロトコルオーバーヘッドを丸ごと予算から
   外せることが利得の源泉。定量的な根拠は本プロジェクトが既に持っている —
   生ペイロードのバイト数（`effective_bits`）と Shannon 等価ビット
   （`k * 0.5*log2(1+SNR_linear)`、`semantic_bits_equivalent`）、および実機側の同期・
   パイロットのオーバーヘッド（`SDRAnalogChannel.payload_accounting`）。
   **この両方を併記すること**（`CLAUDE.md` の指示）。ビット数だけを根拠に帯域の主張をしない。
2. **レイテンシ。** ARQ の往復が無い。エージェント間交渉は往復回数が効くタスクなので相性が良い。
3. **電力・帯域が本当に厳しい端。** 衛星、ドローン、センサ網。DeepSig が NASA の TDRS で
   チャネルオートエンコーダを使ったのは、まさにこの理由。

### 「ベースラインが弱いのではないか（FEC 無しの生ビットは藁人形では）」

そのために `airComp/channel/digital.py` に `arq` モード（CRC-8 検出＋破棄）がある。
`raw` は崖を見せるための対照、`arq` が「現代的なデジタル通信」としての本命ベースライン。
両方を同じ図に出すこと。

### 「LoRa のような確立済み LPWAN の方が優れているのではないか」

Semtech の LoRa（チャープスペクトラム拡散、LPWAN の物理層）と比較すると何が言えるか。
層が違うので「どちらが優れているか」を2つに割る必要がある。

1. **到達距離・省電力は LoRa の圧勝。** 拡散率を上げれば雑音フロアより数 dB 低い SNR
   （実質 −20 dB 程度）でも復調でき、mW 級電力で数 km 届く。認証済みシリコンと
   LoRaWAN という確立したネットワークスタックもある。本プロジェクトの semantic
   パイプラインは `hwlab/` の同軸ベンチでしか検証しておらず、到達距離・電力・電波法
   認証について何も答えを持たない。物理層の生の性能では勝負にならない。
2. **限界 SNR での壊れ方は、本研究の主張が効く場面であり、LoRa に対しても成立する。**
   LoRa は本質的にビット完全再現の伝送で、パケットは CRC を通るか丸ごと落ちるかの
   どちらか（§0 の `compact`/`compact_fec` と同じ壊れ方で、しきい値が低いだけ）。
   拡散率が買っているのは「崖の位置を下げる」ことで「崖を滑らかにする」ことではない。
   semantic の実測特性（`docs/results.md` §2, §6 — オファーが L1 距離的に連続的に
   劣化し、いきなり暗黙の REJECT には落ちない）は程度でなく種類の差であり、
   LoRa の拡散利得では代替できない。ただし LoRa の強みはそもそも semantic が届かない
   ほど劣悪なリンクでも受信できる SNR 領域の広さにあるので、実務上この差が効く場面は
   両者の動作 SNR 域が重なるところに限られる。
3. **構造的なミスマッチもある。** LoRa モデムはバイト列を受け取ってチャープ変調する
   インターフェースしか持たず、`AnalogAWGNChannel`（`airComp/channel/analog.py`）の
   ように k=16 の連続値ベクトルを直接渡す API は無い。LoRa に乗せるには潜在ベクトルを
   一度ビット量子化する必要があり、それは semantic が避けようとしている「デジタルの
   崖」を再導入してしまう。緩やかな劣化はアナログチャネルそのものの性質であって、
   符号化方式だけでは得られない。

**結論**: LoRa は「小さいペイロードを電池で何年も km 級に飛ばす」問題を解き、semantic
は「雑音でタスクの結果が緩やかに劣化するように意味を符号化する」問題を解く。
土俵が違うので置き換えではなく、LoRa（または他の確立した PHY）で到達距離・省電力を
確保した上で JSCC 的な工夫を重ねる、という位置づけが妥当（ただし市販 LoRa チップは
不可、SDR で LoRa 風波形を自前実装する場合に限る）。

### 「LLM の潜在表現は送信側モデルに固有で、受信側で解釈できないのでは」

そのとおりであり、だからこそデコーダは**受信側 LLM に注入可能なコンテキストを復元しようとしない**。
構造化されたオファー（アイテム数・アクション・補助的な連続 intent 信号）を直接再構成する
（`airComp/jscc/modules.py`）。層4 のサーベイが挙げる「クロスアーキテクチャ整合」問題を、
タスク側の構造で回避している格好。

---

## 国内の状況

- **NTT ドコモ** — 「6G Harmonized Intelligence」プロジェクト。ビジョンは
  **「AI のためのネットワーク」**で、本プロジェクトの前提と同じ言葉を使っている。
- **Nokia + NTT ドコモ + NTT** — AI/ML を無線インターフェースに実装した
  AI ネイティブ無線インターフェースを実機化し、MWC 2023 でデモ済み。

「LLM エージェント間の潜在表現を実 RF で交換する」をやっている国内組織は、
探した範囲では見つからなかった。

---

## 出典

主張と1対1で対応させてある。

**層1（AI ネイティブ・エアインターフェース）**
- 3GPP, AI/ML for NR Air Interface — https://www.3gpp.org/technologies/ai-ml-nr
- AI Network Standardisation Moves Towards AI-Native 6G（Rel-20 two-sided model、2027年後半完了予定）
  — https://www.free6gtraining.com/2026/08/ai-network-standardisation-moves.html
- AI-RAN: untying the knot for 6G（132社、MWC 2026 の 33 デモ、O-RAN Release 5）
  — https://the-mobile-network.com/2026/02/ai-ran-untying-the-knot-for-6g/
- NVIDIA, Boosting AI-Driven Innovation in 6G with the AI-RAN Alliance, 3GPP, and O-RAN
  — https://developer.nvidia.com/blog/boosting-ai-driven-innovation-in-6g-with-the-ai-ran-alliance-3gpp-and-o-ran/
- DeepSig, From Lab to Field: Evolving Deep Learning for Communication Systems
  — https://www.deepsig.ai/from-lab-to-field-evolving-deep-learning-for-communication-systems/
- NVIDIA, DeepSig: Deep Learning for Wireless Communications（NASA TDRS のチャネルオートエンコーダ）
  — https://developer.nvidia.com/blog/deepsig-deep-learning-wireless-communications/
- O'Shea & Hoydis, An Introduction to Deep Learning for the Physical Layer (2017),
  IEEE TCCN 3:563-575 — https://arxiv.org/abs/1702.00832

**層2（セマンティック通信 / Deep JSCC）**
- Bourtsoulatze, Kurka, Gündüz, Deep Joint Source-Channel Coding for Wireless Image
  Transmission (2018) — https://arxiv.org/abs/1809.01733
- Xie, Qin, Li, Juang, Deep Learning Enabled Semantic Communication Systems (DeepSC, 2020)
  — https://arxiv.org/abs/2006.10685
- Cognitive Semantic Communication Systems Driven by Knowledge Graph
  （USRP 2954R × 4 + GNU Radio の SDR テストベッド）— https://arxiv.org/abs/2303.08546
- Ding, Jiang, Wen, Jin, Adaptive Wireless Image Semantic Transmission: Design, Simulation,
  and Prototype Validation（ASCViT-JSCC、SDR + 組込み GPU での over-the-air 試験）
  — https://arxiv.org/abs/2410.17536
- DD-JSCC: Dynamic Deep Joint Source-Channel Coding — https://arxiv.org/abs/2507.20467
- Next G Alliance (ATIS), Technology Roadmap Working Group: Semantic Communications
  — https://nextgalliance.org/technology-roadmap-working-group-semantic-communications/

**層3（Over-the-Air Computation）**
- Over-the-Air Computation for 6G: Foundations, Technologies, and Applications
  — https://arxiv.org/abs/2210.10524
- Learned Digital Codes for Over-the-Air Computation in Federated Edge Learning
  — https://arxiv.org/abs/2512.19777

**層4（LLM 潜在通信）**
- Beyond Tokens: A Unified Framework for Latent Communication in LLM-based Multi-Agent
  Systems（18 手法、2024-2026。無線チャネルへの言及なし）— https://arxiv.org/abs/2606.05711
- Du et al., Enabling Agents to Communicate Entirely in Latent Space
  （Interlat = Inter-agent Latent Space Communication、最終層隠れ状態を直送）
  — https://arxiv.org/abs/2511.09149
- Latent Communication Between Language Model Agents: Channels, Alignment, and the Limits
  of Text — https://arxiv.org/abs/2607.14103

**国内**
- NTT ドコモ, 6G Harmonized Intelligence プロジェクト始動
  — https://www.docomo.ne.jp/corporate/anatatodocomo/docomoeveryday/article67/
- Nokia, ノキア、ドコモ、NTT 3社による2つの技術開発により、6Gが大きく前進（MWC 2023 デモ）
  — https://www.nokia.com/ja_jp/about-us/news/releases/2023/02/15/nokia-docomo-and-ntt-make-two-key-6g-advances/

**タスク**
- Lewis, Yarats, Dauphin, Parikh, Batra, Deal or No Deal? End-to-End Learning for
  Negotiation Dialogues (2017) — https://arxiv.org/abs/1706.05125

---

## この文書の更新について

層1（標準化）と層4（LLM 潜在通信）は動きが速い。特に **3GPP Rel-20 の two-sided model は
2027年後半完了予定**なので、その前後で本文の「進行中」という記述は書き換えが要る。
出典の無い断定をここに足さないこと — 上の一覧と本文の主張は1対1で対応させてある。

---

## 追記（2026-08-19）— 主張の下方修正

本文の差別化は「下流タスク効用での評価」「対シードでの実機 vs シミュレーション」の2点を挙げて
いるが、その上で報告していた **20〜25 dB という数字は撤回する**。

当時のデジタル基準線は LLM の生成テキスト全文（約 1000 ビット）を送っており、
セマンティック側は 16 実数だった。6.1 ビットの通信行為に対する 100〜200 倍の水増しであり、
差の半分以上は**ソース符号化と誤り訂正の欠如**で説明がついた。

公平化した基準線（オファーを 8 ビットに源符号化 + Hamming(7,4) + チャネル使用数 16 で一致 +
ターン数も同一）に対する実効 SNR 利得は **11〜14 dB**。詳細と分解は
[docs/results.md](results.md)。

先行研究に対して主張できるのはこの 11〜14 dB のほうであり、**査読者が最初に突くのは
基準線の強さ**なので、`compact_fec` の構成を明示せずに数字だけ出さないこと。
