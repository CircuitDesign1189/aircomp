# hwlab — HackRF One ×2 による実RF伝送レイヤ

AirComp の semantic パイプラインの連続値潜在ベクトル `z` を、**実際のRF経路**で伝送するためのパッケージ。

既存の `airComp/` は**一切変更していない**。ハードウェア経路は `SemanticAgent` を継承して
`channel` 属性を差し替える形で接続する（[agent.py](agent.py)）。シミュレーションのみの実験結果は
そのまま再現可能なまま維持される。

---

## ⚠ 最初に読むこと（機器保護）

**HackRF の TX 出力は最大 +15dBm 程度、RX の最大入力は約 -5dBm。TX と RX を同軸で直結すると
受信側が壊れます。**

- TX と RX の間には**必ず減衰器を入れる**。初期値は 30dB
- **減衰量は実測したリンクバジェットから決める。ただし一度に 10dB ずつしか下げず、その都度
  フェーズ0.5 で測り直す。** 一足飛びに 0dB にしない — 経路のどこかに接触不良があると、
  それが直った瞬間に受信機へ全電力が入る
- TX ポートを**開放のまま送信しない**
- `gains.tx_amp` は `false` のまま（+14dB の RF アンプ。本実験では不要）
- 受信側 RF アンプは常に off（`hackrf_cli.py` が `hackrf_transfer -a 0` を固定で渡す）

## ⚠ 電波法

HackRF One に**技適マークは無い**。同軸ケーブル内を伝わる分には輻射しないため、
**導波路（conducted）構成なら問題にならない**。これが本パッケージの主構成。

アンテナを使った実輻射は**シールドボックス／電波暗室の中でのみ**行うこと。

なお導波路構成は科学的にも最良で、マルチパス・フェージング・外来干渉が無く、
`airComp/channel/analog.py` の AWGN モデルに最も近い。

---

## 構成

```
HackRF #1 ─TX─ 同軸 ─ [減衰器 ≥30dB] ─ 同軸 ─RX─ HackRF #2
     └──────── CLKOUT ──── 同軸 ──── CLKIN ────────┘
```

**HackRF One は半二重**（IFトランシーバの MAX2837 が半二重パートのため）なので 2 台必要。

**クロック同期が本構成の要**。#1 の CLKOUT（10MHz, 3.3V方形波）を #2 の CLKIN に直結すると
両機の LO とサンプルクロックが同一基準から生成され、キャリア周波数オフセット(CFO)がゼロになる。
1 メッセージが**複素8シンボル**しかない本実験では、周波数オフセットを推定・追従する余地が
無いため、これが成立の前提になる。

**ハードウェアトリガは不要**。送信ファイルにバーストを必要回数だけ繰り返して書き込み、
受信はバースト3個分をまとめて取り込むので、プロセス起動のばらつきに関係なく必ず完全な
バーストが 1 個は入る。あとはプリアンブル相関で見つければよい。
（リピートを `hackrf_transfer -R` にやらせない理由は
[hwlab/radio/hackrf_cli.py](radio/hackrf_cli.py) 冒頭を参照。）

### ⚠ クロック基準リンクが受信サンプルを壊すことがある

**本ベンチで実際に起きた最大の落とし穴。** CLKOUT→CLKIN を繋ぐと、従側（…16bc62dc2e679ba7）の
ADC がフルスケールのサンプル破損を起こす。`check_path --noise-only` を CLKOUT の on/off で
実行するだけで再現する:

| lna/vga | `--clock off` | `--clock on` |
|---|---|---|
| 24/20 | rms 2.44 / peak **4** / clip 0% | rms 4.27 / peak **128** / clip 0.004% |
| 0/0 | rms 1.89 / peak **3** / clip 0% | rms 3.97 / peak **128** / clip 0.004% |

RX 利得を 44dB 下げても雑音が動かないので**利得段より後**での破損。約 1700 サンプルに 1 個の
割合でフルスケールのグリッチが入り、guard 領域の雑音推定を水増しして `RX near saturation` を
常時点灯させ、guard と pilot の SNR を 12dB も乖離させる。**DSP や配線の不具合に見える。**

診断:

```powershell
python -m hwlab.scripts.check_path --backend hackrf --clock off --noise-only   # きれいなら
python -m hwlab.scripts.check_path --backend hackrf --clock on  --noise-only   # クロックが犯人
```

対処（この順に）:

1. **クロックの向きを逆にする** — 主従は送受の役割と独立に選べる。基準ケーブルを
   もう一方の CLKOUT から出し、`device.clock_master_serial` にその個体を書く。
   従側が変われば、どちらの個体の CLKIN が弱いのかも同時に分かる
2. クロック用同軸を**短い 50Ω 同軸**に替える（レベル不足・反射・リンギングが原因になり得る）
3. それでも直らない個体は CLKIN 回路が弱い。その個体を**クロック主**（＝CLKOUT を出す側）に固定する

**本ベンチでの決着**: 手順1で解決した。**…24923a5f は CLKIN 従属に耐えるが、
…2e679ba7 は耐えない。** よって …2e679ba7 をクロック主に固定する
（`clock_master_serial`）。送受の役割はこれとは独立に決めてよく、
現在は TX=…24923a5f / RX=…2e679ba7。逆転後は両機とも peak 3〜4 LSB・クリップ 0%、
`f err` は 0 Hz、`--repeat` は spread 0.1 dB。

**クロックを切ったままにはできない。** 非同期だと本ベンチでは搬送波周波数オフセットが
**-24 kHz（26 ppm）**あり、8.71 ms のバースト中に 209 回転する。プリアンブル相関が成立しない。

### 2台の個体差

`hackrf_info` は **…16bc62dc2e679ba7** を *"Hardware does not appear to have been manufactured
by Great Scott Gadgets"* と報告する（クローン）。送信品質（トーンの wanted/image、tx_vga 40）:

| 送信機 | wanted/image | tx_vga に対する応答 |
|---|---|---|
| …a06063c824923a5f | **+35.4 dB** | 滑らか |
| …16bc62dc2e679ba7 | +17.7 dB | tx_vga 20/30 でほぼ出力なし、40 で急に出る |

**…a06063c824923a5f を送信に使う**（`sdr_link.yaml` の現状）。クローン側も致命的ではないが、
イメージ抑圧が 18dB 劣り、利得特性が不連続。

なお USB をハブ経由から PC 直挿しに変えると、クローン側の送信 wanted/image が
-1.7dB → +17.7dB に改善した。**USB 経由でサンプルが壊れると送信波形も壊れる**ので、
両機とも PC 直結にすること。

---

## セットアップ

```powershell
# 1. HackRF ツールを PATH に通す
#    この環境では radioconda 同梱のものが入っている:
#      %LOCALAPPDATA%\radioconda\Library\bin
#    （素で入れる場合は公式 release zip の bin/ を PATH に追加）
hackrf_info                      # 2台のシリアル番号を控える

# 2. Python 側は追加依存なし（プロジェクトの requirements.txt で足りる）
.\.venv\Scripts\Activate.ps1

# 3. シリアル番号を設定に書く
#    hwlab/configs/sdr_link.yaml の device.tx_serial / device.rx_serial
```

シリアル番号は**必ず両方書く**こと。空のままだと `hackrf_transfer` は「最初の1台」を開くので、
送信プロセスと受信プロセスが同じ無線機を取り合う。各スクリプトは起動時に `hackrf_info` を実行して
この状態を検出し、送信する前に止まる（`== devices ==` の行）。

### Windows トラブルシューティング: `HackRF not found (-5)`

```
hackrf_open() failed: HackRF not found (-5)              # 開けない（他プロセスが掴んでいる）
hackrf_open() failed: Access denied (insufficient permissions) (-1000)
```

USB 的には 2 台とも見えているのにこれが出る場合、**前回の `hackrf_transfer` が終了しきれずに
無線機を掴んだまま**になっている。Ctrl-C で中断したときにも起こり得る。

```powershell
Get-CimInstance Win32_Process -Filter "Name='hackrf_transfer.exe'" |
    Select-Object ProcessId, CreationDate, CommandLine
```

残っていた場合、`taskkill /F` は**効かないことがある**。USB 転送の途中で
`TerminateProcess` された プロセスはドライバが保留 I/O を抱えたまま回収できず、
`HasExited=True` かつ生存という状態で固まる。この状態を解く唯一の方法は
**当該 HackRF の USB を抜き差しする**こと。復帰したかどうかは `hackrf_info` で
2台ともシリアル番号まで表示されることで確認する。

なお `hwlab` 側はこれを起こさない作りにしてある（送信は有限長ファイルで自然終了するので
kill しない。[hwlab/radio/hackrf_cli.py](radio/hackrf_cli.py) 冒頭のコメント参照）。
それでも出る場合は他のツール（GNU Radio、SDR#、gqrx 等）が掴んでいないか確認する。

## 手順

### フェーズ0 — 実機なしで DSP を確認する（最初にこれ）

`LoopbackBackend` は実機と同じ IQ サンプル列を受け取り、同じ順序で同じ劣化
（経路損失・AWGN・位相回転・DCオフセット・**8bit量子化**・**ADCクリップ**）を加えて返す。
DSP チェーン全段がここで検証される。

```powershell
pytest hwlab/tests -q                                    # 95 tests、実機不要
python -m hwlab.scripts.check_link --backend loopback
python -m hwlab.scripts.calibrate_snr --backend loopback --bursts 8
```

### フェーズ0.5 — RF 経路そのものの健全性

`check_link` は「バーストが復調できたか」という yes/no を返すので、**一番知りたいときに
黙ってしまう**（`bursts detected 0/10` からは何も分からない）。その下にある2つの問いに
数値で答えるのが `check_path` で、リンクが復調不能なほど弱くても動き続ける。

```powershell
python -m hwlab.scripts.check_path --backend hackrf --repeat 8         # まずこれ（下記）
python -m hwlab.scripts.check_path --backend hackrf --noise-only       # 受信機の素性（アナログ雑音か digital グリッチか）
python -m hwlab.scripts.check_path --backend hackrf                    # TX利得掃引 → 経路利得
python -m hwlab.scripts.check_path --backend hackrf --both-directions  # 共通経路か片側かの切り分け
python -m hwlab.scripts.check_path --backend hackrf --freq-sweep       # 周波数応答（平坦か）
```

**`--repeat` を最初に回すこと。** 絶対レベルも周波数応答も「測定間でベンチが動かない」ことを
前提にしている。接触不良のコネクタはその前提を壊し、以降の数値がすべて無意味になる。
実際、緩んだ SMA を掴んだのはこのモードで、周波数掃引より速かった。

バーストではなく**無変調トーン**を送る。PAPR バックオフが無いので同じ DAC ピークでも平均電力が
約 10 dB 高く、さらに捕捉全体を1つの FFT ビンにコヒーレント積分できる。合わせてプリアンブル相関より
数十 dB 高感度で、サンプルレベルの雑音より 20 dB 低いトーンでも読める。

読み方:

| 列・モード | 判定 |
|---|---|
| `--repeat` で検出が全数に満たない | **間欠**。コネクタが接触したりしなかったりしている。他の数値は全部無意味 |
| `--repeat` の spread > 2 dB | 導体経路は 1 dB 以下に収まる。まだ不安定 |
| `transmit` 列が `-1.9 dBfs` 等でない | 送信機が動いていない。経路ではなく送信側を見る |
| `short by` が 0 付近 | 経路は設計モデル通り。`check_link` へ進む |
| `short by` が大きい | 設計動作点から何 dB 足りないかがそのまま出る |
| `f err` が数百 Hz を超える（クロック同期時） | トーンではなくスプリアスにロックしている。その行は信用しない |
| `--freq-sweep` の spread が小さい | 導体経路と**矛盾しない**（証明ではない。決定的なのは同軸を外す試験） |
| `--freq-sweep` の spread が大きい | 接触不良か空間漏れ。同軸が ANT ポートに刺さっているか確認する |
| `--noise-only` が `DIGITAL, NOT ANALOG` | 利得を下げても雑音が減らない＝サンプル破損。**まず `--clock off` で再測定**（上の節） |
| `f err` が 500Hz を超える | 2台が独立発振している。トーンのレベルは有効だがバーストは同期できない |
| `--noise-only` の peak/rms > 10x | インパルス性妨害（下の「2台の個体差」参照） |

減衰器を 10 dB 入れ替えたのに経路利得が 10 dB 動かなければ、コネクタが接触していない。

**減衰器は表記を信じない。** 実際に入っていたものが 50dB で、30dB と思い込んでいたために
20dB ぶんの「原因不明の損失」を延々と追う羽目になった。交換したら必ず測り直し、
経路利得が表記どおりの差で動くことを確認する。

### フェーズ1 — リンク確立

```powershell
# 配線: 先に減衰器を入れる。CLKOUT→CLKIN も接続する
python -m hwlab.scripts.check_link --backend hackrf --tx-gain 30
```

確認すべき出力:

| 項目 | 期待値 | 外れたときに疑うところ |
|---|---|---|
| `devices` | `2 HackRF(s) available: ...` | USB接続 / シリアル設定 / 残留プロセス（上の節） |
| `clock` | `clock signal detected` | CLKOUT有効化 / SMAケーブル |
| `bursts detected` | 全数 | 配線・周波数・利得・減衰量。まずフェーズ0.5 に戻る |
| `preamble peak` | 閾値5.5を大きく超える | 送信が届いていない |
| `RX level` peak | **120 LSB 未満** | 超えたら rx_vga を下げる（クリップは非線形歪みを撒く） |
| `RX level` rms | **1 LSB 超** | 下回ると量子化雑音が支配する。rx_vga を上げる |
| `image rejection` | 高 SNR 域で **30dB 以上** | 低 SNR では推定自体が雑音に埋もれるので判断に使わない |
| 3つの SNR 推定 | **互いに 3dB 以内** | guard だけ上振れするなら ADC 動作点の問題（下記） |

**peak に「40〜100 LSB」のような下限を課してはいけない。** SNR 掃引の下端では信号が小さいのが
正常であり、下限を課すとその領域を丸ごと捨てることになる。判定すべきは「クリップしないこと」と
「雑音が ADC をディザできていること（rms）」の2つ。

`|h|` は tx/rx 利得の設定値で決まる量なので、絶対値に固定の期待値は無い。
経路が正しいかは `check_path` の `short by` で見ること。
| `guard` と `pilot` の SNR | 差 3dB 以内 | クリップ / 帯域内スプリアス / タイミング誤り |

#### 確定した運用動作点（2026-08-18 実測）

| 項目 | 値 |
|---|---|
| 減衰器 | **50 dB**（固定 SMA パッド） |
| 送信 | …a06063c824923a5f |
| 受信 | …16bc62dc2e679ba7 |
| クロック主 | **…16bc62dc2e679ba7**（受信側。逆向きだと従側 ADC が壊れる） |
| 受信利得 | **lna 24 / vga 20**（掃引中は固定） |
| 到達 SNR | **-11.6 〜 +25.1 dB**（usable 21/24 点） |

`check_link` の結果（両端とも 10/10、exit 0）:

| tx_vga | bursts | preamble | RX peak | guard | pilot | z誤差 |
|---|---|---|---|---|---|---|
| 16 | 10/10 | 31.4x | 4 LSB | +1.98 | +2.05 | +2.02 |
| 36 | 10/10 | 118.9x | 6 LSB | +19.77 | +19.68 | +19.58 |

3つの独立した SNR 推定が **0.2 dB 以内**で一致する。これがリンクが健全であることの
最も強い証拠で、どれか1つだけずれていたら疑うべきは DSP ではなく動作点。

**rx 利得を上げてはいけない。** peak を大きくしようと lna/vga を上げると、guard 領域の雑音が
ADC の固定床（約 2.4 LSB、利得 0〜72dB でほぼ不変）に律速されて利得に追従しない一方で信号だけが
追従するため、**guard 推定だけが最大 12dB 上振れする**。掃引は guard 推定を x 軸に使うので、
SNR を過大評価したプロットになる:

| rx_gain | RX peak | guard | pilot | z誤差 |
|---|---|---|---|---|
| **44 dB（採用）** | 8 | +25.7 | +24.4 | +23.9 ← 一致 |
| 60 dB | 29 | +39.5 | +27.6 | +27.8 |
| 72 dB | 106 | +41.7 | +29.6 | +31.9 |

### フェーズ2 — 較正

```powershell
python -m hwlab.scripts.calibrate_snr --backend hackrf --attenuator-db 40 --bursts 20
```

TX利得を掃引し、**実際に出た SNR** を測って `hwlab/results/calibration.json` に書く。
クリップした点・バースト欠落した点は記録した上で `usable()` から除外される。
`-10 .. +20 dB` をカバーできていない場合は減衰量で調整する。

### フェーズ3 — スイープ

```powershell
python -m hwlab.scripts.run_sdr_sweep --checkpoint checkpoints/jscc_dml.pt --backend hackrf --episodes 15
```

エピソードのシードは `airComp/eval/snr_sweep.py` と**同一の式**
（`int(snr_db*10_000) + 1_000_000`）を使う。同じ SNR 点でシミュレーション実行と
まったく同じ pool / private values が引かれるので、両者の曲線を直接重ねられる。

#### 本番スイープの実測（2026-08-18、7点 × 15 エピソード）

チェックポイント `checkpoints/jscc_dml.pt`（935 サンプル、final loss 0.2312）。
結果は `hwlab/results/sdr_sweep.json`。

| 要求 SNR | 実測 SNR | 差 | 合意率 | バースト欠落 |
|---:|---:|---:|---:|---:|
| −10 | −9.59 | 0.41 | 0.87 | 0% |
| −5 | −4.33 | 0.67 | 1.00 | 0% |
| 0 | −0.67 | 0.67 | 1.00 | 0% |
| +5 | **+6.36** | **1.36** ⚠ | 1.00 | 0% |
| +10 | +9.83 | 0.17 | 1.00 | 0% |
| +15 | +14.70 | 0.30 | 0.93 | 0% |
| +20 | +20.32 | 0.32 | 1.00 | 0% |

**全点でバースト欠落 0%、RX 飽和警告なし、リトライ 0。** フェーズ1で確定した動作点
（減衰器 50 dB / lna 24 / vga 20 / クロック主 …2e679ba7）がスイープ全域で維持できている。

⚠ **+5 dB 点だけ要求と実測が 1.36 dB ずれ、1 dB 基準を外している。** 較正テーブルの
粒度による（他6点は 0.17〜0.67 dB）。図は実測 SNR の位置に描くので結論には影響しないが、
この付近の較正点を増やせば解消する。

**シミュレーションとの一致**（`results/sweep.json` の semantic と対シード比較）:
平均差 +0.014、最大 |差| 0.07。実機は 15 エピソードなので 1 エピソード = 0.067 であり、
**全点のズレが 1 エピソード以内**。系統的オフセットは無く、DSP チェーンは
`airComp/channel/analog.py` を正しく再現している。この判定基準が下の「検証の考え方」。

**オーバーヘッド**: data 8 シンボルに対し overhead 863（プリアンブル 511 / パイロット 32 /
ガード 320）、合計 871、バースト長 8.71 ms、占有帯域 135 kHz。帯域の主張をするときは
必ずこれを併記する。

### 図の出し方

シミュレーションと実機を1枚に重ねる。`plot` は複数ファイルを受け取り、`channel` のような
非系列キーは形で判別して無視し、`_hw` を含む系列を破線で描く。x 軸には
`measured_snr_db_mean` があればそれを使う（**要求 SNR ではない** — 上表のとおり最大 1.36 dB
ずれており、それは検出したい sim-実機オフセットと同オーダー）。

```powershell
foreach ($m in "agreement_rate","avg_social_welfare","avg_pareto_efficiency") {
    python evaluate.py plot --results results/sweep.json hwlab/results/sdr_sweep.json `
        --metric $m --out "results/${m}_vs_snr.png"
}
```

結果の読み方は [docs/results.md](../docs/results.md)、先行研究に対する位置づけは
[docs/related_work.md](../docs/related_work.md)。

---

## 検証の考え方

**最重要の自己検証**: 実機の劣化曲線を**純シミュレーションの曲線に重ねる**。DSP が正しければ
誤差範囲で一致するはず。系統的なオフセットが出たらバグの証拠であり、疑う順序は

1. **SNR の定義**（[dsp/mapping.py](dsp/mapping.py)） — 実数成分あたりで定義している。
   どこかで複素シンボルを 1/√2 で正規化すると 3dB ずれ、曲線全体が平行移動する
2. **利得較正** — RX利得が掃引中に動いていないか、クリップしていないか

`hwlab/tests/test_mapping.py` がこの規約を `AnalogAWGNChannel` に対して機械的に固定している。
ここが落ちたら実機とシミュレーションの比較は無意味になる。

---

## 設計上の判断（変更前に読むこと）

**SNR は実数成分あたりで定義する。** `analog.py` が各実数成分に σ²=1/SNR を加えるため、
`z` を複素シンボルに詰めるときも**再正規化しない**（1シンボルの電力は 1 ではなく 2）。
[dsp/mapping.py](dsp/mapping.py) が唯一の定義場所。

**等化後の `ẑ` を `‖ẑ‖=√k` に再正規化してはいけない。** `SemanticDecoder` は
「単位電力の信号＋雑音」で訓練されている。再正規化すると信号と一緒に雑音も
スケールされ、測ろうとしている SNR 関係そのものが壊れる。プリアンブルから推定した
複素ゲイン `h` で割るだけで正しい絶対スケールが戻る。

**チャネル推定はパイロットではなくプリアンブルから行う。** プリアンブルは同期用に
どのみち送っており 16 倍長いので、約 12dB 分の平均化が無料で得られる。パイロット 32 個
だけだと -10dB SNR で `|h|` の推定値が 25dB も振れ、報告 SNR を壊し等化後の雑音を増やす。
パイロットを推定から外すことで、その残差が**独立した検証量**にもなる。

**RX利得は掃引中固定。** SNR は TX利得と減衰器だけで動かす。HackRF に AGC が無いのは
本実験には好都合（AGC があると受信信号が勝手に再スケールされ、電力正規化が壊れる）。

**単一キャリア。OFDM は使わない。** 複素8シンボルでは OFDM の利点が無く、高い PAPR が
8bit コンバータの数少ないヘッドルームを食うだけ。

**バースト欠落は雑音のみを届ける。** 受信できなかったということは受信機には雑音しか
無かったということ。再送で成功したふりをするのでも、エピソードごと捨てるのでもなく、
物理的に正しい入力をデコーダに渡し、欠落率を記録する。

**オーバーヘッドは正直に計上する。** 1 メッセージ 8 シンボルに対し同期・パイロットの
オーバーヘッドは 863 シンボル。`SDRAnalogChannel.payload_accounting()` が返す。
CLAUDE.md の「bits vs symbols は apples-to-apples ではない」に従い、帯域の主張を
ペイロードだけから行わないこと。

---

## 8bit ADC/DAC について

実効 ENOB を考慮しても量子化ノイズ床は本実験の掃引範囲（-10〜+20dB）よりはるかに下にある。
`hwlab/tests/test_loopback_e2e.py::test_eight_bit_quantization_is_not_the_limit_in_range`
が、掃引上端で量子化を切っても SNR がほとんど動かないことを確認している。

効いてくるのは (a) 掃引を 30dB 以上に伸ばす場合、(b) 高PAPR の OFDM を使う場合のみ。

---

## 却下した選択肢（記録）

**統合トランシーバIC（CC2500 / CC1101 / nRF24L01 / 自社 STD-302Z）** — アナログ `z` 伝送は
**原理的に不可能**。非同期シリアルモードにしてもバイパスされるのは*パケット層*であって
*モデム*ではない。GDO0 等はデジタルピンで、入力は0/1、出力はスライサ後のハードデシジョン。
I/Q には一切アクセスできない。

なお「市販の RF-IC で可能か」の答えは Yes で、**HackRF One の IF トランシーバ MAX2837 が
まさにそのクラスの IC**（アナログI/Qベースバンドを持つ）。必要なものは既に基板の形で
手元にある。

**アナログFM音声リンク（自社 WA-TX-03S 等）** — AWGN モデルを壊すため却下:
1. **コンパンダ**（70dBダイナミックレンジ）— 信号依存の 2:1 対数圧伸が `y = z + n` の
   線形性を根本から壊し、較正で除去できない
2. **FMスレッショルド効果** — 低CNRで急落＝**崖**になり、本研究の主張と**逆の結果**を出す
3. プリエンファシス/デエンファシス＋FM固有の三角雑音特性で雑音が強く有色化
4. サウンドカードのAC結合が DC 近傍を除去し `z` の平均成分が復元不能
5. スケルチが効くと erasure チャネルになり noisy チャネルでなくなる

---

## ファイル構成

```
dsp/        純numpy。実機依存なし、全段ユニットテスト済み
  mapping.py    ★SNR規約の唯一の定義場所
  framing.py    バースト構造（ガード｜プリアンブル｜パイロット｜データ｜ガード）
  pulse.py      RRC整形、デジタルIF、8bit変換
  sync.py       プリアンブル相関によるフレーム検出
  equalize.py   LS推定 / ZF等化
  measure.py    雑音・SNR実測、ADCレベル警告
  burst.py      上記を束ねる BurstCodec（modulate / demodulate）
radio/      SDRBackend 実装
  loopback.py   実機の代わり。同じ劣化を同じ順序で加える
  hackrf_cli.py hackrf_transfer を subprocess で駆動
  clock.py      CLKOUT/CLKIN 確認
channel/
  sdr_analog.py AnalogAWGNChannel のドロップイン代替
  calibration.py 利得↔実測SNR の対応表
agent.py    HardwareSemanticAgent
scripts/    check_link / calibrate_snr / run_sdr_sweep
tests/      全て実機不要
```
