# Отчёт: Поддержка и оптимизация инференса для CPU без AVX2/FMA (Intel Xeon E5-2650 v2, AVX 1.0 + F16C + SSSE3/SSE4.1)

## 1. Цель и контекст

Адаптация и глубокая оптимизация CPU-путей движка **ninfer-v100** для процессоров без инструкций `AVX2` и `FMA3` (целевой стенд Intel Ivy Bridge-EP с `AVX 1.0 + F16C + SSSE3 + SSE4.1`):
- **CPU**: Intel Xeon E5-2650 v2 @ 2.60 GHz (1 сокет, 8 физических ядер / 16 логических потоков, L3 20 MiB; флаги `ssse3 sse4_1 sse4_2 avx f16c`, без `avx2` и `fma`)
- **RAM**: 31 GiB (физическая память хоста)
- **GPU**: Tesla V100-PCIE-32GB (`sm_70`, PCIe 3.0 x16, CUDA 12.8; во время финального прогона тестов CPU/Vision GPU 1 была занята рабочим контейнером `coder390-llama-server` на 32,124 MiB из 32,768 MiB; GPU 0 — `GTX 1060 3GB`, не собирается под `sm_70`)

Решаемые задачи:
1. Устранить `SIGILL` при `--expert-cpu-arithmetic ggml` на CPU без `AVX2/FMA` через опцию CMake `NINFER_GGML_CPU_AVX2` (`ON` по умолчанию, `OFF` для `AVX 1.0 + F16C`) и раннюю проверку на старте (`Executor::Impl`).
2. Реализовать быстрый `AVX 1.0 + F16C + SSSE3 + SSE4.1` путь для точной CPU-арифметики GGUF-экспертов (`GgufCpuBackend::Avx`, побитово эквивалентный GPU и `Scalar`), для режима `ggml` (`quantize_row_q8_K`, прямой `BF16 -> Q8`, батчевые `*_rows` кернелы), и для нативных MoE-экспертов (`CpuExpertBackend::Avx`).
3. Оптимизировать многопоточный runtime CPU-миссов в `Qwen3.8-Flash-Next` (`src/models/qwen4_exp/expert_misses.cu`).
4. Оптимизировать CPU-энкодеры зрения для **Gemma 4** и **Qwen3.5** на `AVX 1.0` (и `AVX2+FMA`).
5. Исправить методику микробенчмарка `ninfer_gguf_expert_cpu_bench` (заменив создание/join `std::thread` на каждой итерации на постоянный spin-barrier пул) и обновить документацию строго по проверенным замерам.

---

## 2. Реализованные изменения и оптимизации

### 2.1. Точная CPU-арифметика GGUF-экспертов (`GgufCpuBackend::Avx`)
[`src/ops/moe_experts/gguf_expert_cpu.h`](src/ops/moe_experts/gguf_expert_cpu.h), [`src/ops/moe_experts/gguf_expert_cpu.cpp`](src/ops/moe_experts/gguf_expert_cpu.cpp), [`src/ops/moe_experts/gguf_expert_cpu_avx.cpp`](src/ops/moe_experts/gguf_expert_cpu_avx.cpp):
- **ISA и защита**: автовыбор `Avx` при наличии `avx`, `f16c`, `ssse3`, `sse4_1` и ОС-поддержки `XCR0`, явный отказ (`std::invalid_argument`) при запросе неподдерживаемого бэкенда.
- **IEEE-754 побитовая точность (`-ffp-contract=off`)**:
  - Векторные эмуляторы `fma4` / `fma8` в AVX 1.0 регистрах (Boldo–Melquiond `TwoSum` + round-to-odd) для `IQ2_S` (`halves == true`) и `swiglu_avx`.
  - Векторизованный `prepare_bf16_avx`: за один проход проверяет конечность входных BF16-слов через битовую маску `(raw & 0x7F80) == 0x7F80` (не проглатывая `NaN`), находит максимум модуля по блокам 32/256, вычисляет масштаб `d` с аппаратным F16C `_mm_cvtps_ph` и округляет `round_ties_away` (`q + copysign(0.49999997F, q)` -> `_mm256_cvttps_epi32`) в int8-коды.
  - Предвычисленная таблица `table_iq4_abs` (`iq4_values_and_abs`) убирает runtime `_mm_sign_epi8(v, v)` при вычислении `IQ4_NL` / `IQ4_XS`.
  - Замена горизонтальных сумм `_mm_hadd_epi32` (12 инструкций суммарно с высоким латенси на Ivy Bridge) на схему `hsum4_epi32` (`unpacklo/hi` + `paddd`).
  - Быстрый потоковый путь для `T == 1` в `iq_rows` и `block32_rows`: исключает 576-байтный буфер `Group` на стеке при одиночном декодировании, сохраняя побитовый порядок `butterfly8`.
  - Векторизованная редукция dot-to-float в `q2_rows` и оптимизированный путь `live == 4` в `live_floats`.
  - Быстрый не-Деккер `swiglu_fast_avx` / `gguf_cpu_swiglu_bf16_fast` для режима `--expert-cpu-arithmetic ggml` и раздельные потокобезопасные методы `prepare_bf16_start` / `prepare_bf16_token` / `copy_token_from`.

### 2.2. Кернелы `ggml` без AVX2 (`NINFER_GGML_CPU_AVX2=OFF`) и многотокенные матрицы
[`CMakeLists.txt`](CMakeLists.txt), [`src/ops/basic_sources.cmake`](src/ops/basic_sources.cmake), [`third_party/ggml-quants/`](third_party/ggml-quants/), [`src/ops/moe_experts/gguf_expert_cpu_ggml.h`](src/ops/moe_experts/gguf_expert_cpu_ggml.h), [`src/ops/moe_experts/gguf_expert_cpu_ggml.cpp`](src/ops/moe_experts/gguf_expert_cpu_ggml.cpp), [`src/models/qwen4_exp/executor.cpp`](src/models/qwen4_exp/executor.cpp):
- **Компиляция и runtime-проверка**: `NINFER_GGML_CPU_AVX2=ON` (по умолчанию) собирает `arch/x86/quants.c` с `-mavx2;-mfma;-mf16c` (`/arch:AVX2`), а `OFF` — с `-mavx;-mf16c` (`/arch:AVX`). `gguf_ggml_cpu_available()` проверяет требуемый набор ISA; `Executor::Impl` выдаёт диагностическую ошибку на старте без `SIGILL`, а `ninfer_gguf_expert_cpu_ggml_test` корректно пропускается (`77`, `Skipped`).
- **Аппаратный F16C на запись, LUT на чтение (`simd-mappings.h`)**:
  - Под `#elif defined(__F16C__)` активирован `GGML_CPU_FP32_TO_FP16` (`_cvtss_sh`).
  - Для скалярного `GGML_CPU_FP16_TO_FP32` осознанно **сохранён** LUT `ggml_table_f32_f16` (256 KiB): A/B-эксперимент на E5-2650 v2 показал, что переключение скалярного `FP16->FP32` на `_cvtsh_ss` вызывает переходы GPR<->XMM в горячих циклах и даёт регрессию **10–21%** на `IQ4_NL` и `Q8_0` при `T=1`.
- **AVX1/SSSE3/SSE4.1 `quantize_row_q8_K` и прямой `BF16 -> Q8`** (`quants.h`, `arch/x86/quants.c`, `gguf_expert_cpu_ggml.cpp`):
  - На ветке AVX без AVX2 `quantize_row_q8_K` больше не падает на скалярный `_ref`: реализована векторизованная квантизация блоками по 256 (с `bsums[16]` на `_mm_madd_epi16`).
  - Добавлены `ggml_quantize_bf16_q8_0` и `ggml_quantize_bf16_q8_K` с прямым AVX-распаковыванием BF16 -> FP32 и квантованием без промежуточного буфера `scratch_`.
- **Батчевые многотокенные матричные кернелы `*_rows`**:
  - Для `iq2_xxs`, `iq2_xs`, `iq2_s`, `iq3_xxs`, `iq3_s`, `iq4_xs` (`q8_K`, группы до `tc = 8` токенов на одну распаковку субблока весов) и `iq4_nl`, `q8_0` (`q8_0`, 2-блочные `__m256` с `tc = 4/2` и делегированием `rem == 1` обратно в однотокенный `ggml_vec_dot_*` без потерь на `T=1`).

### 2.3. Многопоточное исполнение промахов кэша в Flash-Next (`expert_misses.cu`)
[`src/models/qwen4_exp/expert_misses.cu`](src/models/qwen4_exp/expert_misses.cu):
- Привязка потоков `SpinPool` (worker threads и service thread) к **различным физическим ядрам** Linux (`physical_cpus()` через sysfs `thread_siblings_list`) при `phys.size() >= threads`, устраняя конкуренцию SMT-собратьев.
- Дедупликация и распараллеливание входного BF16-квантования по токенам (`prepare_bf16_start` / `prepare_bf16_token` через `pool->run`), исключение копирования буферов активации (`job.use_shared_input`), совместное переиспользование квантованного входа между `gate` и `up` при совпадающем точном формате/раскладке (`same_up_act`), а также параллельное вычисление `SwiGLU` + промежуточного квантования (`middle.prepare_bf16_token`, с `gguf_cpu_swiglu_bf16_fast` при `--expert-cpu-arithmetic ggml`).

### 2.4. Нативные MoE CPU-эксперты (`CpuExpertBackend::Avx`)
[`src/ops/moe_experts/cpu_projection.h`](src/ops/moe_experts/cpu_projection.h), [`src/ops/moe_experts/moe_expert_cpu.cpp`](src/ops/moe_experts/moe_expert_cpu.cpp), [`src/ops/moe_experts/moe_expert_cpu_avx.cpp`](src/ops/moe_experts/moe_expert_cpu_avx.cpp):
- Полностью переписан `moe_expert_cpu_avx.cpp`:
  - Векторные SIMD-декодеры в регистрах для `Q2_G64_FP16`, `Q4_G64_FP16`, `Q5_G64_FP16`, `Q6_G64_FP16` (группы по 64 элемента без буфера `codes[32]` на стеке), `Q8_G32_FP16` и `BF16` с поддержкой обеих раскладок (`RowSplit` и `RowSplitPanel`), режимов `a8=true` (SSSE3 `maddubs`) и `a8=false` (FP32) и переиспользованием весов по чанкам до 16 токенов при `T > 1`.
  - `prepare_input_avx`: слитный проход BF16 -> FP32, проверка конечности `(raw & 0x7F80) == 0x7F80` и квантование `a8` (`_mm256_cvtps_epi32` + `_mm_packs_epi32/16`).
  - `swiglu_quantize_avx`: векторный `exp256_ps` + `SiLU * up` на `__m256` со слитной проверкой конечности и `a8`-квантованием.

### 2.5. Энкодеры зрения на CPU (Gemma 4 и Qwen3.5)
- **Gemma 4 CPU Vision** ([`src/models/gemma4/vision_cpu_kernels.h`](src/models/gemma4/vision_cpu_kernels.h), [`vision_cpu_kernels.cpp`](src/models/gemma4/vision_cpu_kernels.cpp), [`vision_cpu_kernels_avx.cpp`](src/models/gemma4/vision_cpu_kernels_avx.cpp), [`vision_cpu_kernels_avx2.cpp`](src/models/gemma4/vision_cpu_kernels_avx2.cpp), [`vision_cpu.cpp`](src/models/gemma4/vision_cpu.cpp)):
  - AVX 1.0 микроядро `gemm` расширено с `4×16` до **`6×16`** (`12` YMM-аккумуляторов в 15 регистрах без спиллов) с 64-байтным выравниванием `panel` и бесхвостовыми `4×16`/`2×16`/`1×16` остатками; в `linear()` упаковка `x` (`pack_panel`) вынесена из рабочих потоков и параллельно выполняется 1 раз по панелям, а строки `W` распределяются тайлами кратно `12` (`LCM(4, 6)` -> без частичных тайлов ни на AVX 1.0, ни на AVX2).
  - Векторизованы `rms_accum_sq` (в точном FP64 через `_mm256_cvtps_pd`), `scale_mul_row` и `add_scaled_row`.
- **Qwen3.5 CPU Vision** ([`src/models/qwen3_5/execution/vision_cpu_kernels.h`](src/models/qwen3_5/execution/vision_cpu_kernels.h), [`vision_cpu_kernels.cpp`](src/models/qwen3_5/execution/vision_cpu_kernels.cpp), [`vision_cpu_kernels_x86.inl`](src/models/qwen3_5/execution/vision_cpu_kernels_x86.inl), [`vision_cpu_kernels_avx.cpp`](src/models/qwen3_5/execution/vision_cpu_kernels_avx.cpp), [`vision_cpu_kernels_avx2.cpp`](src/models/qwen3_5/execution/vision_cpu_kernels_avx2.cpp), [`vision_cpu.cpp`](src/models/qwen3_5/execution/vision_cpu.cpp), [`execution_sources.cmake`](src/models/qwen3_5/execution_sources.cmake)):
  - Векторизованы с нуля на `AVX 1.0` и `AVX2+FMA` (через общий шаблонный `.inl` с `-ffp-contract=off`): `pack_panel`, `gemm_packed_bias` (`6×16`), `layer_norm_cols`, `add_into`, `gelu_tanh`, `rope_pair`, и 3-проходный `segment_attention_block` (`4×16` QK^T в регистрах + `exp_nonpositive` softmax + PV).
  - Границы параллельных срезов в 1D-проходах выровнены по 8 элементов, сохраняя строгий контракт **побитовой детерминированности `single == parallel`** при любом числе потоков.

---

## 3. Проверенные замеры на `192.168.50.38` (Intel Xeon E5-2650 v2)

### 3.1. Исправление бенчмарка `ninfer_gguf_expert_cpu_bench` (`PersistentPool`)
В [`bench/ops/gguf_expert_cpu_bench.cpp`](bench/ops/gguf_expert_cpu_bench.cpp) создание `std::thread` на каждую итерацию (~`0.3–0.5 мс` оверхеда ОС, искажавшего `thr=6` vs `thr=8`) заменено на постоянный пул с spin-barrier на `_mm_pause()`. В чистых замерах с `PersistentPool` **`thr=8` (8 физических ядер) быстрее `thr=6` на большинстве форматов** (например, на `IQ3_S T=1`: `0.0696`/`0.0743 ms` против `0.0861`/`0.0986 ms`), тогда как автоматическое значение по умолчанию в движке (`6 = логические/2 - 2`) остаётся консервативным выбором для совместной работы с GPU-потоками serving.

### 3.2. GGUF CPU-эксперты: `PersistentPool` до vs после (`ninfer_gguf_expert_cpu_bench`, `NINFER_GGML_CPU_AVX2=OFF`, `iters=15`)
Сравнение медианного времени одного матричного умножения (база `c3a42018` перемеряна с `PersistentPool` до изменений кернелов vs финальная сборка):

| Формат | Форма | `T` | `thr` | `avx` (база -> финал) | `ggml` без AVX2 (база -> финал) | Ускор. к `scalar` (финал) |
|---|---|---:|---:|---:|---:|---:|
| `IQ3_S` | `640x2560` | 1 | 1 | `0.5547 -> 0.5754 ms` | `0.4340 -> 0.5039 ms` | `19.75x` / `22.56x` |
| `IQ3_S` | `640x2560` | 1 | 8 | `0.0730 -> 0.0743 ms` | `0.0591 -> 0.0696 ms` | `15.38x` / `16.41x` |
| `IQ3_S` | `640x2560` | 8 | 1 | `1.4821 -> 1.3649 ms` | `3.5969 -> 1.1047 ms` (**3.26x**) | `19.26x` / **`23.80x`** |
| `IQ3_S` | `640x2560` | 8 | 8 | `0.2743 -> 0.1760 ms` (**1.56x**) | `0.4538 -> 0.1547 ms` (**2.93x**) | `18.68x` / **`21.25x`** |
| `IQ4_NL` | `640x2560` | 1 | 1 | `0.3144 -> 0.2172 ms` (**1.45x**) | `0.1943 -> 0.1926 ms` | `18.43x` / `20.79x` |
| `IQ4_NL` | `640x2560` | 1 | 8 | `0.0407 -> 0.0288 ms` (**1.41x**) | `— -> 0.0339 ms` | **`17.40x`** / `14.79x` |
| `IQ4_NL` | `640x2560` | 8 | 1 | `1.2112 -> 1.0915 ms` | `1.5869 -> 1.1897 ms` (**1.33x**) | **`20.20x`** / `18.53x` |
| `IQ4_NL` | `640x2560` | 8 | 8 | `0.1819 -> 0.1462 ms` (**1.24x**) | `— -> 0.1576 ms` | **`18.91x`** / `17.53x` |
| `Q8_0` | `640x2560` | 1 | 1 | `0.2907 -> 0.1787 ms` (**1.63x**) | `0.1680 ms` | `17.34x` / `18.45x` |
| `Q8_0` | `640x2560` | 1 | 8 | `0.0240 ms` | `0.0316 ms` | **`16.21x`** / `12.30x` |
| `Q8_0` | `2560x640` | 1 | 1 | `0.3901 -> 0.2190 ms` (**1.78x**) | `0.1755 -> 0.1706 ms` | `14.42x` / `18.51x` |
| `IQ2_S` | `640x2560` | 1 | 1 | `0.5720 -> 0.4681 ms` (**1.22x**) | `0.3342 ms` | `19.13x` / `26.80x` |
| `IQ2_S` | `640x2560` | 8 | 1 | `2.3687 -> 2.1237 ms` | `2.7238 -> 1.0190 ms` (**2.67x**) | `24.20x` / **`50.43x`** |
| `IQ2_S` | `640x2560` | 8 | 8 | `0.2843 ms` | `0.3600 -> 0.1448 ms` (**2.49x**) | `22.79x` / **`44.74x`** |
| `Q2_0` | `640x2560` | 1 | 8 | `0.0357 ms` | — | `26.71x` |
| `SwiGLU` | `n=5120` | — | — | `0.1067 -> 0.0963 ms` | — | **`32.45x`** |

### 3.3. Квантование входа (`prepare_bf16`), Native MoE CPU (`moe_expert_cpu`) и CPU Vision (`/tmp/baseline_eval`)
| Компонент / Кейс | До оптимизации (база) | После оптимизации (финал) | Ускорение |
|---|---|---|---|
| `prepare_bf16` (`k=2560, T=1`): `exact_iq3/4` | `0.0163 ms` | `0.0045 ms` | **`3.62x`** (в пике до `0.0032 ms` / `5.1x`) |
| `prepare_bf16` (`k=2560, T=8`): `exact_iq3/4` | `0.1250 ms` | `0.0256 ms` | **`4.88x`** |
| `prepare_bf16` (`k=2560, T=1`): `ggml_q8k` / `ggml_q80` | `0.0068` / `0.0148 ms` | `0.0031` / `0.0029 ms` | **`2.19x`** / **`5.10x`** |
| `prepare_bf16` (`k=2560, T=8`): `ggml_q8k` / `ggml_q80` | `0.0478` / `0.1159 ms` | `0.0152` / `0.0136 ms` | **`3.14x`** / **`8.52x`** |
| `moe_expert_cpu` (`Q4_G64, a8=1, T=1`, `Avx`) | `9.3887 ms` (`0.96x` к scalar) | `0.7026 ms` (`13.06x` к scalar) | **`13.36x` (vs старый AVX)** |
| `moe_expert_cpu` (`Q2_G64, a8=1, T=1`, `Avx`) | `7.8638 ms` (`1.02x` к scalar) | `0.7065 ms` (`11.38x` к scalar) | **`11.13x` (vs старый AVX)** |
| `moe_expert_cpu` (`Q8_G32, a8=1, T=1`, `Avx`) | `1.2701 ms` (`1.47x` к scalar) | `0.5740 ms` (`3.25x` к scalar) | **`2.21x` (vs старый AVX)** |
| `moe_expert_cpu` (`Q4_G64, a8=0, T=1`, `Avx`) | не векторизован в декоде (`~9 ms`) | `1.2387 ms` (`10.07x` к scalar) | **`~7.3x`** |
| `Gemma 4 Vision`: `gemm 1152x1152x256` (`avx` `4x16 -> 6x16`) | `19.2894 ms` | `17.5564 ms` (`16.26x` к portable) | `+9.9%` (~40 GFLOPS, ~96% пика Ivy Bridge) |
| `Gemma 4 Vision`: `gemm 4320x1152x256` (`avx` `4x16 -> 6x16`) | `76.3242 ms` | `67.5055 ms` (`15.91x` к portable) | `+13.1%` |
| `Qwen3.5 Vision CPU` (`2L, 256 patches`, `thr=1`) | `4100.9523 ms` (scalar) | `606.8229 ms` (AVX 1.0) | **`6.76x`** |
| `Qwen3.5 Vision CPU` (`2L, 256 patches`, `thr=6`) | `273.9930 ms` (scalar) | `122.4711 ms` (AVX 1.0) | **`2.24x`** |
| `Qwen3.5 Vision CPU` (`2L, 256 patches`, `thr=8`) | `242.6221 ms` (scalar) | `94.4602 ms` (AVX 1.0) | **`2.57x`** |

### 3.4. Тесты и сборки на `192.168.50.38` (финальная ревизия)
1. **`build-avx` (`-DNINFER_GGML_CPU_AVX2=OFF -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=Release`)**:
   - Сборка `ninfer_tests`, `ninfer_benches` и `expert_misses.cu` — без ошибок.
   - `ctest -R 'gguf_expert_cpu|moe_expert_cpu|vision_cpu|qwen4_exp_config'`: **6/7 Passed, 1 Skipped** (`ninfer_qwen3_5_vision_cpu_real_test` пропущен без real-модели), **0 Failed**:
     - `ninfer_qwen3_5_vision_cpu_test` (0.15s) — проверка побитового равенства `single == parallel` и точности против FP64-оракула
     - `ninfer_qwen4_exp_config_test` (0.11s)
     - `ninfer_gemma4_vision_cpu_kernels_test` (0.29s) — включая тесты `gemm_packed`, `rms_accum_sq`, `scale_mul_row`, `add_scaled_row`
     - `ninfer_moe_expert_cpu_test` (2.70s) — все форматы Q2/Q4/Q5/Q6/Q8/BF16, `a8=false/true`, `RowSplit/RowSplitPanel`, отмена/восстановление и сравнение с FP64
     - `ninfer_gguf_expert_cpu_test` (4.81s) — побитовое совпадение `Avx` и `Scalar` (включая `prepare_bf16` и `swiglu`)
     - `ninfer_gguf_expert_cpu_ggml_test` (1.14s) — проверка квантования и батчевых `ggml` кернелов против FP64-оракула
2. **`build-avx2` (`-DNINFER_GGML_CPU_AVX2=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=Release`)**:
   - Полная компиляция и линковка `ninfer_tests` (включая `arch/x86/quants.c` AVX2-веток, `gemma4`, `qwen3_5` AVX2 и `expert_misses.cu`) — без ошибок.
   - `ctest -R 'gguf_expert_cpu|moe_expert_cpu|vision_cpu|qwen4_exp_config'`: **5/7 Passed, 2 Skipped** (`ninfer_qwen3_5_vision_cpu_real_test` и `ninfer_gguf_expert_cpu_ggml_test` корректно пропущен на AVX-only хосте с кодом `77`), **0 Failed**.
3. **Ограничения финальной верификации на стенде**:
   - Код `AVX2=ON` (`vision_cpu_kernels_avx2` в `qwen3_5`/`gemma4` и `#else`-ветки в `arch/x86/quants.c`) проверен компиляцией в `build-avx2`, но физически на AVX2-процессоре не запускался (CPU стенда не имеет AVX2/FMA).
   - Сквозной прогон `Flash-Next` с GPU+CPU (`expert_misses.cu`, `ninfer_expert_delivery_test`, `ninfer_qwen4_exp_*`) в финальной итерации на `192.168.50.38` не выполнялся из-за ограничения 31 GiB host RAM и занятости `GPU 1` (`coder390-llama-server`: 32,124 / 32,768 MiB VRAM); `expert_misses.cu` проверен компиляцией CUDA 12.8 `sm_70`.

---

## 4. Источники
1. [Intel Xeon Processor E5-2650 v2 Specifications (Intel ARK)](https://www.intel.com/content/www/us/en/products/sku/75789/intel-xeon-processor-e52650-v2-20m-cache-2-60-ghz/specifications.html)
2. [Intel Intrinsics Guide (AVX, F16C, SSSE3, SSE4.1)](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
3. [Agner Fog — Optimizing software in C++ / The microarchitecture of Intel CPUs (Ivy Bridge)](https://www.agner.org/optimize/microarchitecture.pdf)
4. [llama.cpp — ggml-cpu x86 quants](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cpu/arch/x86)
5. Sylvie Boldo, Guillaume Melquiond — *Emulation of a FMA and Correctly Rounded Sums: Proved Algorithms Using Rounding to Odd*, IEEE Transactions on Computers, 57(4):462–471, 2008.
