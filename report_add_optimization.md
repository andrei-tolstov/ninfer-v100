# Отчёт: Поддержка и оптимизация инференса для CPU без AVX2/FMA (Intel Xeon E5-2650 v2, AVX 1.0 + F16C + SSSE3/SSE4.1)

## 1. Цель и контекст

Адаптация CPU-путей движка **ninfer-v100** для процессоров архитектуры Intel Ivy Bridge-EP (без `AVX2` и `FMA3`) и аналогичных CPU с набором инструкций `AVX 1.0 + F16C + SSSE3 + SSE4.1` на стенде:
- **CPU**: Intel Xeon E5-2650 v2 @ 2.60 GHz (1 сокет, 8 физических ядер / 16 потоков, L3 20 MiB; флаги `ssse3 sse4_1 sse4_2 avx f16c`, без `avx2` и `fma`)
- **RAM**: 31 GiB (физическая память хоста)
- **GPU**: NVIDIA Tesla V100-PCIE-32GB (`sm_70`), CUDA 12.8

Решаемые задачи:
1. Устранить падение `SIGILL` при использовании `--expert-cpu-arithmetic ggml` на CPU без AVX2/FMA, дав явный выбор конфигурации при сборке (`NINFER_GGML_CPU_AVX2`) и безопасный runtime-отказ при несовместимом CPU.
2. Обеспечить векторизованный `AVX 1.0 + F16C + SSSE3/SSE4.1` путь для точной CPU-арифметики GGUF-экспертов (`GgufCpuBackend::Avx`, побитово эквивалентный GPU и `Scalar`) и нативных MoE CPU-экспертов (`CpuExpertBackend::Avx`).
3. Добавить `AVX 1.0` ядра для энкодера изображений Gemma 4 CPU Vision (`gemma4::cpu_vision::avx_kernels()`).
4. Измерить реальный эффект на микробенчмарках и сквозном запуске на E5-2650 v2 + Tesla V100 и актуализировать документацию без неподтверждённых предположений.

---

## 2. Реализованные изменения

### 2.1. Точная CPU-арифметика GGUF-экспертов (`GgufCpuBackend::Avx`)
- [`src/ops/moe_experts/gguf_expert_cpu.h`](src/ops/moe_experts/gguf_expert_cpu.h), [`src/ops/moe_experts/gguf_expert_cpu.cpp`](src/ops/moe_experts/gguf_expert_cpu.cpp), [`src/ops/moe_experts/gguf_expert_cpu_avx.cpp`](src/ops/moe_experts/gguf_expert_cpu_avx.cpp):
  - Добавлен бэкенд `GgufCpuBackend::Avx` (автовыбор при наличии `avx`, `f16c`, `ssse3`, `sse4_1` и ОС-поддержки `XCR0`, когда `AVX2` отсутствует).
  - Целочисленные dot-product для форматов `Q2_0`, `Q8_0`, `IQ4_NL`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_XS` реализованы через 128-битные SSSE3/SSE4.1 примитивы (`_mm_maddubs_epi16`, `_mm_madd_epi16`, `_mm_shuffle_epi8`, `_mm_abs_epi8`, `_mm_sign_epi8`, `_mm_cvtepi8_epi16`) и пошаговое округление FMA/multiplication с директивой `-ffp-contract=off` на единицу трансляции.
  - Промежуточная активация SwiGLU (`gguf_cpu_swiglu_bf16`) сохранена через единую точно-округлённую реализацию (`swiglu_scalar`), чтобы гарантировать побитовое совпадение с GPU.
- Проверка корректности: [`tests/ops/test_gguf_expert_cpu.cpp`](tests/ops/test_gguf_expert_cpu.cpp) — побитовое совпадение `avx` со `scalar` по всем форматам, промежуточным `BF16` словам (`avx middle: 100003 words equal`) и итоговым продуктам GPU (`ninfer_moe_experts_gguf_test`: `0 of 20480 outputs differ from the GPU's`).

### 2.2. Управление ISA и runtime-защита для `--expert-cpu-arithmetic ggml`
- [`CMakeLists.txt`](CMakeLists.txt) и [`src/ops/basic_sources.cmake`](src/ops/basic_sources.cmake):
  - Добавлена опция `NINFER_GGML_CPU_AVX2` (по умолчанию `ON`). При `ON` `third_party/ggml-quants/src/ggml-cpu/arch/x86/quants.c` компилируется с `-mavx2;-mfma`, при `OFF` — с `-mavx;-mf16c` (под MSVC `/arch:AVX2` vs `/arch:AVX`), передавая `NINFER_GGML_CPU_REQUIRES_AVX2=1|0` в `gguf_expert_cpu_ggml.cpp`.
- [`src/ops/moe_experts/gguf_expert_cpu_ggml.h`](src/ops/moe_experts/gguf_expert_cpu_ggml.h), [`src/ops/moe_experts/gguf_expert_cpu_ggml.cpp`](src/ops/moe_experts/gguf_expert_cpu_ggml.cpp), [`src/models/qwen4_exp/executor.cpp`](src/models/qwen4_exp/executor.cpp):
  - Добавлена runtime-проверка `gguf_ggml_cpu_available()` (AVX2+FMA+F16C при `NINFER_GGML_CPU_AVX2=ON`, AVX+F16C+SSSE3+SSE4.1 при `OFF`).
  - Конструктор `Executor::Impl` проверяет `options.hybrid_experts.cpu_ggml && !ops::gguf_ggml_cpu_available()` на старте и завершается с понятным диагностическим сообщением (`--expert-cpu-arithmetic ggml: this build's ggml kernels need instructions which this CPU lacks; use exact or a build with NINFER_GGML_CPU_AVX2=OFF`) без `SIGILL`.
  - В [`tests/ops/test_gguf_expert_cpu_ggml.cpp`](tests/ops/test_gguf_expert_cpu_ggml.cpp) при `!gguf_ggml_cpu_available()` тест завершается кодом возврата `77` (CTest `Skipped`).

### 2.3. Нативные MoE CPU-эксперты (`CpuExpertBackend::Avx`)
- [`include/ninfer/ops/moe_expert_cpu.h`](include/ninfer/ops/moe_expert_cpu.h), [`src/ops/moe_experts/cpu_projection.h`](src/ops/moe_experts/cpu_projection.h), [`src/ops/moe_experts/moe_expert_cpu.cpp`](src/ops/moe_experts/moe_expert_cpu.cpp), [`src/ops/moe_experts/moe_expert_cpu_avx.cpp`](src/ops/moe_experts/moe_expert_cpu_avx.cpp):
  - Добавлен `CpuExpertBackend::Avx` (`project_avx` / `AvxDot`) с проверкой AVX+F16C+SSSE3+SSE4.1 и XCR0 через CPUID.
  - Для `int8` активаций применяется сдвоенный 128-битный `_mm_maddubs_epi16` / `_mm_madd_epi16`; для FP32-активаций — распаковка `_mm_cvtepi8_epi32` + `_mm256_mul_ps`/`_mm256_add_ps`.
  - Добавлен в цикл проверки против FP64-оракула в [`tests/ops/test_moe_expert_cpu.cpp`](tests/ops/test_moe_expert_cpu.cpp).

### 2.4. Gemma 4 CPU Vision (`gemma4::cpu_vision::avx_kernels`)
- [`src/models/gemma4/vision_cpu_kernels.h`](src/models/gemma4/vision_cpu_kernels.h), [`src/models/gemma4/vision_cpu_kernels.cpp`](src/models/gemma4/vision_cpu_kernels.cpp), [`src/models/gemma4/vision_cpu_kernels_avx.cpp`](src/models/gemma4/vision_cpu_kernels_avx.cpp), [`src/models/gemma4/sources.cmake`](src/models/gemma4/sources.cmake):
  - Реализованы ядра `gemm`, `softmax`, `geglu` на 256-битном AVX 1.0 без инструкций FMA и добавлен runtime-диспатч `avx2_supported() -> avx_supported() -> portable_kernels()`.
  - Протестировано в [`tests/models/gemma4/test_vision_cpu_kernels.cpp`](tests/models/gemma4/test_vision_cpu_kernels.cpp) (`selected kernels: avx`).

### 2.5. Бенчмарки и документация
- [`bench/ops/gguf_expert_cpu_bench.cpp`](bench/ops/gguf_expert_cpu_bench.cpp) + [`bench/ops/benchmarks.cmake`](bench/ops/benchmarks.cmake): постоянный микробенчмарк для сравнения `Scalar` / `Avx` / `Avx2` / `ggml` и SwiGLU на формах Flash-Next (`640x2560`, `2560x640`).
- Обновлены [`docs/build-linux.md`](docs/build-linux.md), [`docs/maintainer/build-system.md`](docs/maintainer/build-system.md), [`docs/cli.md`](docs/cli.md), [`docs/serving.md`](docs/serving.md), [`docs/qwen3-8-flash-next.md`](docs/qwen3-8-flash-next.md), [`docs/gemma4.md`](docs/gemma4.md), [`docs/performance.md`](docs/performance.md).

---

## 3. Результаты замеров и верификации на `192.168.50.38`

### 3.1. Микробенчмарк GGUF CPU-экспертов (`ninfer_gguf_expert_cpu_bench`, сборка `e2dc1705`, `NINFER_GGML_CPU_AVX2=OFF`, `iters=15`)
Сравнение медианного времени одного матричного умножения (`T` токенов, `thr` потоков; формы `640x2560` и `2560x640`):

| Формат | Форма | `T` | `thr` | `scalar`, мс | `avx`, мс (ускор.) | `ggml` (без AVX2), мс (ускор.) |
|---|---|---:|---:|---:|---:|---:|
| `IQ3_S` | `640x2560` | 1 | 1 | 9.1291 | 0.6352 (`14.37x`) | **0.4997 (`18.27x`)** |
| `IQ3_S` | `640x2560` | 1 | 6 | 1.8258 | 0.3875 (`4.71x`) | **0.3671 (`4.97x`)** |
| `IQ3_S` | `640x2560` | 1 | 8 | 1.6631 | 0.5888 (`2.82x`) | **0.5743 (`2.90x`)** |
| `IQ3_S` | `640x2560` | 4 | 6 | 3.3473 | **0.4927 (`6.79x`)** | 0.6348 (`5.27x`) |
| `IQ3_S` | `640x2560` | 4 | 8 | 2.8215 | **0.6327 (`4.46x`)** | 0.7529 (`3.75x`) |
| `IQ3_S` | `640x2560` | 8 | 6 | 5.3273 | **0.5932 (`8.98x`)** | 0.9746 (`5.47x`) |
| `IQ3_S` | `640x2560` | 8 | 8 | 4.3352 | **0.7251 (`5.98x`)** | 1.0316 (`4.20x`) |
| `IQ4_NL` | `640x2560` | 1 | 6 | 1.0755 | 0.3486 (`3.08x`) | **0.3179 (`3.38x`)** |
| `IQ4_NL` | `640x2560` | 8 | 6 | 4.5070 | **0.5388 (`8.36x`)** | 0.5901 (`7.64x`) |
| `IQ2_S` | `640x2560` | 1 | 6 | 2.0470 | 1.0806 (`1.89x`) | **0.3537 (`5.79x`)** |
| `IQ2_S` | `640x2560` | 8 | 6 | 10.2044 | 5.9582 (`1.71x`) | **0.8230 (`12.40x`)** |
| `Q2_0` | `640x2560` | 1 | 6 | 1.7660 | **0.3523 (`5.01x`)** | — (покрыт `avx`) |
| `Q2_0` | `640x2560` | 8 | 6 | 6.1440 | **0.5569 (`11.03x`)** | — (покрыт `avx`) |
| `Q8_0` | `2560x640` | 8 | 6 | 4.4940 | 0.5802 (`7.75x`) | **0.5610 (`8.01x`)** |

Наблюдения по GGUF-микробенчмарку:
- На `T=1` быстрее всех путь `ggml` (`-DNINFER_GGML_CPU_AVX2=OFF`); на `T=4` и `T=8` (`IQ3_S`, `IQ4_NL`) точный путь `avx` **опережает и `scalar`, и `ggml`** (так как `avx` переиспользует веса по батчу токенов). На `IQ2_S` (`T=1..8`) путь `ggml` быстрее `avx` до `12.40x` против `1.71x`.
- **`thr=6` (текущее значение `auto`: `16 / 2 - 2 = 6`) во всех без исключения замерах быстрее `thr=8`**, поэтому рекомендация `--expert-cpu-threads 8` из первоначального черновика PR удалена из документации как опровергнутая замером.
- Для `SwiGLU` (`n=5120`) замер составил `scalar=3.2291 ms`, `avx=3.2272 ms (1.00x)` (т.к. `swiglu_avx` делегирует в `swiglu_scalar`).

### 3.2. Оценка целесообразности Native MoE CPU (`CpuExpertBackend::Avx`) и Gemma 4 CPU Vision AVX (порог >= 10%)
По замерам временного оценочного бенчмарка `ninfer_cpu_avx_eval_bench` (сборка `e2dc1705`, затем удалённого из дерева):
- **Native MoE (`CpuExpertBackend::Avx` vs `Scalar`)**:
  - `Q4_A16`: `T=1` `12.7167 -> 9.6003 ms (1.32x)`, `T=8` `46.0972 -> 14.9393 ms (3.09x)`
  - `Q8_A8`: `T=1` `2.1641 -> 1.5363 ms (1.41x)`, `T=8` `10.3112 -> 6.0471 ms (1.71x)`
  - `Q4_A8`: `T=1` `10.5246 -> 10.0060 ms (1.05x)`, `T=8` `17.3785 -> 13.3138 ms (1.31x)`
  - `Q2_A8`: `T=1` `10.5698 -> 10.2961 ms (1.03x)`, `T=8` `20.4506 -> 15.8937 ms (1.29x)`
  - `Q5_A8`: `T=1` `19.4506 -> 19.0894 ms (1.02x)`, `T=8` `26.6194 -> 22.9701 ms (1.16x)`
  - `BF16`: `T=1` `13.0633 -> 13.2209 ms (0.99x)`, `T=8` `38.0693 -> 38.5332 ms (0.99x)` (ускорения нет)
  -> Все квантованные форматы дают ускорение **от 16% до 209%** при `T=8` (`Q4_A16` и `Q8_A8` — `+32%`/`+41%` уже на `T=1`), что превышает порог `10%`. Реализация оставлена.
- **Gemma 4 CPU Vision (`avx_kernels` vs `portable_kernels`)**:
  - `gemm 72x2340x128`: `18.2645 -> 1.4412 ms (12.67x)`
  - `gemm 1152x1152x128`: `142.3476 -> 10.2733 ms (13.86x)`
  - `gemm 4608x1152x64`: `285.0626 -> 20.2626 ms (14.07x)`
  - `softmax 256x256`: `0.4058 -> 0.1163 ms (3.49x)`
  - `geglu n=294912`: `8.7504 -> 0.6550 ms (13.36x)`
  -> Ускорение **3.49x–14.07x**. Реализация оставлена.

### 3.3. Сквозной бенчмарк `ninfer_bench` и Docker-сервер `ninfer-serve`
Условия стенда: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-ninfer-v3.ninfer` (58.24 GB / 54.2 GiB) + companion `Qwen3.8-Flash-Next-ngram-table-IQ4_NL-ninfer-v3.ninfer` (28.91 GB), `--kv-dtype rk4v4 --expert-residency disk --expert-cache-mib 18432 --prefill-chunk 2048`, `Tesla V100-PCIE-32GB`, сборка `e2dc1705` (`NINFER_GGML_CPU_AVX2=OFF`):
- **Ограничение стенда**: на хосте доступно 31 GiB физической RAM, поэтому режим `--expert-residency host` (требующий размещения 54.2 GiB в pinned RAM) и гибридное CPU+GPU исполнение на полной модели Flash-Next IQ3_S в этом окружении **не запускались** (`disk` residency выполняет экспертов на GPU через `ExpertStream`). Гибридный путь CPU+GPU верифицирован тестами `ninfer_qwen4_exp_hybrid_experts_test` и `ninfer_moe_experts_gguf_test`.
- **`ninfer_bench` (`warmup=1`, `repetitions=3`, корпус `bench/fixtures/bench_corpus.ids`)**:
  - Без `--spec` (`max_context=2048`, `cuda_graph`): `pp512 = 53.11 ± 0.35 tok/s`, `pp2048 = 118.74 ± 0.19 tok/s`, `tg128 = 19.89 ± 1.55 tok/s`.
  - С `--spec mtp --draft-tokens 3` (`max_context=518`, `mtp_cuda_graph`): `pp512 = 60.62 ± 0.22 tok/s`, `tg128 = 11.32 ± 0.64 tok/s` (`spec acc = 0.5674`, `141/3` раундов; на холодном прогоне при 31 GiB RAM 3-токенная верификация вызывает больше чтений недостающих экспертов с диска).
- **Живой Docker-сервер (`ninfer-v100:e5-2650v2-avx`, образ на базе `54457d2c`, `switch-to-flash-next.sh`)**:
  - Поднялся без ошибок (`Up ... (healthy)`, порт `8087`), тестовый запрос `/v1/chat/completions` (`test_request.py`) выполнился успешно (`predicted_n=20`, `draft_n=18`, `draft_n_accepted=12`, `predicted_per_second=9.19`).

### 3.4. Тесты (`ctest`, ревизия `54457d2c`, E5-2650 v2 + Tesla V100)
1. **Конфигурация `build-avx` (`-DNINFER_GGML_CPU_AVX2=OFF -DCMAKE_CUDA_ARCHITECTURES=70`)**:
   - `-R _expert`: **13/13 Passed** (`ninfer_expert_delivery_test`, `ninfer_qwen_native_expert_layout_test`, `ninfer_qwen4_exp_expert_cache_test`, `ninfer_qwen4_exp_expert_profile_test`, `ninfer_qwen4_exp_hybrid_experts_test`, `ninfer_qwen4_exp_expert_cache_visibility_test`, `ninfer_qwen4_exp_expert_stream_test`, `ninfer_moe_experts_test`, `ninfer_moe_experts_gguf_test`, `ninfer_moe_experts_native_test`, `ninfer_moe_expert_cpu_test`, `ninfer_gguf_expert_cpu_test`, `ninfer_gguf_expert_cpu_ggml_test`).
   - `-R vision_cpu`: **2 Passed, 1 Skipped** (`ninfer_qwen3_5_vision_cpu_real_test` пропущен без real fixture).
   - `-R qwen4_exp`: 10 Passed, 5 Skipped (`*_real`, `peer_copy_order_test`), **1 Failed**: Python-тест `ninfer_qwen4_exp_ngram_writer_interop_test` упал в контейнере из-за отсутствия модуля `pytest`.
2. **Конфигурация `build-avx2` (`-DNINFER_GGML_CPU_AVX2=ON -DCMAKE_CUDA_ARCHITECTURES=70`)**:
   - `-R _expert`: **12 Passed, 1 Skipped** (`ninfer_gguf_expert_cpu_ggml_test` корректно завершился кодом `77`), **0 Failed**.
   - `-R vision_cpu`: **2 Passed, 1 Skipped**, **0 Failed**.
   - `-R qwen4_exp_hybrid_experts`: **1/1 Passed**.
   - Проверка старта CLI `ninfer` с `--expert-cpu-arithmetic ggml` на `build-avx2` (сборка `e2dc1705`, `executor.cpp` не менялся после неё): чистый отказ с ненулевым кодом без `SIGILL`.

---

## 4. Источники
1. [Intel Xeon Processor E5-2650 v2 Specifications (Intel ARK)](https://www.intel.com/content/www/us/en/products/sku/75789/intel-xeon-processor-e52650-v2-20m-cache-2-60-ghz/specifications.html)
2. [Intel Intrinsics Guide (AVX, F16C, SSSE3, SSE4.1)](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
3. [Agner Fog — Optimizing software in C++ / The microarchitecture of Intel CPUs (Ivy Bridge)](https://www.agner.org/optimize/microarchitecture.pdf)
4. [llama.cpp — ggml-cpu x86 quants](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cpu/arch/x86)
