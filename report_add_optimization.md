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
  - Добавлен бэкенд `GgufCpuBackend::Avx` (автовыбор при наличии `avx`, `f16c`, `ssse3`, `sse4_1` и ОС-поддержки `XCR0`, когда `AVX2` отсутствует); флаги `cpu_has_avx_f16c()` и `cpu_has_avx2()` кэшируются через `static const bool`.
  - Явный запрос недоступного бэкенда (`Avx` или `Avx2`) через `gguf_cpu_rows()` и `gguf_cpu_swiglu_bf16()` проверяется в `resolve_backend()` и бросает `std::invalid_argument("GGUF CPU expert: unsupported CPU backend")` (без риска `SIGILL`), по аналогии с `moe_expert_cpu`.
  - Целочисленные dot-product для форматов `Q2_0`, `Q8_0`, `IQ4_NL`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_XS` реализованы через 128-битные SSSE3/SSE4.1 примитивы (`_mm_maddubs_epi16`, `_mm_madd_epi16`, `_mm_shuffle_epi8`, `_mm_abs_epi8`, `_mm_sign_epi8`, `_mm_cvtepi8_epi16`) и пошаговое округление FMA/multiplication с директивой `-ffp-contract=off` на единицу трансляции.
  - По замечаниям ревью PR #1 устранена программная эмуляция скалярного `std::fma` в `IQ2_S` (`halves == true`) и делегирование `swiglu_avx` в `swiglu_scalar`: в AVX 1.0 добавлены векторизованные регистровые эмуляторы `fma4` / `fma8` (на базе двойной точности `TwoSum` + round-to-odd по Болдо–Мелькионду), побитово эквивалентные IEEE-754 `std::fma`, а также полноценный `swiglu_avx` на `AVX 1.0 + SSE4.1` с единой `ops/common/silu.h`.
- Проверка корректности (`ninfer_gguf_expert_cpu_test`, сборка `5e8a74dc`): [`tests/ops/test_gguf_expert_cpu.cpp`](tests/ops/test_gguf_expert_cpu.cpp) — побитовое совпадение `fma_avx` со `std::fma` на 1,020,493 тест-кейсах (краевые случаи ±0/±Inf/NaN/субнормалы/точное взаимное уничтожение, 4096×5 точек double-rounding midpoint, 1,000,003 случайных троек с нечётным хвостом), побитовое совпадение `avx` со `scalar` по всем форматам (`27`, `10`, `23`, `18`, `19`, `20`, `24`), промежуточным `BF16` словам (`avx middle: 100003 words equal`), проверка исключения `std::invalid_argument` для неподдерживаемого `Avx2` и совпадение с GPU в `ninfer_moe_experts_gguf_test`.

### 2.2. Управление ISA и runtime-защита для `--expert-cpu-arithmetic ggml`
- [`CMakeLists.txt`](CMakeLists.txt) и [`src/ops/basic_sources.cmake`](src/ops/basic_sources.cmake):
  - Добавлена опция `NINFER_GGML_CPU_AVX2` (по умолчанию `ON`). При `ON` `third_party/ggml-quants/src/ggml-cpu/arch/x86/quants.c` компилируется с `-mavx2;-mfma;-mf16c` (добавлен `-mf16c` для согласования с проверками CPUID и MSVC `/arch:AVX2`), при `OFF` — с `-mavx;-mf16c` (под MSVC `/arch:AVX2` vs `/arch:AVX`), передавая `NINFER_GGML_CPU_REQUIRES_AVX2=1|0` в `gguf_expert_cpu_ggml.cpp`. *(Примечание: на стенде E5-2650 v2 без AVX2 флаг `-mf16c` в `AVX2`-ветке проверен сборкой, но runtime-выполнение `AVX2+FMA+F16C` на данном CPU физически не запускалось.)*
- [`src/ops/moe_experts/gguf_expert_cpu_ggml.h`](src/ops/moe_experts/gguf_expert_cpu_ggml.h), [`src/ops/moe_experts/gguf_expert_cpu_ggml.cpp`](src/ops/moe_experts/gguf_expert_cpu_ggml.cpp), [`src/models/qwen4_exp/executor.cpp`](src/models/qwen4_exp/executor.cpp):
  - В `gguf_ggml_cpu_available()` на `x86_64` дублирующийся разбор CPUID/XGETBV удалён и унифицирован через `gguf_cpu_backend_available(NINFER_GGML_CPU_REQUIRES_AVX2 ? GgufCpuBackend::Avx2 : GgufCpuBackend::Avx)` (`AVX2+FMA+F16C` vs `AVX+F16C+SSSE3+SSE4.1`).
  - Вне `x86_64` сохраняется `return false;` с явным комментарием: вендоризована только директория `third_party/ggml-quants/src/ggml-cpu/arch/x86/quants.c`, а `arch-fallback.h` не мапит `ggml_vec_dot_*` на ARM (`__aarch64__`).
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

### 3.1. Микробенчмарк GGUF CPU-экспертов (`ninfer_gguf_expert_cpu_bench`, сборка `5e8a74dc`, `NINFER_GGML_CPU_AVX2=OFF`, `iters=15`)
Сравнение медианного времени одного матричного умножения (`T` токенов, `thr` потоков; формы `640x2560` и `2560x640`):

| Формат | Форма | `T` | `thr` | `scalar`, мс | `avx`, мс (ускор.) | `ggml` (без AVX2), мс (ускор.) |
|---|---|---:|---:|---:|---:|---:|
| `IQ3_S` | `640x2560` | 1 | 1 | 6.9996 | 0.5024 (`13.93x`) | **0.3916 (`17.87x`)** |
| `IQ3_S` | `640x2560` | 1 | 6 | 1.5908 | 0.3430 (`4.64x`) | **0.3260 (`4.88x`)** |
| `IQ3_S` | `640x2560` | 1 | 8 | 1.4583 | 0.5921 (`2.46x`) | **0.5786 (`2.52x`)** |
| `IQ3_S` | `640x2560` | 4 | 6 | 3.3353 | **0.5019 (`6.64x`)** | 0.6443 (`5.18x`) |
| `IQ3_S` | `640x2560` | 4 | 8 | 2.8308 | **0.6482 (`4.37x`)** | 0.7693 (`3.68x`) |
| `IQ3_S` | `640x2560` | 8 | 6 | 5.3226 | **0.5904 (`9.02x`)** | 0.9773 (`5.45x`) |
| `IQ3_S` | `640x2560` | 8 | 8 | 4.3504 | **0.7253 (`6.00x`)** | 1.0325 (`4.21x`) |
| `IQ4_NL` | `640x2560` | 1 | 6 | 1.0734 | 0.3514 (`3.06x`) | **0.3271 (`3.28x`)** |
| `IQ4_NL` | `640x2560` | 8 | 6 | 4.5138 | **0.5413 (`8.34x`)** | 0.5693 (`7.93x`) |
| `IQ2_S` | `640x2560` | 1 | 1 | 10.2726 | 0.6721 (`15.28x`) | **0.3892 (`26.40x`)** |
| `IQ2_S` | `640x2560` | 1 | 6 | 2.0036 | 0.3982 (`5.03x`) | **0.3585 (`5.59x`)** |
| `IQ2_S` | `640x2560` | 4 | 6 | 5.6109 | **0.5664 (`9.91x`)** | 0.5793 (`9.69x`) |
| `IQ2_S` | `640x2560` | 8 | 1 | 60.0830 | **2.7491 (`21.86x`)** | 3.1426 (`19.12x`) |
| `IQ2_S` | `640x2560` | 8 | 6 | 10.3203 | **0.7573 (`13.63x`)** | 0.8325 (`12.40x`) |
| `Q2_0` | `640x2560` | 1 | 6 | 1.7573 | **0.3454 (`5.09x`)** | — (покрыт `avx`) |
| `Q2_0` | `640x2560` | 8 | 6 | 6.1604 | **0.5682 (`10.84x`)** | — (покрыт `avx`) |
| `Q8_0` | `2560x640` | 8 | 6 | 4.5123 | 0.5934 (`7.60x`) | **0.5608 (`8.05x`)** |
| `SwiGLU` | `n=5120` | — | — | 3.3362 | **0.1259 (`26.51x`)** | — |

Наблюдения по GGUF-микробенчмарку (`5e8a74dc` vs база `e2dc1705`):
- **Эффект векторизованного `fma8` в `IQ2_S`**: до оптимизации (на сборке `e2dc1705` со скалярным `std::fma`) `IQ2_S avx` давал `1.0806 ms` (`1.89x`) при `T=1, thr=6` и `5.9582 ms` (`1.71x`) при `T=8, thr=6`. После замены на регистровый `fma8` время сократилось до `0.3982 ms` (`5.03x`) и `0.7573 ms` (`13.63x`), и на `T=4`/`T=8` точный путь `avx` **вышел вперёд и относительно `ggml`** (`0.5664` vs `0.5793 ms` на `T=4`; `0.7573` vs `0.8325 ms` на `T=8`; `2.7491` vs `3.1426 ms` однопоточно при `T=8`).
- **Эффект векторизованного `swiglu_avx`**: вместо делегирования в `swiglu_scalar` (`3.2291 -> 3.2272 ms`, `1.00x` на `e2dc1705`) достигнуто `3.3362 -> 0.1259 ms` (**`26.51x`**).
- На `T=1` быстрее всех по-прежнему путь `ggml` (`-DNINFER_GGML_CPU_AVX2=OFF`); на `T=4` и `T=8` точный путь `avx` опережает и `scalar`, и `ggml` на `IQ3_S`, `IQ4_NL` (`T=8`) и `IQ2_S`.
- **`thr=6` (текущее значение `auto`: `16 / 2 - 2 = 6`) во всех без исключения замерах быстрее `thr=8`**, поэтому рекомендация `--expert-cpu-threads 8` не возвращается.

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
- **Сквозной `ninfer_bench`** (`warmup=1`, `repetitions=3`, корпус `bench/fixtures/bench_corpus.ids`, сборка `e2dc1705` до векторизации FMA8/SwiGLU):
  - Без `--spec` (`max_context=2048`, `cuda_graph`): `pp512 = 53.11 ± 0.35 tok/s`, `pp2048 = 118.74 ± 0.19 tok/s`, `tg128 = 19.89 ± 1.55 tok/s`.
  - С `--spec mtp --draft-tokens 3` (`max_context=518`, `mtp_cuda_graph`): `pp512 = 60.62 ± 0.22 tok/s`, `tg128 = 11.32 ± 0.64 tok/s` (`spec acc = 0.5674`, `141/3` раундов; на холодном прогоне при 31 GiB RAM 3-токенная верификация вызывает больше чтений недостающих экспертов с диска).
- **Docker-образ (`ninfer-v100:e5-2650v2-avx`)**:
  - На ревизии `5e8a74dc` образ пересобран и проверен командой `ninfer-serve --version` -> `ninfer-serve 1.7.2+5e8a74dc`. Живой запрос `/v1/chat/completions` (`test_request.py`, порт `8087`) выполнялся ранее на образе `54457d2c` (`predicted_n=20`, `draft_n=18`, `draft_n_accepted=12`, `predicted_per_second=9.19`) и после обновления до `5e8a74dc` повторно не прогонялся.

### 3.4. Тесты (`ctest`, ревизия `5e8a74dc`, E5-2650 v2 + Tesla V100)
1. **Конфигурация `build-avx` (`-DNINFER_GGML_CPU_AVX2=OFF -DCMAKE_CUDA_ARCHITECTURES=70`)**:
   - `-R _expert`: **13/13 Passed** (`ninfer_expert_delivery_test`, `ninfer_qwen_native_expert_layout_test`, `ninfer_qwen4_exp_expert_cache_test`, `ninfer_qwen4_exp_expert_profile_test`, `ninfer_qwen4_exp_hybrid_experts_test`, `ninfer_qwen4_exp_expert_cache_visibility_test`, `ninfer_qwen4_exp_expert_stream_test`, `ninfer_moe_experts_test`, `ninfer_moe_experts_gguf_test`, `ninfer_moe_experts_native_test`, `ninfer_moe_expert_cpu_test`, `ninfer_gguf_expert_cpu_test`, `ninfer_gguf_expert_cpu_ggml_test`).
   - `-R vision_cpu`: **2 Passed, 1 Skipped** (`ninfer_qwen3_5_vision_cpu_real_test` пропущен без real fixture).
   - *(Примечание: расширенный фильтр `-R qwen4_exp` прогонялся на `54457d2c`: 10 Passed, 5 Skipped, 1 Failed у Python-теста `ninfer_qwen4_exp_ngram_writer_interop_test` из-за отсутствия `pytest` в контейнере.)*
2. **Конфигурация `build-avx2` (`-DNINFER_GGML_CPU_AVX2=ON -DCMAKE_CUDA_ARCHITECTURES=70`)**:
   - `-R _expert`: **12 Passed, 1 Skipped** (`ninfer_gguf_expert_cpu_ggml_test` корректно завершился кодом `77`), **0 Failed**.
   - `-R vision_cpu`: **2 Passed, 1 Skipped**, **0 Failed**.
   - Проверка старта CLI `ninfer` с `--expert-cpu-arithmetic ggml` на `build-avx2` (`5e8a74dc`): чистый отказ с ненулевым кодом и сообщением `error: --expert-cpu-arithmetic ggml: this build's ggml kernels need instructions which this CPU lacks; use exact or a build with NINFER_GGML_CPU_AVX2=OFF` без `SIGILL`.

---

## 4. Источники
1. [Intel Xeon Processor E5-2650 v2 Specifications (Intel ARK)](https://www.intel.com/content/www/us/en/products/sku/75789/intel-xeon-processor-e52650-v2-20m-cache-2-60-ghz/specifications.html)
2. [Intel Intrinsics Guide (AVX, F16C, SSSE3, SSE4.1)](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
3. [Agner Fog — Optimizing software in C++ / The microarchitecture of Intel CPUs (Ivy Bridge)](https://www.agner.org/optimize/microarchitecture.pdf)
4. [llama.cpp — ggml-cpu x86 quants](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cpu/arch/x86)
5. Sylvie Boldo, Guillaume Melquiond — *Emulation of a FMA and Correctly Rounded Sums: Proved Algorithms Using Rounding to Odd*, IEEE Transactions on Computers, 57(4):462–471, 2008.
