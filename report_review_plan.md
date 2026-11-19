# Отчёт: проверка замечаний PR #1 и план исправлений (ветка `feat/e5-2650v2-avx-support`)

## 1. Цель и контекст

Проверить каждое замечание из `review.md` (отзыв на PR #1: AVX 1.0 + F16C/SSSE3/SSE4.1 CPU-пути и
двухсокетный NUMA, 68 файлов, +6914/−528), подтвердить или опровергнуть их по коду, подготовить и
выполнить план исправлений, проверить тестами на целевом хосте и обновить PR #1.

- **Хост проверки**: Intel Xeon E5-2650 v2 @ 2.60 GHz (1 сокет, 8C/16T; флаги `ssse3 sse4_1 sse4_2 avx f16c`, без `avx2`/`fma`), RAM 31 GiB, Tesla V100-PCIE-32GB (`sm_70`, CUDA 12.8, `GPU-733ae7cc-c1bf-3352-42d5-ba2762470325`), страница 4096 Б.
- **Сборка**: каталог `build-avx` в контейнере `ninfer-dev:12.8` (host nvcc 12.4 не проходит требуемые 12.8+), `Release`, `CMAKE_CUDA_ARCHITECTURES=70`, `NINFER_GGML_CPU_AVX2=OFF`, `BUILD_TESTING=ON`.
- **Базовый HEAD проверки**: `c65cde843`.

---

## 2. Вердикты по пунктам `review.md`

| # | Пункт | Вердикт | Доказательство (file:line в `c65cde843`) |
|---|---|---|---|
| M1 | `live_floats` при `live==4`: старшие 128 бит YMM не определены | **CONFIRMED** | `src/ops/moe_experts/gguf_expert_cpu_avx.cpp:99` — единственный голой `_mm256_castps128_ps256(_mm_loadu_ps(...))` в репозитории (все 21 остальные применение кормят `insertf128`). Проявилось бы как NaN в `butterfly8` только при мусоре NaN/Inf в верхней половине; GCC/Clang на VEX `vmovups %xmm` обнуляют её, поэтому тесты проходили |
| M2a | `prepare_input_avx` без проверки `a8` (в отличие от скалярного) | **REFUTED** (в заявленной формулировке) | `moe_expert_cpu.cpp:174-176,195-197` — скалярный путь так же gated по `any_in_a8`; `moe_expert_cpu_avx.cpp:537-551` `prepare_input_avx` при `!a8` уходит в ранний return без проверки — паритет с скаляром сохранён |
| M2b | ggml-квантизаторы bf16 без проверки NaN/Inf | **CONFIRMED** | `third_party/ggml-quants/src/ggml-cpu/arch/x86/quants.c:630-647` (`ggml_quantize_bf16_q8_0`), `:649-708` (`ggml_quantize_bf16_q8_K`) — `_mm256_max_ps` по amax: NaN → `d`=NaN → коды насыщаются в −128, без исключения. Достигается через `--expert-cpu-arithmetic ggml` (`hybrid_expert_options.h:82-90`, `executor.cpp:1569`) |
| m1 | `ScopedThreadMemPolicy` без save/restore | **CONFIRMED** | `numa.cpp:438-481` — конструкторы ставят `set_mempolicy` напрямую; dtor `:483-487` бездумно `MPOL_DEFAULT`, чужая (numactl/вложенная) политика теряется |
| m2 | `bind_thread_to_cpu` без гварда `CPU_SETSIZE` | **CONFIRMED** | `numa.cpp:362-372` — `CPU_SET(cpu_id,…)` без границы (у `bind_thread_to_node` гвард есть, `:383`); при >1024 CPU — запись за пределы статического `cpu_set_t` |
| m3 | `page_size` по умолчанию 4096 | **CONFIRMED, латентно** | `numa.h:133`, `numa.cpp:683` (`psize = max(page_size, 4096)`); на x86 с 4K-страницами и page-aligned mmap безвредно, на 64K-хостах срезало бы границы шардов |
| m4 | Конструктор `PinnedHostBuffer` без `cudaGetLastError` до register; `(void)bind…`; тихий деград в Interleave | **CONFIRMED** | `arena.cu:481-523` (основной конструктор проверяет ошибку на `:427`, шардовый — нет) |
| m5 | `Shmem` не вычитается из доступной памяти | **CONFIRMED** | `numa.cpp:77-115` `parse_node_meminfo` — строки `Shmem` нет; tmpfs-память завышала `free_bytes` |
| m6 | Sharded — равномерный split, строгий `MPOL_BIND` | **CONFIRMED** | `numa.cpp:683-700` (split `pinned*(idx+1)/k`), `:407` (`MPOL_BIND`); при асимметрии free-памяти оба сокета тянут из бедного |
| m7 | `raw + bytes` переполнение | **CONFIRMED, уже смягчено** | `numa.cpp:404-406` — `aligned_end <= aligned_begin` уже ловит wrap; явный overflow-гвард не добавлен (косметика) |
| Спорный 1 | «Auto не использует второй сокет» | **REFUTED как баг** | Поведение намеренное и задокументировано (`docs/cli.md:271`): Auto изолируется на сокете GPU, пока туда влезают веса и треды |
| Спорный 2 | Разные планы NUMA в разных местах | **CONFIRMED и сильнее** | Три вызова `plan_host_numa`: `model.cpp:491` (пакетный `pinned_capacity_bytes`), `expert_misses.cu:782` (неспакетный `total_host_bytes`; при disk-резидентности `0` → `ram_fits` на `:515-517` форсирует Isolate), `materializer.cpp:476` (fallback). Rank gpu_node различается при `--devices` |
| Спорный 3 | Explicit mirror/sharded/isolate без проверки RAM | **CONFIRMED** | `numa.cpp:501-506` — explicit-ветки не проверяют свободную память; при нехватке срыв на first-touch в `expert_misses.cu:815-847` (`MPOL_BIND`) |
| CRITICAL-1 | Тесты не покрывают `live != 8` | **REFUTED** | `tests/ops/test_moe_expert_cpu.cpp:17` `H=2560, I=640`; `test_gguf_expert_cpu.cpp:277-290,458-461` — bit-exact сравнение скаляр/AVX при k=640 покрывает `live=4` |
| Притязания | `maxnode`, SpinPool/RemoteRunner | **CONFIRMED** | `numa.cpp:159-167` — `maxnode = words*bits + 1` корректно (ядро `get_nodes()` вычитает 1); пер-сокетные пулы без cross-socket barrier (нюанс: `spin_all()` `:1016-1025` держит оба пула в одной спин-фазе) |
| Нюанс | «iq3_s: 4×`_mm_mullo_epi32` = эмуляция PMADDWD» | **REFUTED** | `PMULLD` (SSE4.1) нативен на Ivy Bridge — это и есть native-инструкция, bench не нужен |
| Нюанс | thread_local в vision-кernels | **PARTLY** | Деструкторы исполняются, объём <2 MiB; риск только при явном detach — в стеке такого нет |
| Нюанс | AVX1 TUs без AVX2/FMA | **CONFIRMED** | Per-file флаги `basic_sources.cmake:104-121` (`-mavx;-mf16c;-mssse3;-msse4.1;-ffp-contract=off`) + runtime CPUID-диспач `moe_expert_cpu.cpp:85-120`; objdump объектов `build-avx`: 0 ymm-integer, 0 FMA. Нюанс: AVX-TU gemma4 без `-ffp-contract=off` (`gemma4/sources.cmake:31`) |

---

## 3. Новые находки (N1–N4) — вопросы мейнтейнеру, без изменения кода

- **N1. Конфликт геометрии панелей.** Канонический `kRowSplitPanelRows = 4` (`src/core/weight.h:113`, формат `row_split_panel4_v1`) против 64-строчного адресования в MoE-ядрах (`cpu_projection.h:59-61`, `moe_expert_cpu_avx.cpp:214`, `moe_experts_native.cu:102-104`, `include/ninfer/ops/moe_expert_cpu.h:53`). `row_split_panel_supported` проверяет `% 4` (`permute_row_split.cu:70-76`), но панельный permute — device-only и применяется только к dense-проекциям qwen3_5 (`storage_trades.cpp:178-183`): в проде MoE панельных байт не видит, конфликт латентен, но один enum несёт два несовместимых смысла.
- **N2. iq_rows T==1: безусловная загрузка 8 scale-слов** (`gguf_expert_cpu_avx.cpp:313`) при `k%256≠0` — OOB-чтение `act_scales` и весовых блоков; исправлено в Fix 1 (live-guard).
- **N3. `free_bytes` — статический снапшот на старте процесса** (`host_numa_topology`, `numa.cpp:299-337`): память не обновляется.
- **N4. AVX-TU gemma4 без `-ffp-contract=off`** (`gemma4/sources.cmake:31`) → ожидаемое расхождение AVX1↔AVX2 в vision-ядрах; задокументировать.

---

## 4. Выполненные исправления (5 фиксов)

### Fix 1 — `fix(moe-cpu-avx)` (`gguf_expert_cpu_avx.cpp`)
- **M1**: `live_floats` при `live==4` — явный `_mm256_insertf128_ps(_mm256_castps128_ps256(lo), _mm_setzero_ps(), 1)` вместо голого `castps128_ps256`: старшая половина YMM теперь гарантированно нулевая по спецификации, а не «как скомпилирует компилятор».
- **N2**: iq_rows T==1 — live-guard: `run_half` пропускает под-срезы `sub >= live` (нулевые `u_lo/u_hi`, `factor0/factor1 = 0`), а scale загружается через `live_floats(act_scales + 8*g, live)`. Устраняет OOB-чтение весовых блоков и scale-слов при `k%256≠0` (640-колоночная down-проекция).
- **Приёмка**: codegen — 76× `vinsertf128`, 0 legacy `movups %xmm→%ymm`; bit-exact тесты gguf/moe зелёные.

### Fix 2 — `fix(gguf-cpu-ggml)` (`gguf_expert_cpu_ggml.cpp` + тест)
- **M2b**: скалярный non-finite-скан BF16 (`exponent == 0x7F` покрывает NaN и ±Inf) в `GgmlCpuActivation::prepare_bf16_token/prepare_bf16` до ggml-квантизаторов; бросает `GGUF CPU expert: nonfinite activation` — как точный путь. Новый тест: NaN / +Inf / −Inf отклоняются.

### Fix 3 — `fix(numa)` (`numa.h`/`numa.cpp` + тесты)
- **m1**: `ScopedThreadMemPolicy` сохраняет предыдущую политику через `get_mempolicy` и восстанавливает её при выходе из области (не слепой `MPOL_DEFAULT`), — numactl-политика процесса и вложенные области переживают скоуп.
- **m2**: `bind_thread_to_cpu` — гвард `cpu_id < CPU_SETSIZE`.
- **m3**: `page_size = 0` (дефолт) → `sysconf(_SC_PAGESIZE)` вместо жёсткого 4096.
- **m7**: задокументирован как смягчённый (`aligned_end <= aligned_begin`).
- **Тесты**: save/restore политики (syscall-проверка), `bind_thread_to_cpu(1<<20)==false`, `(-1)==false`.

### Fix 4 — `fix(numa)` (`numa.cpp` + `model.h/cpp`, `expert_misses.h/cu`, `executor.cpp`)
- **m5**: `parse_node_meminfo` вычитает `Shmem` из reclaimable (suffix-match, чтобы `ShmemHugePages` не подмешался).
- **m6**: Sharded split со взвешиванием по free-памяти узлов (через `__uint128_t` против переполнения `uint64` при гигабайтных значениях); равномерный split — fallback при неизвестных free_bytes.
- **Спорный 3**: explicit mirror/sharded/isolate — проверка свободной RAM до аллокации с явной ошибкой (`NUMA mirror/isolate/sharded: …cannot pin … B`) вместо ENOMEM на first-touch.
- **Спорный 2**: единый `HostNumaPlan` вычисляется один раз в `load_model` против реального `pinned_capacity_bytes` и сокета первичного ранка, хранится в `Model::numa_plan_` и передаётся в `ExpertMisses` через `MissOptions.numa_plan` (fallback — перепланирование, если план не передан).
- **Тесты**: Shmem (delta ≈10 GiB), free-weighted (отношение ~1.67 при 90:54, не 1:1), explicit Mirror с нехваткой RAM бросает.

### Fix 5 — `fix(vision-cpu)` (`gemma4/vision_cpu.*`, `qwen3_5/execution/vision*.cpp/h`, `gemma4_core.cpp`)
- **4 места** `device_numa_node(0)` → фактический выбранный CUDA-device: конструктор и `encode` gemma4-башни (`CpuVisionTower(model, cuda_device, threads)`), `encode_vision_on_cpu(..., cuda_device)` и `CpuVisionSession(weights, cuda_device)` (qwen3_5). CPU-воркеры vision теперь пинятся к сокету обслуживаемого GPU, а не к сокету device 0.

---

## 5. Результаты тестов (на хосте, `build-avx`, `ninfer-dev:12.8`)

| Тест | Результат |
|---|---|
| `ninfer_gguf_expert_cpu_test` | OK — bit-exact скаляр/AVX, включая `live=4` (k=640) и iq_rows T==1 |
| `ninfer_moe_expert_cpu_test` | CPU_EXPERT_ORACLE_PASS (a8-пути `formats=4,28,3 panel=1 a8=1`) |
| `ninfer_gguf_expert_cpu_ggml_test` | OK — включая новый non-finite-тест (NaN/±Inf отклонены) |
| `ninfer_numa_test` | PASS — включая review-фиксы (Shmem, free-weighted, explicit Mirror, `CPU_SETSIZE`, save/restore) |
| `ninfer_gemma4_vision_cpu_kernels_test` | PASS (kernels: avx) |
| `ninfer_cli_options_test` | OK |
| `ninfer_serve_options_test` | ok |

Codegen M1 (objdump `gguf_expert_cpu_avx.cpp.o`): `vinsertf128` = 76, legacy `movups %xmm→%ymm` = 0.

---

## 6. Открытые вопросы для мейнтейнера

1. **N1**: 4 vs 64 панели — какой смысл канонический для MoE? (один enum несёт два значения)
2. **A16**: валидация активаций шире non-finite (диапазон, denormal) перед CPU-экспертами?
3. **Спорный 3**: explicit mirror — ошибка против деградации с предупреждением, какая семантика ожидается?
4. **N4**: добавить `-ffp-contract=off` для AVX-TU gemma4?
5. **m4**: нужно ли закрыть `PinnedHostBuffer` (шардовый конструктор) до register + явно report-ить деградацию вместо тихого Interleave?
6. thread_local в vision-kernels — оставить как есть?

---

## 7. Обновление контейнера flash-next

- **Бинарник**: собран `ninfer-serve` (`build 1.7.2+cb38ca5a6`, `build-avx`, `NINFER_GGML_CPU_AVX2=OFF`, sm_70) и скопирован в `/opt/models/flash_next/bin/ninfer-serve`.
- **Образ**: `docker build -f Dockerfile.ninfer-v100-avx.local -t ninfer-v100:e5-2650v2-avx` пересобран с новым бинарником (создан 2026-11-19 16:13). Контейнер `flash-next-llama-server` пересоздан на новом образе.
- **Проверка на хосте**: с освобождённой V100 (после остановки `coder390-llama-server`, который на той же карте держит `coder390-27b`) новый бинарник загрузил модель `Qwen3.8-Flash-Next` (weights 4.57 GiB, experts в host-памяти, n-gram таблица 27465 MiB в RAM) за 1m41s и вышел в `listening on 0.0.0.0:8080`. По решению оператора дальнейший запуск flash-next не проводился — **проверка обновлённого контейнера засчитана успешной**.
- **Итоговое состояние**: `flash-next-llama-server` остановлен (обновлён, готов к запуску `switch-to-flash-next.sh`), `restart`-политика возвращена к исходной `unless-stopped`, LLM сессии `coder390-llama-server` восстановлен и healthy (держит V100), остальные контейнеры (`deepseek-*`, omniroute, patroni, etcd, webtop) не тронуты.
