# План оптимизации ninfer-v100 под Intel Xeon E5-2650 v2 (без AVX2)

- [x] Анализ характеристик и набора инструкций Intel Xeon E5-2650 v2 (Ivy Bridge-EP)
- [x] Исследование паттернов и подходов оптимизации LLM-движков на CPU без AVX2 (llama.cpp, ggml, FastLLM)
- [x] Комплексный аудит кодовой базы ninfer-v100: выявление CPU-компонентов, критических путей и жестких флагов AVX2/FMA
- [x] Разработка бэкенда `CpuExpertBackend::Avx` в MoE CPU Projection (`moe_expert_cpu`): гибрид SSSE3/SSE4.1 (128-bit int dot) + AVX 1.0 (256-bit FP dot)
- [x] Реализация AVX 1.0 кернелов без FMA для Gemma 4 Vision CPU (`vision_cpu_kernels_avx.cpp`)
- [x] Интеграция сборки и рантайм-диспатча для GGUF CPU Experts (`gguf_expert_cpu` / `quants.c` c флагами `-mavx -mf16c`)
- [x] Документирование и настройка потоков для Xeon E5-2650 v2 (`--expert-cpu-threads 8`, 8 физических ядер)
- [x] Формирование отдельного файла отчета `report_add_optimization.md` со ссылками на источники и описанием изменений
- [x] Верификация математической точности и стабильности через оракулы FP64 в тестах на хосте 192.168.50.38 (Xeon E5-2650 v2)
- [x] Создание Pull Request (MR) в ветку `main` с полным описанием и ссылками на источники

## Итоги выполнения
Все задачи успешно завершены. Pull Request #1 открыт: https://github.com/andrei-tolstov/ninfer-v100/pull/1
Отчет зафиксирован в файле `report_add_optimization.md`.
