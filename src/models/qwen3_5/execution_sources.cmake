target_sources(ninfer_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/execution/parameters.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/expert_runtime.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/ffn.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/gdn.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/mtp.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/text.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/vision.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/vision_overlay.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/vision_cpu.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/vision_cpu_kernels.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/draft.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/visual_scatter.cpp"
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  set(qwen_vision_cpu_avx "${CMAKE_CURRENT_LIST_DIR}/execution/vision_cpu_kernels_avx.cpp")
  set(qwen_vision_cpu_avx2 "${CMAKE_CURRENT_LIST_DIR}/execution/vision_cpu_kernels_avx2.cpp")
  target_sources(ninfer_model_runtime PRIVATE "${qwen_vision_cpu_avx}" "${qwen_vision_cpu_avx2}")
  set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/execution/vision_cpu_kernels.cpp"
    PROPERTIES COMPILE_DEFINITIONS NINFER_VISION_CPU_X86=1)
  if(MSVC)
    set_source_files_properties("${qwen_vision_cpu_avx}"
      PROPERTIES COMPILE_OPTIONS /arch:AVX COMPILE_DEFINITIONS NINFER_VISION_CPU_X86=1)
    set_source_files_properties("${qwen_vision_cpu_avx2}"
      PROPERTIES COMPILE_OPTIONS /arch:AVX2 COMPILE_DEFINITIONS NINFER_VISION_CPU_X86=1)
  else()
    set_source_files_properties("${qwen_vision_cpu_avx}"
      PROPERTIES COMPILE_OPTIONS "-mavx;-ffp-contract=off" COMPILE_DEFINITIONS NINFER_VISION_CPU_X86=1)
    set_source_files_properties("${qwen_vision_cpu_avx2}"
      PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-ffp-contract=off" COMPILE_DEFINITIONS NINFER_VISION_CPU_X86=1)
  endif()
endif()
