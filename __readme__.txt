https://github.com/hoivb612/onnx-genai

Branch hv/affin

==================

C:\llama.cpp\Onnx-GenAI>set path=e:\xCloud_Reuse\Python38.diffusers.v2;%path%

C:\llama.cpp\Onnx-GenAI>build [--use_dml] --skip_wheel

 Directory of C:\llama.cpp\Onnx-GenAI\build\Windows\RelWithDebInfo\RelWithDebInfo

10/02/2024  01:58 PM    <DIR>          .
10/02/2024  01:59 PM    <DIR>          ..
10/02/2024  01:52 PM        33,208,696 onnxruntime-genai-static.lib
10/02/2024  01:52 PM         9,367,552 onnxruntime-genai-static.pdb
10/02/2024  01:58 PM         2,114,560 onnxruntime-genai.dll
10/02/2024  01:58 PM            10,439 onnxruntime-genai.exp
10/02/2024  01:58 PM            18,480 onnxruntime-genai.lib
10/02/2024  01:58 PM        25,186,304 onnxruntime-genai.pdb
               6 File(s)     69,906,031 bytes
               2 Dir(s)  495,532,843,008 bytes free

 Directory of C:\llama.cpp\Onnx-GenAI\build\Windows\RelWithDebInfo

10/02/2024  01:59 PM         3,328,032 D3D12Core.dll
10/02/2024  01:59 PM        18,527,264 DirectML.dll
10/02/2024  01:59 PM        14,613,552 onnxruntime.dll
               3 File(s)     36,468,848 bytes
               0 Dir(s)  495,529,390,080 bytes free

========================================================================================================

C:\llama.cpp\Onnx-GenAI>cmake --preset windows_x64_cpu_relwithdebinfo -B build_cpu_pdb -DENABLE_PYTHON=OFF -DENABLE_TESTS=OFF -DUSE_DML=OFF
Preset CMake variables:

  CMAKE_BUILD_TYPE="RelWithDebInfo"
  CMAKE_CXX_FLAGS="/EHsc /Qspectre /MP /guard:cf /DWIN32 /D_WINDOWS /DWINAPI_FAMILY=100 /DWINVER=0x0A00 /D_WIN32_WINNT=0x0A00 /DNTDDI_VERSION=0x0A000000 /O2 /Ob1 /DNDEBUG"
  CMAKE_C_FLAGS="/EHsc /Qspectre /MP /guard:cf /DWIN32 /D_WINDOWS /DWINAPI_FAMILY=100 /DWINVER=0x0A00 /D_WIN32_WINNT=0x0A00 /DNTDDI_VERSION=0x0A000000 /O2 /Ob1 /DNDEBUG"
  CMAKE_EXE_LINKER_FLAGS_INIT="/profile /DYNAMICBASE"
  CMAKE_MODULE_LINKER_FLAGS_INIT="/profile /DYNAMICBASE"
  CMAKE_SHARED_LINKER_FLAGS_INIT="/profile /DYNAMICBASE"
  USE_CUDA="OFF"
  USE_ROCM="OFF"



C:\llama.cpp\Onnx-GenAI\build_cpu_pdb>cmake --build . --config RelWithDebInfo
MSBuild version 17.12.12+1cce77968 for .NET Framework

  Checking File Globs
  1>Checking Build System
  Building Custom Rule C:/llama.cpp/Onnx-GenAI/build_cpu_pdb/_deps/onnxruntime_extensions-src/CMakeLists.txt
  base64.cc
  ocos.cc
  string_tensor.cc
  string_utils.cc
  audio.cc
  audio_decoder.cc
  noexcep_operators.vcxproj -> C:\llama.cpp\Onnx-GenAI\build_cpu_pdb\lib\RelWithDebInfo\noexcep_operators.lib
  Building Custom Rule C:/llama.cpp/Onnx-GenAI/build_cpu_pdb/_deps/onnxruntime_extensions-src/CMakeLists.txt
  ocos_operators_placeholder.cc
  math.cc
  segment_extraction.cc
  segment_sum.cc
  test_for_odr_violations.cpp
  bpe_kernels.cc
  unicode.cc
  tokenizers.cc
  ocos_operators.vcxproj -> C:\llama.cpp\Onnx-GenAI\build_cpu_pdb\lib\RelWithDebInfo\ocos_operators.lib
  Building Custom Rule C:/llama.cpp/Onnx-GenAI/build_cpu_pdb/_deps/onnxruntime_extensions-src/CMakeLists.txt
  ops_registry.cc
  c_api_utils.cc
  c_api_tokenizer.cc
  tokenizer_impl.cc
  c_api_feature_extraction.cc
  speech_extractor.cc
  c_api_processor.cc
  image_processor.cc
  image_resample.c
  ortcustomops.vcxproj -> C:\llama.cpp\Onnx-GenAI\build_cpu_pdb\lib\RelWithDebInfo\ortcustomops.lib
  Building Custom Rule C:/llama.cpp/Onnx-GenAI/CMakeLists.txt
  beam_search_scorer.cpp
  config.cpp
  interface.cpp
  generators.cpp
  json.cpp
  logging.cpp
  adapters.cpp
  audio_processor.cpp
  captured_graph_pool.cpp
  debugging.cpp
  decoder_only.cpp
  decoder_only_pipeline.cpp
  embeddings.cpp
  env_utils.cpp
  extra_inputs.cpp
  gpt.cpp
  image_features.cpp
  input_ids.cpp
  kv_cache.cpp
  logits.cpp
  model.cpp
  multi_modal_vision_model.cpp
  position_inputs.cpp
  prompt_image_processor.cpp
  static_buffer.cpp
  utils.cpp
  whisper.cpp
  ort_genai_c.cpp
  runtime_settings.cpp
  search.cpp
  sequences.cpp
  softmax_cpu.cpp
  top_k_cpu.cpp
  onnxruntime-genai-static.vcxproj -> C:\llama.cpp\Onnx-GenAI\build_cpu_pdb\RelWithDebInfo\onnxruntime-genai-static.lib
  Building Custom Rule C:/llama.cpp/Onnx-GenAI/benchmark/c/CMakeLists.txt
  main.cpp
  options.cpp
  resource_utils.cpp
LINK : warning LNK4075: ignoring '/INCREMENTAL' due to '/PROFILE' specification [C:\llama.cpp\Onnx-GenAI\build_cpu_pdb\
benchmark\c\model_benchmark.vcxproj]
     Creating library C:/llama.cpp/Onnx-GenAI/build_cpu_pdb/benchmark/c/RelWithDebInfo/model_benchmark.lib and object C
  :/llama.cpp/Onnx-GenAI/build_cpu_pdb/benchmark/c/RelWithDebInfo/model_benchmark.exp
  model_benchmark.vcxproj -> C:\llama.cpp\Onnx-GenAI\build_cpu_pdb\benchmark\c\RelWithDebInfo\model_benchmark.exe
  Building Custom Rule C:/llama.cpp/Onnx-GenAI/CMakeLists.txt
  beam_search_scorer.cpp
  config.cpp
  interface.cpp
  generators.cpp
  json.cpp
  logging.cpp
  adapters.cpp
  audio_processor.cpp
  captured_graph_pool.cpp
  debugging.cpp
  decoder_only.cpp
  decoder_only_pipeline.cpp
  embeddings.cpp
  env_utils.cpp
  extra_inputs.cpp
  gpt.cpp
  image_features.cpp
  input_ids.cpp
  kv_cache.cpp
  logits.cpp
  model.cpp
  multi_modal_vision_model.cpp
  position_inputs.cpp
  prompt_image_processor.cpp
  static_buffer.cpp
  utils.cpp
  whisper.cpp
  ort_genai_c.cpp
  runtime_settings.cpp
  search.cpp
  sequences.cpp
  softmax_cpu.cpp
  top_k_cpu.cpp
LINK : warning LNK4075: ignoring '/INCREMENTAL' due to '/PROFILE' specification [C:\llama.cpp\Onnx-GenAI\build_cpu_pdb\
onnxruntime-genai.vcxproj]
     Creating library C:/llama.cpp/Onnx-GenAI/build_cpu_pdb/RelWithDebInfo/onnxruntime-genai.lib and object C:/llama.cp
  p/Onnx-GenAI/build_cpu_pdb/RelWithDebInfo/onnxruntime-genai.exp
  onnxruntime-genai.vcxproj -> C:\llama.cpp\Onnx-GenAI\build_cpu_pdb\RelWithDebInfo\onnxruntime-genai.dll
  Building Custom Rule C:/llama.cpp/Onnx-GenAI/CMakeLists.txt

C:\llama.cpp\Onnx-GenAI\build_cpu_pdb>

C:\llama.cpp\Onnx-GenAI\build_cpu_pdb>dir /s *.dll *.exe
 Volume in drive C is Local Disk
 Volume Serial Number is 82FA-0DBF

 Directory of C:\llama.cpp\Onnx-GenAI\build_cpu_pdb\benchmark\c\RelWithDebInfo

12/13/2024  05:41 PM         1,433,600 model_benchmark.exe
               1 File(s)      1,433,600 bytes

 Directory of C:\llama.cpp\Onnx-GenAI\build_cpu_pdb\CMakeFiles\3.29.2\CompilerIdC

12/13/2024  05:37 PM            14,848 CompilerIdC.exe
               1 File(s)         14,848 bytes

 Directory of C:\llama.cpp\Onnx-GenAI\build_cpu_pdb\CMakeFiles\3.29.2\CompilerIdCXX

12/13/2024  05:37 PM            15,360 CompilerIdCXX.exe
               1 File(s)         15,360 bytes

 Directory of C:\llama.cpp\Onnx-GenAI\build_cpu_pdb\RelWithDebInfo

12/13/2024  05:42 PM         1,416,704 onnxruntime-genai.dll
               1 File(s)      1,416,704 bytes

 Directory of C:\llama.cpp\Onnx-GenAI\build_cpu_pdb\_deps\dlib-src\docs\docs\chm\htmlhelp

12/13/2024  05:39 PM           837,904 hha.dll
12/13/2024  05:39 PM           154,352 itcc.dll
12/13/2024  05:39 PM           155,552 itircl.dll
12/13/2024  05:39 PM           138,048 itss.dll

 Directory of C:\llama.cpp\Onnx-GenAI\build_cpu_pdb\_deps\dlib-src\docs\docs\chm\htmlhelp

12/13/2024  05:39 PM            51,472 hhc.exe
               5 File(s)      1,337,328 bytes

 Directory of C:\llama.cpp\Onnx-GenAI\build_cpu_pdb\_deps\onnxruntime_extensions-src\cmake\externals\git.Win32.2.41.03.patch

12/13/2024  05:39 PM         3,458,952 msys-2.0.dll
12/13/2024  05:39 PM           103,166 msys-gcc_s-1.dll

 Directory of C:\llama.cpp\Onnx-GenAI\build_cpu_pdb\_deps\onnxruntime_extensions-src\cmake\externals\git.Win32.2.41.03.patch

12/13/2024  05:39 PM           207,279 patch.exe
               3 File(s)      3,769,397 bytes

 Directory of C:\llama.cpp\Onnx-GenAI\build_cpu_pdb\_deps\ortlib-src\runtimes\win-arm64\native

12/13/2024  05:38 PM        11,761,696 onnxruntime.dll
12/13/2024  05:38 PM            21,048 onnxruntime_providers_shared.dll
               2 File(s)     11,782,744 bytes

 Directory of C:\llama.cpp\Onnx-GenAI\build_cpu_pdb\_deps\ortlib-src\runtimes\win-x64\native

12/13/2024  05:38 PM        11,532,848 onnxruntime.dll
12/13/2024  05:38 PM            22,048 onnxruntime_providers_shared.dll
               2 File(s)     11,554,896 bytes

 Directory of C:\llama.cpp\Onnx-GenAI\build_cpu_pdb\_deps\ortlib-src\runtimes\win-x86\native

12/13/2024  05:38 PM        10,169,912 onnxruntime.dll
12/13/2024  05:38 PM            20,536 onnxruntime_providers_shared.dll
               2 File(s)     10,190,448 bytes

     Total Files Listed:
              18 File(s)     41,515,325 bytes
               0 Dir(s)  362,728,431,616 bytes free


===========================================================================================


commit 7ade78de57e6c04b709fd9fece0749dec81dad2a (HEAD -> hv/dc_matmul)
Merge: 81cf4fa 66e8817
Author: hoivb612 <hoivo63@gmail.com>
Date:   Wed Jan 29 12:00:33 2025 -0800

    Merge remote-tracking branch 'origin/main' into hv/dc_matmul

commit 66e8817c00c3690b8b717853c694096c7b72893a (origin/main, origin/HEAD, main)
Author: Abhishek Jindal <abjindal@microsoft.com>
Date:   Mon Jan 27 14:05:00 2025 -0800

    Update examples and prompts (#1199)

    - Add default system prompt templates for phi2, phi3, phi4, llama2, and
    llama3 models to improve the user experience and provide more accurate
    responses.
    - Improve chat templates for phi2, phi3, phi4, llama2 and llama3 models
    to enhance the user experience
    - Add info about system prompt


=====================================================================================================================
