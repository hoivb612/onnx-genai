https://github.com/hoivb612/onnx-genai

==================

C:\llama.cpp\Onnx-GenAI>set path=e:\xCloud_Reuse\Python38.diffusers.v2;%path%

C:\llama.cpp\Onnx-GenAI>build --use_dml --skip_wheel

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

 Directory of C:\llama.cpp\phi-Onnx\examples\c\build\RelWithDebInfo

10/02/2024  02:20 PM    <DIR>          .
10/02/2024  11:18 AM    <DIR>          ..
10/02/2024  01:59 PM         3,328,032 D3D12Core.dll
10/02/2024  01:59 PM        18,527,264 DirectML.dll
10/02/2024  01:58 PM         2,114,560 onnxruntime-genai.dll
10/02/2024  01:59 PM        14,613,552 onnxruntime.dll
10/02/2024  11:18 AM           100,864 phi3.exe
10/02/2024  11:18 AM         1,445,888 phi3.pdb
               6 File(s)     40,130,160 bytes

===================================

C:\llama.cpp\Onnx-GenAI>git fetch --all
warning: fetch normally indicates which branches had a forced update,
but that check has been disabled; to re-enable, use '--show-forced-updates'
flag or run 'git config fetch.showForcedUpdates true'

C:\llama.cpp\Onnx-GenAI>git merge origin/main
Merge made by the 'ort' strategy.
 .github/policies/issueLabeler.yml                  |  43 +-
 .github/workflows/linux-cpu-arm64-build.yml        |   2 +-
 .github/workflows/linux-cpu-x64-build.yml          |  10 +-
 .github/workflows/linux-cpu-x64-nightly-build.yml  |   3 +-
 .github/workflows/linux-gpu-x64-build.yml          |   7 +-
 .github/workflows/mac-cpu-arm64-build.yml          |  15 +-
 .github/workflows/win-cpu-arm64-build.yml          |   4 +-
 .github/workflows/win-cpu-x64-build.yml            |   9 +-
 .github/workflows/win-cuda-x64-build.yml           |  20 +-
 .github/workflows/win-directml-x64-build.yml       |   7 +-
 .pipelines/codeql.yaml                             |   4 +-
 .pipelines/nuget-publishing.yml                    |  13 +-
 .pipelines/pypl-publishing.yml                     |  37 +-
 .pipelines/stages/capi-packaging-stage.yml         |   8 +
 .pipelines/stages/jobs/android-java-api-aar.yml    |   7 +-
 .pipelines/stages/jobs/capi-packaging-job.yml      |  47 +-
 .pipelines/stages/jobs/nuget-packaging-job.yml     |  63 ++-
 .pipelines/stages/jobs/nuget-validation-job.yml    | 119 ++---
 .pipelines/stages/jobs/py-packaging-job.yml        |  43 +-
 .pipelines/stages/jobs/py-validation-job.yml       | 120 ++---
 .../stages/jobs/steps/capi-appleframework-step.yml |  10 +-
 .pipelines/stages/jobs/steps/capi-linux-step.yml   |   1 -
 .pipelines/stages/jobs/steps/capi-macos-step.yml   |   1 -
 .pipelines/stages/jobs/steps/capi-win-step.yml     |  19 -
 .../jobs/steps/compliant-and-cleanup-step.yml      |   3 +-
 .../stages/jobs/steps/nuget-validation-step.yml    |  88 ++++
 .../stages/jobs/steps/python-validation-step.yml   |  97 ++++
 .../steps/utils/download-huggingface-model.yml     |  12 +-
 .../utils/flex-download-pipeline-artifact.yml      |   1 +
 .../jobs/steps/utils/set-cmake-build-type.yml      |  23 +
 .pipelines/stages/nuget-packaging-stage.yml        |  10 +-
 .pipelines/stages/py-packaging-stage.yml           |  21 -
 .pipelines/stages/py-validation-stage.yml          |  19 -
 CMakeLists.txt                                     |  38 +-
 README.md                                          |   4 +-
 VERSION_INFO                                       |   2 +-
 benchmark/c/main.cpp                               |  30 +-
 benchmark/c/options.h                              |   4 +-
 benchmark/python/README                            |   2 +-
 benchmark/python/benchmark_e2e.py                  |  12 +-
 benchmark/python/benchmark_e2e_continuous_test.py  | 127 ++++++
 build.py                                           |   8 +
 cmake/check_cuda.cmake                             |  12 +-
 cmake/check_webgpu.cmake                           |   6 +
 cmake/global_variables.cmake                       |   2 +
 cmake/options.cmake                                |   2 +-
 cmake/ortlib.cmake                                 |   5 +-
 cmake/package.cmake                                |   8 +
 documents/Runtime_option.md                        |  15 +
 examples/c/src/phi3.cpp                            | 136 ++++--
 examples/c/src/phi3v.cpp                           |   2 -
 examples/chat_app/app.py                           |  24 +-
 .../interface/multimodal_onnx_interface.py         |  21 +-
 examples/csharp/HelloPhi/HelloPhi.csproj           |   6 +-
 examples/csharp/HelloPhi/Program.cs                |  14 +-
 examples/csharp/HelloPhi3V/HelloPhi3V.csproj       |   6 +-
 examples/csharp/HelloPhi3V/Program.cs              | 199 +++++---
 examples/python/awq-quantized-model.md             |  12 +-
 examples/python/awq-quantized-model.py             |   6 +-
 examples/python/model-generate.py                  |  52 ++-
 examples/python/model-qa.py                        |  31 +-
 examples/python/phi-3-vision.md                    |  48 +-
 examples/python/phi-3.5-vision.md                  | 118 +++++
 examples/python/phi3v.py                           |  95 +++-
 nuget/Microsoft.ML.OnnxRuntimeGenAI.Managed.nuspec |  13 +
 nuget/PACKAGE.md                                   |   8 +-
 .../Microsoft.ML.OnnxRuntimeGenAI.targets          |   8 +
 .../Microsoft.ML.OnnxRuntimeGenAI.targets          |  13 +
 nuget/targets/net8.0-maccatalyst/README.md         |   3 +
 nuget/targets/net8.0-maccatalyst/_._               |   0
 .../Microsoft.ML.OnnxRuntimeGenAI.props            |   0
 .../Microsoft.ML.OnnxRuntimeGenAI.targets          |   0
 src/beam_search_scorer.cpp                         |  69 +--
 src/beam_search_scorer.h                           |  30 +-
 src/beam_search_scorer_cuda.h                      |  54 ---
 src/config.cpp                                     |  79 +++-
 src/config.h                                       |  18 +-
 src/cpu/interface.cpp                              |  60 +++
 src/cpu/interface.h                                |   8 +
 src/csharp/Adapters.cs                             |  62 +++
 src/csharp/Config.cs                               |  55 +++
 src/csharp/Exceptions.cs                           |   1 -
 src/csharp/Generator.cs                            |  49 +-
 src/csharp/GeneratorParams.cs                      |  16 -
 src/csharp/Microsoft.ML.OnnxRuntimeGenAI.csproj    |  38 +-
 src/csharp/Model.cs                                |  10 +-
 src/csharp/MultiModalProcessor.cs                  |   1 -
 src/csharp/NativeMethods.cs                        | 100 +++--
 src/csharp/Result.cs                               |   1 -
 src/csharp/Sequences.cs                            |   1 -
 src/csharp/Tensor.cs                               |  51 ++-
 src/csharp/Tokenizer.cs                            |   2 -
 src/csharp/TokenizerStream.cs                      |   1 -
 src/csharp/Utils.cs                                |   1 -
 src/{ => cuda}/beam_search_scorer_cuda.cpp         |  54 +--
 src/{ => cuda}/beam_search_scorer_cuda.cu          |  16 +-
 src/{ => cuda}/beam_search_scorer_cuda.cuh         |   1 +
 src/cuda/beam_search_scorer_cuda.h                 |  40 ++
 src/{ => cuda}/beam_search_topk.cu                 |   0
 src/cuda/cuda_common.h                             |  88 ++++
 src/{ => cuda}/cuda_sampling.cu                    |  57 ++-
 src/{ => cuda}/cuda_sampling.cuh                   |   5 +-
 src/cuda/exported_symbols.lst                      |   1 +
 src/cuda/interface.cpp                             | 247 ++++++++++
 src/cuda/interface.h                               |  50 +++
 src/{models/kernels.cu => cuda/model_kernels.cu}   | 104 +++--
 src/{ => cuda}/search_cuda.cpp                     | 170 ++++---
 src/{ => cuda}/search_cuda.cu                      |  61 ++-
 src/{ => cuda}/search_cuda.cuh                     |   6 +-
 src/{ => cuda}/search_cuda.h                       |  48 +-
 src/cuda/symbols.def                               |   2 +
 src/cuda/version_script.lds                        |   9 +
 src/cuda_common.h                                  |  12 -
 src/dml/dml_helpers.cpp                            |  24 +-
 src/dml/dml_helpers.h                              |   2 +-
 src/generators.cpp                                 | 291 +++++++++---
 src/generators.h                                   |  39 +-
 .../src/main/java/ai/onnxruntime/genai/Config.java |  59 +++
 .../main/java/ai/onnxruntime/genai/Generator.java  |  47 +-
 .../java/ai/onnxruntime/genai/GeneratorParams.java |  55 ---
 .../src/main/java/ai/onnxruntime/genai/Model.java  |  28 +-
 .../java/ai/onnxruntime/genai/SimpleGenAI.java     |  53 +--
 .../src/main/java/ai/onnxruntime/genai/Tensor.java |  14 +-
 .../main/native/ai_onnxruntime_genai_Config.cpp    |  52 +++
 .../main/native/ai_onnxruntime_genai_Generator.cpp |  25 +-
 .../ai_onnxruntime_genai_GeneratorParams.cpp       |  20 -
 .../src/main/native/ai_onnxruntime_genai_Model.cpp |  25 +-
 .../example/javavalidator/SimpleTest.kt            |  10 +-
 .../java/ai/onnxruntime/genai/GenerationTest.java  |  24 +-
 .../ai/onnxruntime/genai/GeneratorParamsTest.java  |   4 +-
 .../onnxruntime/genai/MultiModalProcessorTest.java |   1 -
 .../test/java/ai/onnxruntime/genai/TensorTest.java |   2 +-
 src/json.cpp                                       |   3 +-
 src/models/adapters.cpp                            |  73 +++
 src/models/adapters.h                              |  47 ++
 src/models/debugging.cpp                           |  18 +-
 src/models/debugging.h                             |   4 -
 src/models/decoder_only.cpp                        |  30 +-
 src/models/decoder_only.h                          |  10 +-
 src/models/decoder_only_pipeline.cpp               |  35 +-
 src/models/decoder_only_pipeline.h                 |  16 +-
 src/models/embeddings.cpp                          |  48 +-
 src/models/embeddings.h                            |   2 +-
 src/models/gpt.cpp                                 |  28 +-
 src/models/gpt.h                                   |  10 +-
 src/models/image_features.cpp                      |   4 +-
 src/models/image_features.h                        |   2 +-
 src/models/input_ids.cpp                           | 126 +++---
 src/models/input_ids.h                             |   8 +-
 src/models/kernels.h                               |   7 +-
 src/models/kv_cache.cpp                            | 218 +++++++--
 src/models/kv_cache.h                              |  27 +-
 src/models/logits.cpp                              |  98 ++--
 src/models/logits.h                                |  18 +-
 src/models/model.cpp                               | 153 +++++--
 src/models/model.h                                 |  35 +-
 src/models/multi_modal_vision_model.cpp            |  62 ++-
 src/models/multi_modal_vision_model.h              |  35 +-
 src/models/onnxruntime_api.h                       |  14 +
 src/models/onnxruntime_inline.h                    |  11 +
 src/models/position_inputs.cpp                     | 488 +++++++++++---------
 src/models/position_inputs.h                       |  43 +-
 src/models/whisper.cpp                             |  67 +--
 src/models/whisper.h                               |  12 +-
 src/ort_genai.h                                    | 106 ++++-
 src/ort_genai_c.cpp                                | 209 ++++++---
 src/ort_genai_c.h                                  | 184 ++++++--
 src/python/CMakeLists.txt                          |   3 +-
 src/python/py/models/README.md                     |  16 +
 src/python/py/models/builder.py                    | 498 +++++++++++++++++----
 src/python/py/models/quantized_model.py            | 156 +++++--
 src/python/python.cpp                              | 131 +++---
 src/python/setup.py.in                             |  13 +-
 src/runtime_settings.cpp                           |  43 ++
 src/runtime_settings.h                             |  20 +
 src/search.cpp                                     | 181 +++++---
 src/search.h                                       |  66 +--
 src/sequences.cpp                                  |  72 +--
 src/sequences.h                                    |  38 +-
 src/sequences_cuda.cpp                             |  67 ---
 src/sequences_cuda.cu                              |  36 --
 src/sequences_cuda.h                               |  34 --
 src/smartptrs.h                                    | 278 ++++--------
 test/CMakeLists.txt                                |  14 +-
 test/c_api_tests.cpp                               | 451 +++++++++++++++----
 .../Microsoft.ML.OnnxRuntimeGenAI.Tests.csproj     |   8 +
 test/csharp/TestOnnxRuntimeGenAIAPI.cs             | 420 +++++++++++++----
 test/model_tests.cpp                               | 154 ++++---
 test/python/cpu/ort/requirements.txt               |   2 +
 test/python/cpu/torch/requirements.txt             |   2 +
 test/python/cuda/ort/requirements.txt              |   2 +
 test/python/cuda/torch/requirements.txt            |   2 +
 test/python/directml/ort/requirements.txt          |   2 +
 test/python/directml/torch/requirements.txt        |   2 +
 test/python/macos/ort/requirements.txt             |   2 +
 test/python/macos/torch/requirements.txt           |   2 +
 test/python/requirements-cpu.txt                   |   4 -
 test/python/requirements-cuda.txt                  |   4 -
 test/python/requirements-directml.txt              |   4 -
 test/python/requirements-macos.txt                 |   4 -
 test/python/test_onnxruntime_genai_api.py          | 294 +++++++++++-
 test/python/test_onnxruntime_genai_e2e.py          |  15 +-
 test/sampling_benchmark.cpp                        | 402 ++++++-----------
 test/sampling_tests.cpp                            | 208 ++++-----
 test/tests_helper.cu                               |  19 +-
 tools/ci_build/github/android/build_aar_package.py |  11 +-
 tools/nuget/generate_nuspec_for_native_nuget.py    |  33 +-
 tools/python/model_validation/README.md            |  59 +++
 .../python/model_validation/perplexity_metrics.py  |  90 ++++
 tools/python/model_validation/requirements.txt     |  15 +
 .../python/model_validation/validation_config.json |  47 ++
 tools/python/model_validation/validation_tool.py   | 131 ++++++
 212 files changed, 7365 insertions(+), 3372 deletions(-)
 create mode 100644 .pipelines/stages/jobs/steps/nuget-validation-step.yml
 create mode 100644 .pipelines/stages/jobs/steps/python-validation-step.yml
 create mode 100644 .pipelines/stages/jobs/steps/utils/set-cmake-build-type.yml
 create mode 100644 benchmark/python/benchmark_e2e_continuous_test.py
 create mode 100644 cmake/check_webgpu.cmake
 create mode 100644 documents/Runtime_option.md
 create mode 100644 examples/python/phi-3.5-vision.md
 create mode 100644 nuget/targets/net8.0-android/Microsoft.ML.OnnxRuntimeGenAI.targets
 create mode 100644 nuget/targets/net8.0-ios/Microsoft.ML.OnnxRuntimeGenAI.targets
 create mode 100644 nuget/targets/net8.0-maccatalyst/README.md
 create mode 100644 nuget/targets/net8.0-maccatalyst/_._
 rename nuget/targets/{ => netstandard}/Microsoft.ML.OnnxRuntimeGenAI.props (100%)
 rename nuget/targets/{ => netstandard}/Microsoft.ML.OnnxRuntimeGenAI.targets (100%)
 delete mode 100644 src/beam_search_scorer_cuda.h
 create mode 100644 src/cpu/interface.cpp
 create mode 100644 src/cpu/interface.h
 create mode 100644 src/csharp/Adapters.cs
 create mode 100644 src/csharp/Config.cs
 rename src/{ => cuda}/beam_search_scorer_cuda.cpp (65%)
 rename src/{ => cuda}/beam_search_scorer_cuda.cu (95%)
 rename src/{ => cuda}/beam_search_scorer_cuda.cuh (98%)
 create mode 100644 src/cuda/beam_search_scorer_cuda.h
 rename src/{ => cuda}/beam_search_topk.cu (100%)
 create mode 100644 src/cuda/cuda_common.h
 rename src/{ => cuda}/cuda_sampling.cu (92%)
 rename src/{ => cuda}/cuda_sampling.cuh (90%)
 create mode 100644 src/cuda/exported_symbols.lst
 create mode 100644 src/cuda/interface.cpp
 create mode 100644 src/cuda/interface.h
 rename src/{models/kernels.cu => cuda/model_kernels.cu} (79%)
 rename src/{ => cuda}/search_cuda.cpp (60%)
 rename src/{ => cuda}/search_cuda.cu (58%)
 rename src/{ => cuda}/search_cuda.cuh (53%)
 rename src/{ => cuda}/search_cuda.h (64%)
 create mode 100644 src/cuda/symbols.def
 create mode 100644 src/cuda/version_script.lds
 delete mode 100644 src/cuda_common.h
 create mode 100644 src/java/src/main/java/ai/onnxruntime/genai/Config.java
 create mode 100644 src/java/src/main/native/ai_onnxruntime_genai_Config.cpp
 create mode 100644 src/models/adapters.cpp
 create mode 100644 src/models/adapters.h
 create mode 100644 src/runtime_settings.cpp
 create mode 100644 src/runtime_settings.h
 delete mode 100644 src/sequences_cuda.cpp
 delete mode 100644 src/sequences_cuda.cu
 delete mode 100644 src/sequences_cuda.h
 create mode 100644 test/python/cpu/ort/requirements.txt
 create mode 100644 test/python/cpu/torch/requirements.txt
 create mode 100644 test/python/cuda/ort/requirements.txt
 create mode 100644 test/python/cuda/torch/requirements.txt
 create mode 100644 test/python/directml/ort/requirements.txt
 create mode 100644 test/python/directml/torch/requirements.txt
 create mode 100644 test/python/macos/ort/requirements.txt
 create mode 100644 test/python/macos/torch/requirements.txt
 delete mode 100644 test/python/requirements-cpu.txt
 delete mode 100644 test/python/requirements-cuda.txt
 delete mode 100644 test/python/requirements-directml.txt
 delete mode 100644 test/python/requirements-macos.txt
 create mode 100644 tools/python/model_validation/README.md
 create mode 100644 tools/python/model_validation/perplexity_metrics.py
 create mode 100644 tools/python/model_validation/requirements.txt
 create mode 100644 tools/python/model_validation/validation_config.json
 create mode 100644 tools/python/model_validation/validation_tool.py

C:\llama.cpp\Onnx-GenAI>git log
commit 991a794d151e1e339b4e45b13c200dfffe0686cd (HEAD -> hv/dc_matmul)
Merge: 52d6779 c5745fd
Author: hoivb612 <hoivo63@gmail.com>
Date:   Wed Nov 27 13:22:15 2024 -0800

    Merge remote-tracking branch 'origin/main' into hv/dc_matmul

commit c5745fd6d91a2ba2d94c6a52f6255652dde02508 (origin/main, origin/HEAD)
Author: Ryan Hill <38674843+RyanUnderhill@users.noreply.github.com>
Date:   Mon Nov 25 16:55:30 2024 -0800

    Debug build fixes missed by pipelines (#1101)

    Looks like our pipelines missed these build errors due to lack of debug
    builds for cuda & windows cuda.

commit e27e2b577dba7da8d2c7da247f5692685cc41ffe
Author: Guenther Schmuelling <guschmue@microsoft.com>
Date:   Sat Nov 23 09:49:55 2024 -0800

    handle webgpu case for positionID updates after continuation change (#1095)

================================================

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

