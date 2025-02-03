FusionAI14:

02/02/25 using Vinal's gguf_model.py

C:\llama.cpp\Onnx-GenAI\src\python\py\models>e:\xCloud_Reuse\python312\python.exe builder.py -m microsoft/phi-2 -p fp32 -e cpu --extra_options xx_xxxxx="xx_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" -c c:\llama.cpp\models\Phi-2\microsoft-Phi-2 -o e:\llama.cpp\models\Phi-2\onnx
\gguf_onnx_2 -i e:\llama.cpp\models\Phi-2\microsoft-Phi-2-f32-dc_iqk.gguf
Valid precision + execution provider combinations are: FP32 CPU, FP32 CUDA, FP16 CUDA, FP16 DML, INT4 CPU, INT4 CUDA, INT4 DML
Extra options: {'xx_xxxxx': 'xx_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'}
config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 735/735 [00:00<?, ?B/s]
e:\xCloud_Reuse\python312\Lib\site-packages\huggingface_hub\file_download.py:139: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\llama.cpp\models\Phi-2\microsoft-Phi-2\models--microsoft--phi-2. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)
GroupQueryAttention (GQA) is used in this model.
Reading embedding layer
Reading decoder layer 0
Reading decoder layer 1
Reading decoder layer 2
Reading decoder layer 3
Reading decoder layer 4
Reading decoder layer 5
Reading decoder layer 6
Reading decoder layer 7
Reading decoder layer 8
Reading decoder layer 9
Reading decoder layer 10
Reading decoder layer 11
Reading decoder layer 12
Reading decoder layer 13
Reading decoder layer 14
Reading decoder layer 15
Reading decoder layer 16
Reading decoder layer 17
Reading decoder layer 18
Reading decoder layer 19
Reading decoder layer 20
Reading decoder layer 21
Reading decoder layer 22
Reading decoder layer 23
Reading decoder layer 24
Reading decoder layer 25
Reading decoder layer 26
Reading decoder layer 27
Reading decoder layer 28
Reading decoder layer 29
Reading decoder layer 30
Reading decoder layer 31
Reading final norm
Reading LM head
Saving ONNX model in e:\llama.cpp\models\Phi-2\onnx\gguf_onnx_2
generation_config.json: 100%|████████████████████████████████████████████████████████████████████████████████| 124/124 [00:00<?, ?B/s]
Saving GenAI config in e:\llama.cpp\models\Phi-2\onnx\gguf_onnx_2
tokenizer_config.json: 100%|█████████████████████████████████████████████████████████████████████████████| 7.34k/7.34k [00:00<?, ?B/s]
vocab.json: 100%|██████████████████████████████████████████████████████████████████████████████████| 798k/798k [00:00<00:00, 3.70MB/s]
merges.txt: 100%|██████████████████████████████████████████████████████████████████████████████████| 456k/456k [00:00<00:00, 3.25MB/s]
tokenizer.json: 100%|████████████████████████████████████████████████████████████████████████████| 2.11M/2.11M [00:00<00:00, 6.62MB/s]
added_tokens.json: 100%|█████████████████████████████████████████████████████████████████████████████████| 1.08k/1.08k [00:00<?, ?B/s]
special_tokens_map.json: 100%|█████████████████████████████████████████████████████████████████████████████| 99.0/99.0 [00:00<?, ?B/s]
Saving processing files in e:\llama.cpp\models\Phi-2\onnx\gguf_onnx_2 for GenAI

----------------------------------------------------------------

 Directory of E:\llama.cpp\models\Phi-2\onnx\gguf_onnx_2

02/02/2025  06:25 PM    <DIR>          .
02/02/2025  06:23 PM    <DIR>          ..
02/02/2025  06:25 PM             1,120 added_tokens.json
02/02/2025  06:25 PM             1,515 genai_config.json
02/02/2025  06:25 PM           456,318 merges.txt
02/02/2025  06:25 PM           172,222 model.onnx
02/02/2025  06:25 PM    11,118,997,504 model.onnx.data
02/02/2025  06:25 PM               464 special_tokens_map.json
02/02/2025  06:25 PM         3,564,952 tokenizer.json
02/02/2025  06:25 PM             7,728 tokenizer_config.json
02/02/2025  06:25 PM           798,156 vocab.json
               9 File(s) 11,123,999,979 bytes