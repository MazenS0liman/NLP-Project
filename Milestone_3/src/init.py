import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import torch 
torch.classes.__path__ = []
torch.cuda.init()

import transformers