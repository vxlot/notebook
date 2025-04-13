# export HF_ENDPOINT=https://hf-mirror.com
# export HF_HOME="/ifs/root/ipa01/110/user_110003/download"

from datasets import load_dataset
from datasets import get_dataset_split_names

# 这个数据集太大了 可以只下载一个parque文件测试
# dataset = load_dataset("yuvalkirstain/pickapic_v2")


dataset = load_dataset("yuvalkirstain/pickapic_v2", cache_dir="/ifs/root/ipa01/110/user_110003/picapic_v2")
# ds = load_dataset("yuvalkirstain/pickapic_v1")

# from datasets import load_dataset
# import io
# from PIL import Image


# dataset = load_dataset("parquet", data_files="/ifs/root/ipa01/110/user_110003/download/hub/datasets--yuvalkirstain--pickapic_v2/snapshots/12d45c8a6fcbc35c18a067efb24d993caaf4b8a7/data/test-00000-of-00014-387db523fa7e7121.parquet")
print(dataset['train'].column_names)
print(dataset['train']['label'][0])

# im_scalar = dataset['train']['jpg_0'][0]  
# im_bytes = im_scalar.as_buffer().to_pybytes() 
# Image.open(io.BytesIO(im_bytes)).convert("RGB").save("jpg_0.png")

# im_scalar = dataset.data['train']['jpg_1'][0]  
# im_bytes = im_scalar.as_buffer().to_pybytes() 
# Image.open(io.BytesIO(im_bytes)).convert("RGB").save("jpg_1.png")