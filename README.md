# notebook
some notification about deep learning

## 🐛 Accelerate 报错：`__init__() got an unexpected keyword argument 'debug'`

### ❓ 报错信息

运行 `./run.sh` 时，出现如下报错：

```bash
Traceback (most recent call last):
  File ".../accelerate/commands/config/config_args.py", line 135, in from_yaml_file
    return cls(**config_dict)
TypeError: __init__() got an unexpected keyword argument 'debug'

该错误是由于 accelerate launch 默认会读取缓存目录下的配置文件（例如：~/.cache/huggingface/accelerate/default_config.yaml），其中包含了当前版本不支持的字段 debug，导致构造配置类时报错。

### ✅ 方法重新 ：配置 Accelerate

通过命令重新生成一个干净的配置文件，避免遗留字段污染：

accelerate config

配置过程中会出现提示，例如：

What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]:

    如果你只打算使用 GPU 4，输入：4

    如果打算使用所有 GPU，直接按回车即可

配置完后，将自动生成新的默认配置文件，之后运行 ./run.sh 即可正常启动。


## 🐛 Diffusers 报错：`Cannot import "cached_download" from "huggingface_hub"`

### ❓ 报错信息  

This issue seems to arise from a breaking change in huggingface_hub version 0.26.0, 
where the cached_download() function has been fully removed.
Downgrading to an earlier version of huggingface_hub should resolve this problem.

```bash
pip install huggingface_hub==0.23.0

```bash
pip install huggingface_hub==0.23.0
