# 服务器训练模型指引

在服务器上训练模型是深度学习项目的标准流程，因为服务器通常拥有更强大的计算资源（尤其是 GPU）。本指引将引导您完成从连接服务器到获取训练结果的全过程。

## 过程概述

1.  **连接服务器并配置环境**：在服务器上搭建一个和您本地相似的 Python 环境。
2.  **上传项目文件**：将您的代码和数据传输到服务器。
3.  **开始训练**：在服务器上以后台模式运行您的训练脚本。
4.  **监控与下载结果**：查看训练进度，并将生成的模型或结果文件下载回本地。

---

### 第一步：连接服务器并配置环境

#### 1. 连接到服务器
您需要一个 SSH 客户端来连接。Windows 系统现在自带了 OpenSSH，您可以直接在 **命令提示符 (cmd)** 或 **PowerShell** 中使用。

```shell
# 将 username 替换为您的服务器用户名，server_ip_address 替换为服务器的IP地址
ssh username@server_ip_address
```
连接时会提示您输入密码。

#### 2. 检查 GPU（如果可用）
连接成功后，首先检查服务器是否有可用的 NVIDIA GPU。
```shell
nvidia-smi
```
如果这个命令成功执行并显示了 GPU 信息和 CUDA 版本，您就可以使用 GPU 进行训练。如果提示命令未找到，说明服务器可能没有 NVIDIA GPU 或驱动未安装，您将只能使用 CPU 训练。

#### 3. 配置 Python 环境
为了不与服务器上其他用户的环境冲突，强烈建议创建一个独立的虚拟环境。推荐使用 `conda` 或 `venv`。

**方法 A：使用 Conda (推荐)**
```shell
# 1. 创建一个新的 conda 环境（例如，命名为 climate_env）
#    将 python=3.12 替换为您需要的 Python 版本
conda create --name climate_env python=3.12

# 2. 激活环境
conda activate climate_env

# 3. 安装依赖包
#    (您需要先将项目上传，见第二步，然后进入项目目录)
#    假设您的 requirements.txt 文件是完整的
pip install -r requirements.txt

#    如果需要安装 PyTorch，请务必根据服务器的 CUDA 版本选择正确的命令
#    例如，如果 nvidia-smi 显示 CUDA Version: 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**方法 B：使用 venv 和 uv (如果您本地在使用)**
```shell
# 1. 创建虚拟环境
python3 -m venv .venv

# 2. 激活环境
#    在 Linux/macOS 上:
source .venv/bin/activate
#    在 Windows Server 上 (虽然少见):
#    .venv\Scripts\activate

# 3. 使用 pip 或 uv 安装依赖
pip install -r requirements.txt
# 或者，如果服务器上安装了 uv
uv pip install -r requirements.txt
```

---

### 第二步：上传您的项目文件

在**您本地的电脑**上打开一个新的终端（不要在 SSH 连接里），使用 `scp` 命令来上传整个项目文件夹。

```shell
# 语法: scp -r [本地文件夹路径] [服务器用户名]@[服务器IP]:[服务器目标路径]
# 示例：将本地的 E:\Climate-D-S 文件夹上传到服务器用户的主目录下

# 注意：Windows 路径中的反斜杠 \ 可能需要转换
# -r 表示递归复制整个文件夹
scp -r E:\Climate-D-S username@server_ip_address:~/
```
*   `~/` 代表服务器上您的用户主目录。传输可能需要一些时间，具体取决于您的数据大小和网络速度。

---

### 第三步：开始训练

现在，回到您与服务器连接的 SSH 终端中。

#### 1. 进入项目目录
```shell
# 如果您上传到了主目录，那么现在它应该在那里
cd Climate-D-S
```

#### 2. 激活虚拟环境
```shell
# 如果使用 conda
conda activate climate_env

# 如果使用 venv
source .venv/bin/activate
```

#### 3. 以后台模式运行训练脚本
直接运行脚本会在您关闭 SSH 连接后中断。为了让训练在您离线后继续进行，我们使用 `nohup` 和 `&`。

*   `nohup` (no hang up): 即使终端关闭，命令也会继续运行。
*   `&`: 让命令在后台执行。
*   `>`: 将标准输出重定向到文件，方便您查看日志。

```shell
# 运行训练，并将所有输出日志保存到 train.log 文件中
nohup python scripts/train_eval.py --config config.yaml > train.log 2>&1 &
```
*   `2>&1` 的意思是将错误输出也重定向到与标准输出相同的地方（即 `train.log`）。
*   执行后，会显示一个进程ID（PID），例如 `[1] 12345`。

---

### 第四步：监控与下载结果

#### 1. 实时查看训练日志
您可以使用 `tail` 命令来实时查看 `train.log` 文件的最新内容。
```shell
tail -f train.log
```
*   按 `Ctrl + C` 可以退出 `tail` 的实时查看模式，但训练任务仍在后台运行。

#### 2. 检查训练进程是否仍在运行
```shell
ps aux | grep train_eval.py
```
如果能看到您的 python 进程，说明训练还在进行。

#### 3. 下载结果
训练完成后，您在 `outputs` 目录下生成的 `run_name` 文件夹就是您的结果。您可以在**本地电脑**上使用 `scp` 将其下载回来。

```shell
# 语法: scp -r [服务器用户名]@[服务器IP]:[服务器上的文件夹路径] [本地目标路径]
# 示例：将服务器上的某个结果文件夹下载到本地的 E:\Downloads

scp -r username@server_ip_address:~/Climate-D-S/outputs/stT_sst-u10-v10_... E:\Downloads
```

---

### 高级技巧与建议

*   **使用 `tmux` 或 `screen`**：`nohup` 非常简单，但不够灵活。`tmux` 和 `screen` 是更强大的终端复用工具，它们可以创建持久化的会话。您可以断开 SSH，下次再连接时可以“附加”回之前的会话，看到程序运行的实时界面，就像从未离开过一样。这是在服务器上运行长任务的**首选方式**。

*   **检查配置文件**：上传前，请确保 `config.yaml` 中的数据路径 (`data.path`) 和输出路径 (`output.output_dir`) 对于服务器的文件系统是正确的。例如，您可能需要将其从本地的 `E:\Climate-D-S\data` 改为服务器上的 `~/Climate-D-S/data`。不过，如果您使用了相对路径（如 `../data`），并且在项目根目录运行脚本，那么通常无需修改。

