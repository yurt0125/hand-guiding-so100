# HIL-SERL 复现操作流程（Rita 本机）

本文档记录在 `lerobot` 环境下复现 LeRobot HIL-SERL 的最小可跑流程。

## 0) 环境前提

- Conda 环境：`/home/rita/下载/enter/envs/lerobot`
- LeRobot 源码目录：`/home/rita/HandGuiding_test/project/third-party/lerobot`
- 机械臂：SO101 / SO100（按配置改）
- 相机：D435（可选 fixed RGB）

---

## 1) 激活环境

```bash
source /home/rita/下载/enter/etc/profile.d/conda.sh
conda activate lerobot
```

可选检查：

```bash
python -V
which python
```

---

## 2) 修复 / 对齐 PyTorch（已知稳定组合）

当出现 `torchaudio requires torch==...` 冲突时，使用：

```bash
pip uninstall -y torch torchaudio torchvision

pip install --no-cache-dir \
  torch==2.4.1+cu118 \
  torchaudio==2.4.1+cu118 \
  torchvision==0.19.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

验证：

```bash
python - <<'PY'
import torch, torchaudio, torchvision
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())
print('torchaudio', torchaudio.__version__)
print('torchvision', torchvision.__version__)
PY
```

---

## 3) 安装 HIL-SERL 关键依赖

```bash
pip install -i https://pypi.org/simple --no-cache-dir flax optax jax jaxlib
```

验证：

```bash
python - <<'PY'
import flax, optax, jax
print('flax', flax.__version__)
print('optax', optax.__version__)
print('jax', jax.__version__)
PY
```

---

## 4) 确认 HIL-SERL 入口可用

```bash
cd /home/rita/HandGuiding_test/project/third-party/lerobot

python -m lerobot.rl.learner --help
python -m lerobot.rl.actor --help
```

---

## 5) 最小复现（官方 so100 配置）

开两个终端。

### 终端 A：启动 learner

```bash
source /home/rita/下载/enter/etc/profile.d/conda.sh
conda activate lerobot
cd /home/rita/HandGuiding_test/project/third-party/lerobot

python -m lerobot.rl.learner \
  --config_path src/lerobot/configs/train_config_hilserl_so100.json
```

### 终端 B：启动 actor

```bash
source /home/rita/下载/enter/etc/profile.d/conda.sh
conda activate lerobot
cd /home/rita/HandGuiding_test/project/third-party/lerobot

python -m lerobot.rl.actor \
  --config_path src/lerobot/configs/train_config_hilserl_so100.json
```

---

## 6) 改成你的 SO101 配置

先复制一份配置：

```bash
cd /home/rita/HandGuiding_test/project/third-party/lerobot
cp src/lerobot/configs/train_config_hilserl_so100.json \
   src/lerobot/configs/train_config_hilserl_so101_rita.json
```

重点修改项：

- `env.robot.type` -> `so101_follower`
- `env.robot.port` -> `/dev/ttyACM1`
- `env.teleop.type` -> `so101_leader`（或先用 `keyboard`）
- `env.teleop.port` -> `/dev/ttyACM0`
- `env.robot.cameras`：
  - D435 使用键名 `d435`
  - 顶部 RGB 可用 `fixed`
- 数据集路径、任务名按你的实验改

然后用新配置启动：

```bash
python -m lerobot.rl.learner \
  --config_path src/lerobot/configs/train_config_hilserl_so101_rita.json

python -m lerobot.rl.actor \
  --config_path src/lerobot/configs/train_config_hilserl_so101_rita.json
```

---

## 7) 常见报错与处理

### 报错：`repo_id` 格式错误

需要 `owner/name` 形式，例如：

```bash
--dataset.repo_id=rita/test
```

### 报错：`torchaudio ... requires torch==...`

说明 torch 家族版本不一致，回到第 2 步重装对齐。

### 报错：缺 `flax` / `optax`

回到第 3 步安装。

### 报错：连接机械臂异常

先用你已稳定的录制/控制脚本确认串口与供电，再启动 actor。

---

## 8) 你的当前推荐实践

- 先在 `fps=15` 下跑稳（相机 + 训练链路）
- 再尝试 `fps=30`
- HIL-SERL 建议与日常 `lerobot` 采集流程分开验证，避免同时改太多变量

