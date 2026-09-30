# tft-sim：自走棋离线模拟器 + 强化学习 / 模仿学习

## 目标
离线自走棋模拟器（金铲铲/云顶之弈风格），用来训练一个会自己选海克斯、买牌、选秀、升级、对位的 AI。
**只在模拟器里跑，不接入、不操控真实游戏客户端。**

## 已确定的设计
- 语言：Python 3.10+；训练用 PyTorch（Apple Silicon 用 `mps`）
- 复杂度：共享卡池 + 各等级商店概率、经济（利息/连胜连败）、升级经验表、羁绊、
  装备合成、**海克斯（2-1 / 3-2 / 4-2 三选一）**、**选秀轮（血量低者先选）**、8 人对位扣血
- 战斗：简化的 tick 自动战斗（前排/后排、攻击、护甲、法力值、技能），要求足够快以支持大量对局
- AI：**直接上强化学习**——PPO + 动作掩码（action masking），单一离散动作空间覆盖所有阶段：
  结束回合 / 刷新 / 买经验 / 买商店第 i 张 / 卖备战席或场上单位 / 上下场 / 海克斯三选一 / 选秀 9 选一
- 对手：7 个规则 bot（运营流、D 牌流、随机）起步，之后加自我对弈（历史策略快照池）
- 奖励：终局名次为主（例如 (4.5 - 名次)/3.5），少量血量差 shaping
- 英雄/羁绊数据放在可替换的 JSON 里（先用自编数据，之后可导入真实赛季数据）

## 新需求：用高分对局数据做模仿学习
- 数据源：**Riot 官方 TFT API**（仅覆盖 PC 云顶之弈，金铲铲没有公开 API）
  - `tft-league-v1` 取大师以上玩家 → `tft-match-v1` 取对局详情
  - 能拿到：终局阵容、星级、装备、海克斯、等级、名次；**拿不到逐回合操作**
  - 需要 Riot 开发者 key（放 `.env`，不要提交），遵守速率限制，数据仅本地非商业使用
- 用途：
  1. 行为克隆预训练「终局阵容 / 海克斯选择」策略头，作为 PPO 初始化或 KL 先验
  2. 用高分阵容统计做奖励 shaping（往强势阵容靠拢）
- **不做**：爬取 B 站/抖音视频或第三方统计站（版权与站点条款问题）；视频逐帧识别作为以后可选项
- 本地导入页面：一个本地网页（Flask 或纯静态 + 本地 API），可以
  - 导入 Riot API 抓下来的 JSON / 用户手动整理的对局 JSON
  - 浏览、筛选对局（按名次、阵容、海克斯），查看数据集统计
  - 一键生成训练用的数据集文件

## 建议目录
```
tft_sim/      data.py game.py combat.py augments.py items.py carousel.py
bots/         rule_bots.py
rl/           env.py (Gymnasium) ppo.py train.py evaluate.py
imitation/    riot_fetch.py dataset.py bc_pretrain.py
webui/        app.py templates/ static/
play.py       命令行里自己下场和 bot 打
tests/
```

## 开发顺序
1. 模拟器核心 + 单元测试（卡池守恒、概率、经济、扣血）
2. 规则 bot，跑 1000 局检查平衡与速度
3. Gymnasium 环境 + PPO，先打赢规则 bot
4. Riot 抓取 + 数据集 + 本地导入页面
5. 行为克隆预训练 → 接 PPO 微调；对比有无预训练的名次曲线

## 进度
- [x] 第 1 步：模拟器核心 + 单元测试（`tft_sim/`，`tests/test_core.py`）
- [x] 第 2 步：规则 bot（`bots/rule_bots.py`：运营流 / D 牌流 / 随机）；`python -m bots.arena --games 1000` 跑平衡与速度
  - 1000 局（4 进程，35 秒，约 29 局/秒）：运营流均名 3.43（前四 70%），D 牌流 4.57（49%），随机 6.00（22%）
- [x] 第 3 步：Gymnasium 环境 + PPO（`rl/env.py` `rl/ppo.py` `rl/train.py` `rl/evaluate.py`）
  - 65 个离散动作 + 动作掩码；学习者在座位 0，对手是 7 个规则 bot
  - **与最初设计的差别**：回合结束自动排最佳阵容并装备（上场/下场/装备动作保留但被掩码）；纯 PPO 从零训练一直是第 8 名，
    所以先用脚本"运营流老师"（`rl/teacher.py`）做行为克隆（`rl/bc.py`），再 PPO 微调
  - 评估（`python -m rl.evaluate --baselines`，100 局，3 运营流 / 3 D 牌流 / 1 随机）：
    PPO 均名 1.19（前四 98%，夺冠 90%）；BC 3.09；同座位的运营流 4.12、D 牌流 4.79、随机 5.87
  - 复现：`python -m rl.bc --episodes 500` → `python -m rl.train --steps 2500000 --lr 5e-5 --ent 0.005 --opponents eval --resume checkpoints/bc.pt`
- [x] 第 4 步：Riot 抓取 + 数据集 + 本地导入页面
  - `imitation/riot_fetch.py`：tft-league-v1 → tft-match-v1，密钥从 `RIOT_API_KEY` 或 git 忽略的 `.env` 读取（见 `.env.example`），
    滑动窗口限速（20/秒、100/2分钟）、429 按 Retry-After 重试、断点续抓；原始 JSON 存 `imitation_data/raw/`
  - `imitation/dataset.py`：统一 Riot 对局/手写对局格式，去重存储，筛选，统计，生成 `.npz` + 词表 + 元数据；
    可选 `imitation_data/id_map.json`（Riot 英雄 id → 模拟器英雄名）额外产出 `sim_units` 矩阵
  - `webui/app.py`：`python -m webui.app` → http://127.0.0.1:5000（只监听本机）；导入、筛选、统计、一键生成/下载数据集
  - `imitation/sample.py`：无密钥时用合成对局（ids 以 `SYN_` 开头）试用整个流程
  - 流程：`python -m imitation.riot_fetch ...` → `python -m imitation.dataset ingest` （或网页导入）→ 网页/`dataset build` 生成数据集
- [~] 换成真实赛季数据（进行中，先做通用部分）
  - 现状：**真实数据还没导入**。沙箱网络拦了 `raw.communitydragon.org` 和 `ddragon.leagueoflegends.com`，且还没拿到 `en_us.json`
  - 已完成：引擎数据驱动化
    - `tft_sim/data/` 下 `champions.json traits.json items.json augments.json rules.json`（后三个可选，缺省回退到内置默认）；
      卡池数量、商店概率、经验表、等级上限、利息/连胜、阶段伤害、海克斯轮次都在 `rules.json`
    - 装备支持配方表（`items.json` 的 `completed`），没列配方的组合仍按"属性相加"合成
    - 战斗新增：法抗、暴击（按期望值，保持确定性）、伤害类型（物理/魔法/真实）、技能 `stun` / `buff_as`
    - 羁绊/装备可用属性：`atk atk_pct aspd_pct ap armor mr hp hp_pct mana mana_start crit_chance`
    - RL 环境：板上可寻址单位 9→10，羁绊特征按当前数据生成；旧模型不兼容，已用默认数据重训（`checkpoints/ppo.pt`，100 局评估：均名 1.22、前四 99%、夺冠 88%）
  - `python -m tft_sim.season_import en_us.json --set N --out tft_sim/data/sets/setN`：Community Dragon → 赛季目录
    + `IMPORT_REPORT.md`（哪些译了、哪些没译）。**仅用手写样例文件测试过，没对过真实文件**，字段名可能要调
  - 用法：`GameData(set_dir)`；`python -m bots.arena --set-dir ...`；`rl.bc/train/evaluate --set-dir ...`
  - 导入的英雄 id 用 Riot apiName（如 `TFT13_Jinx`），与对局 API 一致，数据集的 `sim_units` 自动对上，不再需要 id_map
  - 仍需手工/写代码：英雄独有技能、非纯属性的羁绊效果、海克斯（只保留通用默认集）、回合流程（PvE/选秀轮）写死在 `game.py`
