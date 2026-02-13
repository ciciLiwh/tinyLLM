import json
import math
import random
import re
import os
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tokenizers import ByteLevelBPETokenizer
import openai

from model import DecoderOnlyTransformer, select_device


# =========================
# 配置参数（请根据实际情况修改）
# =========================

# 模型和分词器路径（使用你SFT训练好的模型）
VOCAB_PATH = Path("data") / "bbpe" / "vocab.json"
MERGES_PATH = Path("data") / "bbpe" / "merges.txt"
SFT_CKPT = Path("out") / "sft_latest.pt"          # SFT训练好的模型
OUT_DIR = Path("out")

# DeepSeek-V3 API配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")        # 替换为你的API密钥
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1" # 替换为实际API地址
DEEPSEEK_MODEL = "deepseek-chat"                   # 模型名称

# 训练超参数
BATCH_SIZE = 1                                     # 每个step处理的prompt数量（因API调用慢，保持较小值）
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5
KL_COEFF = 0.1
GROUP_SIZE = 4                                      # 每个prompt生成的样本数
MAX_NEW_TOKENS = 128
LOG_INTERVAL = 10
SEED = 42
IGNORE_INDEX = -100

# 可自定义的词牌名列表（请在此填入你想要训练的词牌名）
CI_PAI_LIST = [
    "浣溪沙",
    "蝶恋花",
    "菩萨蛮",
    "水调歌头",
    "念奴娇",
    "满江红",
    "临江仙",
    "鹧鸪天",
    "虞美人",
    "清平乐",
    "卜算子",
    "采桑子",
    "定风波",
    "江城子",
    "渔家傲",
    "青玉案",
    "雨霖铃",
    "永遇乐",
    "贺新郎",
    "摸鱼儿",
    "桂枝香",
    "声声慢",
    "点绛唇",
    "苏幕遮",
    "踏莎行",
    "八声甘州",
    "生查子",
    "玉楼春",
    "忆王孙",
    "浪淘沙"
]

# =========================
# 基于DeepSeek-V3 API的奖励模型
# =========================

class APIBasedReward:
    """调用DeepSeek-V3 API评估生成宋词的质量，返回0~1的奖励分数"""
    
    def __init__(self, max_workers=4, timeout=15):
        self.client = openai.OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_API_BASE
        )
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        # 评估提示模板
        self.prompt_template = (
            "你是一个宋词专家，请严格评估以下生成的宋词是否符合词牌【{ci_pai}】的格式要求（如行数、每行字数等），"
            "并且内容必须全部为有效汉字（无乱码、无英文、无数字、无特殊符号）。如果出现任何非中文字符（包括乱码），则质量评分为0。"
            "如果平仄很好或者内容很好则有加分。如果照抄了古人的作品则给低分。"
            "请给出一个综合质量评分（0到1之间的小数，1为完美，0为完全不符）。"
            "仅返回一个数字，不要包含其他文字。\n\n"
            "生成内容：\n{generated_text}"
        )
    
    def _call_api(self, text: str, ci_pai: str) -> float:
        """同步调用API获取评分"""
        prompt = self.prompt_template.format(ci_pai=ci_pai, generated_text=text)
        try:
            response = self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个专业的宋词评分员，只输出数字。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=10,
                timeout=self.timeout
            )
            content = response.choices[0].message.content.strip()
            # 提取数字（支持0-1的小数）
            match = re.search(r"0?\.\d+|\d+\.?\d*", content)
            if match:
                score = float(match.group())
                return max(0.0, min(1.0, score))  # 限制在 [0,1]
            else:
                print(f"警告: API返回无法解析 -> {content}")
                return 0.0
        except Exception as e:
            print(f"API调用失败: {e}")
            return 0.0
    
    def __call__(self, generated_texts: List[str], ci_pai: str) -> List[float]:
        """批量计算奖励（并发请求提升效率）"""
        futures = [self.executor.submit(self._call_api, text, ci_pai) 
                   for text in generated_texts]
        rewards = [f.result() for f in futures]
        return rewards


# =========================
# 数据集：从词牌名列表构造prompt
# =========================

class GRPODataset(Dataset):
    """使用预定义的词牌名列表生成prompt"""
    
    def __init__(self, ci_pai_list: List[str]) -> None:
        super().__init__()
        self.ci_pai_list = ci_pai_list
        # 为每个词牌名构造prompt
        self.prompts = [
            f"Instruction:\n{ci_pai}\n\nOutput:\n"
            for ci_pai in ci_pai_list
        ]
        print(f"加载了 {len(self.prompts)} 个词牌名: {ci_pai_list}")
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """返回prompt和对应的词牌名"""
        return self.prompts[idx], self.ci_pai_list[idx]


# =========================
# 模型生成
# =========================

@torch.no_grad()
def generate_samples(
    model: DecoderOnlyTransformer,
    tokenizer: ByteLevelBPETokenizer,
    prompt: str,
    num_samples: int = GROUP_SIZE,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> List[str]:
    """为单个prompt生成多个样本"""
    device = next(model.parameters()).device
    eos_id = tokenizer.token_to_id("<|endoftext|>") or 0
    
    # 编码prompt
    input_ids = tokenizer.encode(prompt).ids
    base_input = torch.tensor(input_ids, dtype=torch.long, device=device)
    
    all_generated = []
    
    for _ in range(num_samples):
        current_ids = base_input.clone().unsqueeze(0)
        
        for _ in range(max_new_tokens):
            if current_ids.size(1) >= model.rope.cos.size(0):
                break
            
            logits = model(current_ids)
            next_logits = logits[:, -1, :]
            
            # 应用top-p采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits / temperature, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits = next_logits.masked_fill(indices_to_remove, float('-inf'))
            
            # 采样
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            
            current_ids = torch.cat([current_ids, next_id], dim=1)
            
            if next_id.item() == eos_id:
                break
        
        # 解码生成文本（包括prompt）
        generated_ids = current_ids[0].tolist()
        generated_text = tokenizer.decode(generated_ids)
        all_generated.append(generated_text)
    
    return all_generated


# =========================
# GRPO损失函数（简化版本，基于序列平均对数概率）
# =========================

def compute_grpo_loss(
    logprobs: torch.Tensor,          # [group_size, 1]  每个样本生成部分的平均对数概率
    old_logprobs: torch.Tensor,       # [group_size, 1]
    rewards: torch.Tensor,            # [group_size]
    kl_coeff: float = KL_COEFF,
    advantage_clip: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算GRPO损失
    注意：这里简化处理，将每个序列的平均对数概率作为该序列的代表。
    """
    group_size = logprobs.size(0)
    
    # 计算优势函数（组内归一化奖励）
    rewards_mean = rewards.mean()
    rewards_std = rewards.std() + 1e-8
    advantages = (rewards - rewards_mean) / rewards_std
    
    # 概率比（exp(新logprob - 旧logprob)）
    ratio = torch.exp(logprobs.squeeze() - old_logprobs.squeeze())  # [group_size]
    
    # 策略梯度损失（PPO clip）
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - advantage_clip, 1 + advantage_clip) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # KL散度惩罚
    kl_div = (logprobs.squeeze() - old_logprobs.squeeze()).mean()   # KL(new||old)
    kl_penalty = kl_coeff * kl_div
    total_loss = policy_loss + kl_penalty
    
    metrics = {
        "policy_loss": policy_loss.item(),
        "kl_div": kl_div.item(),
        "avg_reward": rewards.mean().item(),
        "reward_std": rewards_std.item(),
    }
    
    return total_loss, metrics


# =========================
# 主训练循环
# =========================

def main() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)
    
    device = select_device()
    print("使用设备:", device)
    
    # 初始化分词器
    tokenizer = ByteLevelBPETokenizer(
        str(VOCAB_PATH),
        str(MERGES_PATH),
    )
    eos_id = tokenizer.token_to_id("<|endoftext|>") or 0
    pad_id = eos_id
    
    # 加载SFT模型
    print("加载SFT模型...")
    checkpoint = torch.load(SFT_CKPT, map_location=device)
    vocab_size = checkpoint["vocab_size"]
    cfg = checkpoint["config"]
    
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        dim=cfg["dim"],
        num_layers=cfg["num_layers"],
        num_q_heads=cfg["num_q_heads"],
        num_kv_heads=cfg["num_kv_heads"],
        moe_hidden=cfg["moe_hidden"],
        num_experts=cfg["num_experts"],
        max_seq_len=cfg["max_seq_len"],
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state"])
    model.train()
    
    # 创建旧策略模型（用于计算KL散度，固定不变）
    old_model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        dim=cfg["dim"],
        num_layers=cfg["num_layers"],
        num_q_heads=cfg["num_q_heads"],
        num_kv_heads=cfg["num_kv_heads"],          # 原代码错误使用了num_experts，已修正
        moe_hidden=cfg["moe_hidden"],
        num_experts=cfg["num_experts"],
        max_seq_len=cfg["max_seq_len"],
    ).to(device)
    old_model.load_state_dict(model.state_dict())
    old_model.eval()
    
    # 初始化基于API的奖励模型
    reward_model = APIBasedReward(max_workers=4)   # 可调整并发数
    
    # 创建数据集（从词牌名列表）
    print("创建GRPO数据集...")
    dataset = GRPODataset(CI_PAI_LIST)
    
    def collate_fn(batch):
        prompts, ci_pais = zip(*batch)
        return list(prompts), list(ci_pais)
    
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 训练循环
    global_step = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"GRPO Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        
        epoch_rewards = []
        
        for batch_idx, (prompts, ci_pais) in enumerate(train_loader):
            batch_losses = []
            batch_metrics = []
            
            # 对每个prompt单独处理
            for prompt, ci_pai in zip(prompts, ci_pais):
                # 生成多个样本
                generated_texts = generate_samples(
                    model,
                    tokenizer,
                    prompt,
                    num_samples=GROUP_SIZE,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=0.8,
                    top_p=0.9,
                )
                
                # 调用API计算奖励
                rewards = reward_model(generated_texts, ci_pai)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
                epoch_rewards.extend(rewards)
                
                # 计算每个样本生成部分的平均对数概率（当前策略和旧策略）
                all_logprobs = []
                all_old_logprobs = []
                
                for text in generated_texts:
                    # 编码整个文本（包括prompt）
                    input_ids = tokenizer.encode(text).ids
                    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
                    
                    # 计算当前策略的对数概率（生成部分）
                    # 计算当前策略的对数概率（生成部分）——需要梯度
                    logits = model(input_tensor)
                    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
                    token_logprobs = torch.gather(
                        log_probs, 
                        dim=-1, 
                        index=input_tensor[:, 1:].unsqueeze(-1)
                    ).squeeze(-1)  # [1, seq_len-1]
                    
                    # 计算旧策略的对数概率
                    with torch.no_grad():
                        old_logits = old_model(input_tensor)
                        old_log_probs = F.log_softmax(old_logits[:, :-1, :], dim=-1)
                        old_token_logprobs = torch.gather(
                            old_log_probs,
                            dim=-1,
                            index=input_tensor[:, 1:].unsqueeze(-1)
                        ).squeeze(-1)
                    
                    # 只考虑生成部分（不包括prompt）
                    prompt_ids = tokenizer.encode(prompt).ids
                    gen_start = len(prompt_ids) - 1  # 第一个生成token的索引（因为token_logprobs对应input[1:]）
                    
                    if gen_start < token_logprobs.size(1):
                        gen_logprobs = token_logprobs[:, gen_start:]   # [1, gen_len]
                        gen_old_logprobs = old_token_logprobs[:, gen_start:]
                        
                        # 平均对数概率（标量）
                        avg_logprob = gen_logprobs.mean()
                        avg_old_logprob = gen_old_logprobs.mean()
                        
                        all_logprobs.append(avg_logprob)
                        all_old_logprobs.append(avg_old_logprob)
                    else:
                        # 没有生成任何有效token，给予极小值
                        all_logprobs.append(torch.tensor(-10.0, device=device))
                        all_old_logprobs.append(torch.tensor(-10.0, device=device))
                
                # 堆叠对数概率
                if all_logprobs:
                    logprobs_tensor = torch.stack(all_logprobs).unsqueeze(1)      # [group_size, 1]
                    old_logprobs_tensor = torch.stack(all_old_logprobs).unsqueeze(1)
                    
                    # 计算GRPO损失
                    loss, metrics = compute_grpo_loss(
                        logprobs_tensor,
                        old_logprobs_tensor,
                        rewards_tensor,
                        kl_coeff=KL_COEFF,
                    )
                    
                    batch_losses.append(loss)
                    batch_metrics.append(metrics)
            
            # 聚合批次损失并更新模型
            if batch_losses:
                total_loss = torch.stack(batch_losses).mean()
                
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # 定期更新旧策略（每20步同步一次）
                if global_step % 20 == 0:
                    old_model.load_state_dict(model.state_dict())
                    old_model.eval()
                
                # 记录指标
                avg_reward = sum(m["avg_reward"] for m in batch_metrics) / len(batch_metrics)
                avg_policy_loss = sum(m["policy_loss"] for m in batch_metrics) / len(batch_metrics)
                avg_kl_div = sum(m["kl_div"] for m in batch_metrics) / len(batch_metrics)
                
                if (batch_idx + 1) % LOG_INTERVAL == 0:
                    print(f"Step {global_step + 1} | "
                          f"Loss: {total_loss.item():.4f} | "
                          f"Policy Loss: {avg_policy_loss:.4f} | "
                          f"KL Div: {avg_kl_div:.4f} | "
                          f"Avg Reward: {avg_reward:.4f}")
                
                global_step += 1
        
        # Epoch结束统计
        if epoch_rewards:
            avg_epoch_reward = sum(epoch_rewards) / len(epoch_rewards)
            success_rate = sum(1 for r in epoch_rewards if r > 0.5) / len(epoch_rewards)  # 假设>0.5为合格
            print(f"\nEpoch {epoch + 1} 统计:")
            print(f"  平均奖励: {avg_epoch_reward:.4f}")
            print(f"  合格率: {success_rate:.2%}")
            print(f"  总样本数: {len(epoch_rewards)}")
        
        # 保存检查点
        ckpt_path = OUT_DIR / f"grpo_epoch_{epoch + 1}.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "vocab_size": vocab_size,
                "config": cfg,
                "epoch": epoch + 1,
                "global_step": global_step,
                "avg_reward": avg_epoch_reward if epoch_rewards else 0,
            },
            ckpt_path,
        )
        print(f"保存检查点到: {ckpt_path}")
        
        # 保存最新模型
        latest_path = OUT_DIR / "grpo_latest.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "vocab_size": vocab_size,
                "config": cfg,
                "epoch": epoch + 1,
            },
            latest_path,
        )
        
        # 测试生成示例（使用验证集中的几个词牌）
        print("\n测试生成示例:")
        test_ci_pais = CI_PAI_LIST[:3]  # 取前三个
        model.eval()
        for ci_pai in test_ci_pais:
            prompt = f"Instruction:\n生成一首{ci_pai}\n\nOutput:\n"
            
            generated = generate_samples(
                model,
                tokenizer,
                prompt,
                num_samples=1,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                top_p=0.9,
            )[0]
            
            # 提取生成部分
            if "Output:" in generated:
                generated_text = generated.split("Output:")[-1].strip()
            else:
                generated_text = generated.replace(prompt, "").strip()
            
            # 可选：调用API快速评估（但为节省时间，可以跳过）
            # reward = reward_model._call_api(generated_text, ci_pai)
            print(f"\n词牌: {ci_pai}")
            # print(f"奖励: {reward:.2f}")
            print(f"生成结果:\n{generated_text}")
            print("-" * 40)
        
        model.train()

if __name__ == "__main__":
    main()