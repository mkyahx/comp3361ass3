import os

# 1. 核心操作：通过 fuser 命令强行踢掉占用 GPU 显存的残留进程
# /dev/nvidia0 对应你的第一张显卡，如果有两张则加上 /dev/nvidia1
try:
    print("正在强制释放 GPU 显存...")
    os.system("fuser -v /dev/nvidia* -k -9") 
    print("清理成功！")
except Exception as e:
    print(f"清理失败或无需清理: {e}")

# 2. 紧接着设置环境变量
os.environ["VLLM_USE_V1"] = "0"  # 禁用容易产生残留进程的 V1 引擎
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# 3. 后续再进行 import 
import multiprocessing as mp
# ... 你的其他代码

import torch
from datetime import datetime

print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
print(f"CUDA: {torch.cuda.is_available()}")

!pip install -q vllm transformers datasets evaluate

import sys
import io
import multiprocessing as mp
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
try:
    mp.set_start_method("spawn",force=True)
except RuntimeError:
    pass

try:
    sys.stdout.fileno()
except io.UnsupportedOperation:
    sys.stdout.fileno = lambda: sys.__stdout__.fileno()
import vllm

"""Set device and random seeds"""

######################################################
#  The following helper code is given to you.
######################################################

from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')

def set_seed(seed=19260817):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

from datasets import load_dataset

dataset = load_dataset('igormorgado/ROCStories2016')
train_data, dev_data, test_data = dataset['train'], dataset['validation'], dataset['test']

# Construct 'prompt' field from sentence1 for story generation
def add_prompt(example):
    example['prompt'] = example['sentence1']
    return example

train_data = train_data.map(add_prompt)
dev_data = dev_data.map(add_prompt)
test_data = test_data.map(add_prompt)

print(train_data[0])

!pip install "protobuf<4.25" "transformers>=4.40.0" "sentencepiece" -U

"""Prepare evaluation metrics"""

######################################################
#  The following helper code is given to you.
######################################################

from transformers import RobertaForSequenceClassification, RobertaTokenizer

cola_model_name = "textattack/roberta-base-CoLA"
cola_tokenizer = RobertaTokenizer.from_pretrained(cola_model_name)
cola_model = RobertaForSequenceClassification.from_pretrained(cola_model_name).to(device)

def batchify(data, batch_size):
    assert batch_size > 0
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

"""Evaluation functions"""

######################################################
#  The following helper code is given to you.
######################################################

from transformers import GPT2LMHeadModel, GPT2TokenizerFast

_ppl_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
_ppl_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
_ppl_model.eval()

def compute_perplexity(texts, batch_size=4, max_length=512):
    """Compute mean perplexity of a list of texts using GPT-2."""
    nlls = []
    for text in texts:
        if not text.strip():
            continue
        encodings = _ppl_tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length).to(device)
        input_ids = encodings.input_ids
        if input_ids.shape[1] < 2:
            continue
        with torch.no_grad():
            outputs = _ppl_model(input_ids, labels=input_ids)
        nll = outputs.loss.item()
        nlls.append(nll)
    if not nlls:
        return float('inf')
    import math
    return math.exp(sum(nlls) / len(nlls))


def compute_fluency(texts, batch_size=8):
    scores = []
    for b_texts in batchify(texts, batch_size):
        inputs = cola_tokenizer(b_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = cola_model(**inputs).logits
        predictions = torch.argmax(logits, dim=-1)
        scores.extend(predictions.cpu().tolist())
    return sum(scores) / len(scores)


def compute_diversity(texts, n=3):
    all_ngrams = []
    for text in texts:
        tokens = text.split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        all_ngrams.extend(ngrams)
    if len(all_ngrams) == 0:
        return 0
    return len(set(all_ngrams)) / len(all_ngrams)


def compute_repetition(texts, n=4):
    total = 0
    repeated = 0
    for text in texts:
        tokens = text.split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        total += len(ngrams)
        repeated += len(ngrams) - len(set(ngrams))
    if total == 0:
        return 0
    return repeated / total

"""Load model and tokenizer"""

######################################################
#  The following helper code is given to you.
######################################################

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name, pad_token="<|endoftext|>")
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
model.eval()

def decode(prompts, max_len, method, **kwargs):
    encodings_dict = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = encodings_dict['input_ids'].to(device)
    attention_mask = encodings_dict['attention_mask'].to(device)

    batch_size, input_seq_len = input_ids.shape
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
    eos_token_id_tensor = torch.tensor([tokenizer.eos_token_id]).to(device)

    for step in range(max_len):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        next_token_logits = outputs.logits[:, -1, :]
        next_tokens = method(next_token_logits, **kwargs)
        next_tokens = next_tokens * unfinished_sequences + tokenizer.pad_token_id * (1 - unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((batch_size, 1), dtype=torch.long, device=device)], dim=-1
        )
        unfinished_sequences = unfinished_sequences.mul(
            next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
        )
        if unfinished_sequences.max() == 0:
            break

    decoded = tokenizer.batch_decode(input_ids[:, input_seq_len:], skip_special_tokens=True)
    return decoded

"""Debug helper"""

######################################################
#  The following helper code is given to you.
######################################################

dev_prompt = dev_data[0]['prompt']
print(f'Dev prompt: {dev_prompt}')
print()

def show_generations(method_name, method, n=3, max_len=100, **kwargs):
    """Generate n continuations from the dev prompt and display them."""
    for i in range(n):
        output = decode([dev_prompt], max_len, method, **kwargs)
        print(f'[{i+1}] {output[0][:200]}')
    print()

def greedy(next_token_logits):
    '''
    inputs:
    - next_token_logits: Tensor(size = (B, V), dtype = float)
    outputs:
    - next_tokens: Tensor(size = (B,), dtype = long)
    '''
    # TODO: Implement greedy decoding
    return torch.argmax(next_token_logits, dim =-1)
    raise NotImplementedError

show_generations('Greedy', greedy)

def sample(next_token_logits):
    '''
    inputs:
    - next_token_logits: Tensor(size = (B, V), dtype = float)
    outputs:
    - next_tokens: Tensor(size = (B,), dtype = long)
    Hint: use torch.multinomial()
    '''
    # TODO: Implement vanilla sampling
    probs = torch.softmax(next_token_logits, dim=-1)
    return torch.multinomial(probs,num_samples=1).squeeze(-1)
    raise NotImplementedError

show_generations('Vanilla Sampling', sample)

def temperature(next_token_logits, t):
    '''
    inputs:
    - next_token_logits: Tensor(size = (B, V), dtype = float)
    - t: float, temperature parameter (t > 0)
    outputs:
    - next_tokens: Tensor(size = (B,), dtype = long)
    '''
    # TODO: Implement temperature sampling
    return sample(next_token_logits/t)
    raise NotImplementedError

show_generations('Temperature (t=0.8)', temperature, t=0.8)

def topk(next_token_logits, k):
    '''
    inputs:
    - next_token_logits: Tensor(size = (B, V), dtype = float)
    - k: int, number of top tokens to consider
    outputs:
    - next_tokens: Tensor(size = (B,), dtype = long)
    '''
    topkv,topkidx = torch.topk(next_token_logits,k=k)
    mask = torch.full_like(next_token_logits,float("-inf"))
    filtered = mask.scatter_(1,topkidx,topkv)
    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs,num_samples=1).squeeze(-1)
    # TODO: Implement top-k sampling
    raise NotImplementedError

show_generations('Top-k (k=20)', topk, k=20)

def topp(next_token_logits, p):
    '''
    inputs:
    - next_token_logits: Tensor(size = (B, V), dtype = float)
    - p: float, cumulative probability threshold (0 <= p <= 1)
    outputs:
    - next_tokens: Tensor(size = (B,), dtype = long)
    '''
    # TODO: Implement top-p (nucleus) sampling
    sorted, sorted_idx = torch.sort(next_token_logits,descending=True,dim=-1)
    probs = torch.softmax(sorted,dim=-1)
    cum_probs = torch.cumsum(probs,dim=-1)
    idx_remove = cum_probs > p
    idx_remove[...,1:] = idx_remove[...,:-1].clone()
    idx_remove[...,0] = False
    sorted[idx_remove] = float("-inf")
    logit = torch.full_like(next_token_logits, float('-inf'))
    logit.scatter_(dim=-1,index = sorted_idx,src = sorted)

    probs_normalized = torch.softmax(logit,dim=-1)
    return torch.multinomial(probs_normalized, num_samples=1).squeeze(-1)




    raise NotImplementedError

show_generations('Top-p (p=0.7)', topp, p=0.7)

"""Evaluation"""

######################################################
#  The following helper code is given to you.
######################################################

prompts = [item['prompt'] for item in test_data][:5]
GENERATIONS_PER_PROMPT = 5
MAX_LEN = 100

methods = {
    'greedy': {'method': greedy},
    'sample': {'method': sample},
    'temperature_0.8': {'method': temperature, 't': 0.8},
    'topk_20': {'method': topk, 'k': 20},
    'topp_0.7': {'method': topp, 'p': 0.7},
}

results = {}
for name, config in methods.items():
    all_texts = []
    for prompt in tqdm(prompts, desc=name):
        for _ in range(GENERATIONS_PER_PROMPT):
            texts = decode([prompt], MAX_LEN, **config)
            all_texts.extend(texts)
    
    ppl = compute_perplexity(all_texts)
    flu = compute_fluency(all_texts)
    div = compute_diversity(all_texts)
    rep = compute_repetition(all_texts)
    results[name] = {'perplexity': ppl, 'fluency': flu, 'diversity': div, 'repetition': rep}
    print(f'{name}: PPL={ppl:.2f}, Fluency={flu:.4f}, Diversity={div:.4f}, Repetition={rep:.4f}')

# TODO: Run temperature sweep and plot the curves
# For each t in [0.3, 0.5, 0.8, 1.0, 1.5]:
#   - Generate texts using temperature sampling (use 3 prompts, 5 generations each)
#   - Compute perplexity, diversity, repetition
# Plot 3 subplots: (1) t vs perplexity, (2) t vs diversity, (3) t vs repetition
import matplotlib.pyplot as plt


t_values = [0.3, 0.5, 0.8, 1.0, 1.5]
sweep_prompts = [item['prompt'] for item in test_data][:3]
G_PER_PROMPT = 5
MAX_LEN = 100

# 存储结果
ppl_list = []
div_list = []
rep_list = []

for t in t_values:
    all_texts = []
    for prompt in tqdm(sweep_prompts, desc=f"Temperature Sweep T={t}"):
        texts = decode([prompt], MAX_LEN, method=temperature, t=t)
        all_texts.extend(texts)
    
    ppl_list.append(compute_perplexity(all_texts))
    div_list.append(compute_diversity(all_texts))
    rep_list.append(compute_repetition(all_texts))


fig, axes = plt.subplots(1, 3, figsize=(18, 5))


axes[0].plot(t_values, ppl_list, marker='o', color='b')
axes[0].set_title('T vs Perplexity')
axes[0].set_xlabel('Temperature')
axes[0].set_ylabel('PPL')
axes[0].grid(True)


axes[1].plot(t_values, div_list, marker='s', color='g')
axes[1].set_title('T vs Diversity')
axes[1].set_xlabel('Temperature')
axes[1].set_ylabel('Diversity')
axes[1].grid(True)


axes[2].plot(t_values, rep_list, marker='^', color='r')
axes[2].set_title('T vs Repetition')
axes[2].set_xlabel('Temperature')
axes[2].set_ylabel('Repetition Rate')
axes[2].grid(True)

plt.tight_layout()
plt.show()
# raise NotImplementedError

import os
# Use default HF cache (~/.cache/huggingface) - no need to override
# os.environ["HF_HOME"] = "..."

import sys
import io
import multiprocessing as mp
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
try:
    mp.set_start_method("spawn",force=True)
except RuntimeError:
    pass

try:
    sys.stdout.fileno()
except io.UnsupportedOperation:
    sys.stdout.fileno = lambda: sys.__stdout__.fileno()


from vllm import LLM, SamplingParams
model_id = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
llm = LLM(model=model_id, enforce_eager=True,gpu_memory_utilization=0.7)

######################################################
#  The following helper code is given to you.
######################################################

from openai import OpenAI
from transformers import AutoTokenizer

class VLLMClient:
    def __init__(self, model_id, **kwargs):
        self.model_id = model_id

    def __call__(self, prompt: str, **kwargs):
        response = llm.generate(
            prompts=prompt,
            sampling_params=SamplingParams(
                temperature=kwargs.get("temperature", 0.2),
                max_tokens=kwargs.get("max_tokens", 256),
            )
        )
        return response[0].outputs[0].text

model_llm = VLLMClient(model_id)
model_llm("San Francisco is a", max_tokens=42)

######################################################
#  The following helper code is given to you.
######################################################

ARC_EXAMPLARS = [
    {
        "question": "A ball is thrown straight up into the air. At the highest point, the ball's velocity is",
        "choices": ["zero", "at its maximum", "equal to the initial velocity", "negative"],
        "cot_answer": "When a ball is thrown straight up, it decelerates due to gravity. At the highest point, the ball momentarily stops before falling back down. Therefore, the velocity at the highest point is zero. The answer is 0.",
        "short_answer": "0"
    },
    {
        "question": "Which of the following is the best example of a chemical change?",
        "choices": ["crushing a rock", "melting butter", "burning wood", "dissolving sugar in water"],
        "cot_answer": "Crushing a rock, melting butter, and dissolving sugar are physical changes because the substance's chemical composition doesn't change. Burning wood is a chemical change because it produces new substances (ash, carbon dioxide, water vapor). The answer is 2.",
        "short_answer": "2"
    },
    {
        "question": "A student wants to determine how the mass of an object affects the force needed to move it. Which tool should the student use to measure mass?",
        "choices": ["ruler", "balance", "thermometer", "graduated cylinder"],
        "cot_answer": "A ruler measures length, a thermometer measures temperature, and a graduated cylinder measures volume. A balance is used to measure mass by comparing the unknown mass to known masses. The answer is 1.",
        "short_answer": "1"
    },
    {
        "question": "Earth's core is primarily made up of which of the following materials?",
        "choices": ["ite andite", "ite and silica", "iron and nickel", "ite andite"],
        "cot_answer": "Scientists have determined through seismic studies and analysis of meteorites that Earth's core is primarily composed of iron and nickel. These heavy elements sank to the center during Earth's formation. The answer is 2.",
        "short_answer": "2"
    },
    {
        "question": "Which of these is a function of the skeletal system?",
        "choices": ["to transport oxygen", "to protect organs", "to digest food", "to regulate body temperature"],
        "cot_answer": "Transporting oxygen is the circulatory system's function, digesting food is the digestive system's, and regulating body temperature involves the integumentary and circulatory systems. Protecting organs (like the brain by the skull, heart by the ribcage) is a key function of the skeletal system. The answer is 1.",
        "short_answer": "1"
    },
    {
        "question": "What happens to the resistance in a wire as the temperature increases?",
        "choices": ["resistance decreases", "resistance increases", "resistance stays the same", "resistance becomes zero"],
        "cot_answer": "In most conductors (like metals), increasing temperature causes atoms to vibrate more, which increases collisions with electrons flowing through the wire. This increased collision rate means higher resistance. The answer is 1.",
        "short_answer": "1"
    },
    {
        "question": "A plant wilts when it does not receive enough",
        "choices": ["carbon dioxide", "water", "oxygen", "nitrogen"],
        "cot_answer": "Wilting occurs when plant cells lose turgor pressure due to water loss. While plants need carbon dioxide for photosynthesis, oxygen for respiration, and nitrogen for growth, it is the lack of water that directly causes wilting by reducing turgor pressure in cells. The answer is 1.",
        "short_answer": "1"
    },
    {
        "question": "Which of these would be the most effective way to reduce the amount of fossil fuel used for transportation?",
        "choices": ["build wider roads", "use public transportation", "increase parking spaces", "lower speed limits"],
        "cot_answer": "Building wider roads and increasing parking spaces would likely encourage more driving, thus more fossil fuel use. Lowering speed limits has a minor effect. Public transportation is the most effective because one bus or train replaces many individual cars, significantly reducing per-person fossil fuel consumption. The answer is 1.",
        "short_answer": "1"
    }
]

######################################################
#  The following helper code is given to you.
######################################################

!mkdir -p data
# Download ARC-Challenge test set (first 50 examples)
from datasets import load_dataset
import json

arc_dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
arc_data = []
for item in list(arc_dataset)[:50]:
    choices = item['choices']['text']
    labels = item['choices']['label']
    answer_key = item['answerKey']
    # Convert letter answer to index
    answer_idx = labels.index(answer_key) if answer_key in labels else ord(answer_key) - ord('A')
    arc_data.append({
        "question": item['question'],
        "choices": choices,
        "true_answer": str(answer_idx),
        "source": "ARC"
    })

with open("data/arc.jsonl", "w") as f:
    for item in arc_data:
        f.write(json.dumps(item) + "\n")

print(f"Loaded {len(arc_data)} ARC-Challenge questions")
print(json.dumps(arc_data[0], indent=2))

######################################################
#  The following helper code is given to you.
######################################################

import json

def load_eval_data(task):
    with open(f"data/{task}.jsonl", "r") as f:
        return [json.loads(line) for line in f]

print("Example ARC question:")
print(json.dumps(load_eval_data("arc")[0], indent=4))

######################################################
#  The following helper code is given to you.
######################################################

import os
import json
import datetime
from pathlib import Path
from tqdm import tqdm

def answer_questions(task, agent, action_type, answers_file):
    """Run agent on all questions in a task and save answers."""
    data = load_eval_data(task)
    Path("output").mkdir(exist_ok=True)
    
    existing = set()
    if os.path.exists(answers_file):
        with open(answers_file, "r") as f:
            for line in f:
                item = json.loads(line)
                existing.add(item["question"])
    
    for item in tqdm(data, desc=f"Running {action_type} on {task}"):
        q = item["question"]
        if isinstance(item.get("choices"), list):
            choices_str = "\n".join(f"  {i}. {c}" for i, c in enumerate(item["choices"]))
            q = f"{q}\nChoices:\n{choices_str}\nAnswer with only the choice number (0, 1, 2, ...)."
        if q in existing:
            continue
        try:
            answer = agent.run(q)
        except Exception as e:
            answer = f"ERROR: {e}"
        
        result = {
            "question": item["question"],
            "answer": answer,
            "true_answer": item["true_answer"],
            "source": item["source"],
            "model_id": model_llm.model_id,
            "agent_action_type": action_type,
            "timestamp": str(datetime.datetime.now())
        }
        with open(answers_file, "a") as f:
            f.write(json.dumps(result) + "\n")

%cat data/arc.jsonl

!rm -rf output/*.jsonl

from eval_utils import score_answers

class FewShotReasoner:
    def __init__(self, model, n_shots):
        self.model = model
        self.n_shots = n_shots

    def build_input(self, question):
        """Build an n-shot direct prompt using ARC_EXAMPLARS."""
        # TODO: Implement this
        import json
        
        # 1. 定义 Prompt 头部
        prompt = "Answer the following questions. Choose the correct option by its number.\n\n"
        
        # 2. 加载示例数据 (data/arc.jsonl)
        exemplars = []
        try:
            with open("data/arc.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    ex = json.loads(line)
                    # 检查是否为当前题目（防止将题目本身作为示例）
                    # 这里的 question 是字符串，我们检查它是否包含示例中的问题文本
                    if ex['question'] in question:
                        continue
                    exemplars.append(ex)
                    if len(exemplars) >= self.n_shots:
                        break
        except Exception as e:
            # 如果读取失败，为了保证程序不崩，至少返回原始问题
            return f"Question: {question}\nAnswer:"

        # 3. 拼接 Few-shot 示例
        for ex in exemplars:
            prompt += f"Question: {ex['question']}\nChoices:\n"
            for idx, choice in enumerate(ex['choices']):
                prompt += f"  {idx}. {choice}\n"
            prompt += f"Answer: {ex['true_answer']}\n\n"
        
        # 4. 拼接当前需要推理的题目
        # 由于传入的 question 已经包含了 "Choices:..." 和 "Answer with only..."
        # 我们需要将其格式标准化，去掉末尾的提示语，换成要求的 "Answer:"
        
        # 清理传入的 question 字符串，去掉尾部的提示语
        clean_question = question.split("Answer with only")[0].strip()
        
        prompt += f"Question: {clean_question}\nAnswer:"
        
        return prompt
        
        raise NotImplementedError

    def run(self, question):
        # print(question)
        prompt = self.build_input(question)
        
        response = self.model(prompt=prompt, max_tokens=32, temperature=0.0)
        
        # vLLM 返回的通常是一个列表或包含 text 的对象，取决于你的 model_llm 封装
        # 假设 response 是直接的字符串，如果不是，请用 response[0].outputs[0].text
        model_answer = response.strip() 
        
        # 2. 打印对比
        print("-" * 30)
        print(f"题目: {question['question'][:50]}...") # 只打印前50个字符
        print(f"模型预测输出: '{model_answer}'")
        print(f"标准答案索引: '{question['true_answer']}'")



        return self.model(prompt=prompt, max_tokens=32, temperature=0.0)


def run_arc_fewshot(task="arc", action_type="fewshot"):
    answers_file = f"output/{model_id.replace('/', '__')}__{action_type}__{task}.jsonl"
    reasoner = FewShotReasoner(model_llm, 8)
    answer_questions(task, reasoner, action_type, answers_file)
    df = score_answers([answers_file])
    print(df)

run_arc_fewshot()

import re

class FewShotCoTReasoner:
    def __init__(self, model, n_shots):
        self.model = model
        self.n_shots = n_shots

    def build_input(self, question):
        """Build an n-shot chain-of-thought prompt using ARC_EXAMPLARS."""
        # TODO: Implement this
        prompt = "Answer the following questions. Provide your reasoning followed by the final answer in the format 'The answer is X'.\n\n"
        
        # 1. 遍历前 n_shots 个示例
        for i in range(self.n_shots):
            ex = ARC_EXAMPLARS[i]
            prompt += f"Question: {ex['question']}\nChoices:\n"
            for idx, choice in enumerate(ex['choices']):
                prompt += f"  {idx}. {choice}\n"
            
            # 核心：将 cot_answer（含推理逻辑）放入 Prompt
            prompt += f"Answer: {ex['cot_answer']}\n\n"
        
        # 2. 拼接当前待预测的问题
        prompt += f"Question: {question['question']}\nChoices:\n"
        for idx, choice in enumerate(question['choices']):
            prompt += f"  {idx}. {choice}\n"
        
        # 强制诱导模型开始推理
        prompt += "Answer: Let's think step by step."
        return prompt
        raise NotImplementedError

    def extract_answer(self, response):
        """Extract the final numeric answer from CoT response."""
        # TODO: Extract the answer number from the response
        # Hint: look for patterns like 'The answer is X' or the last number
        match = re.search(r"The answer is\s*(\d+)", response)
        if match:
            return match.group(1)
        
        # 策略 B：寻找 'Answer: X' 模式
        match_alt = re.search(r"[Aa]nswer:\s*(\d+)", response)
        if match_alt:
            return match_alt.group(1)
        
        # 策略 C：如果都没有，取文本中出现的最后一个数字
        numbers = re.findall(r"\d+", response)
        if numbers:
            return numbers[-1]
        
        return "None"
        raise NotImplementedError

    def run(self, question):
        prompt = self.build_input(question)
        response = self.model(prompt=prompt, max_tokens=512, temperature=0.0)
        return self.extract_answer(response)


def run_arc_fewshot_cot(task="arc", action_type="fewshot_cot"):
    answers_file = f"output/{model_id.replace('/', '__')}__{action_type}__{task}.jsonl"
    reasoner = FewShotCoTReasoner(model_llm, 8)
    answer_questions(task, reasoner, action_type, answers_file)
    df = score_answers([answers_file])
    print(df)

run_arc_fewshot_cot()

!pip install -q wikipedia

import wikipedia

class WikiSearchTool:
    """Search Wikipedia and return a summary."""
    name = "wiki_search"
    description = "Search Wikipedia for a topic and return a summary. Input should be a search query string."

    def __call__(self, query: str) -> str:
        """Search Wikipedia and return the summary of the top result.
        Handle disambiguation and page not found errors gracefully."""
        # TODO: Implement this
        # Hint: use wikipedia.search() to find pages, then wikipedia.summary() or wikipedia.page()
        # Handle wikipedia.exceptions.DisambiguationError and wikipedia.exceptions.PageError
        raise NotImplementedError


# Test
wiki_tool = WikiSearchTool()
print(wiki_tool("Eiffel Tower")[:500])

class CalculatorTool:
    """Evaluate a mathematical expression."""
    name = "calculator"
    description = "Evaluate a mathematical expression. Input should be a valid Python math expression as a string."

    def __call__(self, expression: str) -> str:
        """Safely evaluate a math expression.
        Only allow basic math operations. Do NOT use eval() directly on untrusted input."""
        # TODO: Implement this
        # Hint: you can use Python's ast module or a restricted eval
        # Only allow: numbers, +, -, *, /, **, (, ), math functions
        # Return the result as a string
        raise NotImplementedError


# Test
calc_tool = CalculatorTool()
print(calc_tool("67905000 * 3"))
print(calc_tool("(8848.86 * 3.28084)"))
print(calc_tool("import os"))  # Should return an error message, not execute

class ReActAgent:
    def __init__(self, model, tools, max_steps=8):
        """
        Args:
            model: VLLMClient instance
            tools: list of tool instances (WikiSearchTool, CalculatorTool)
            max_steps: maximum number of Thought-Action-Observation cycles
        """
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.tools["finish"] = None  # special action to end
        self.max_steps = max_steps

    def build_system_prompt(self):
        """Build the system prompt describing tools and the ReAct format."""
        # TODO: Implement this
        # Should describe:
        # 1. Available tools (name + description for each)
        # 2. The ReAct format (Thought/Action/Action Input/Observation)
        # 3. The 'finish' action for returning the final answer
        raise NotImplementedError

    def parse_action(self, text):
        """Parse the LLM output to extract Action and Action Input.
        
        Returns:
            tuple: (action_name: str, action_input: str) or (None, None) if parsing fails
        """
        # TODO: Implement this
        # Parse lines like:
        #   Action: wiki_search
        #   Action Input: Eiffel Tower
        raise NotImplementedError

    def run(self, question):
        """Run the ReAct loop.
        
        Args:
            question: The user's question
            
        Returns:
            The final answer string, or None if max_steps reached
        """
        # TODO: Implement the ReAct loop
        # 1. Build the initial prompt with system prompt + question
        # 2. Loop up to max_steps:
        #    a. Call the model to get the next Thought + Action
        #    b. Parse the action
        #    c. If action is 'finish', return the action_input as final answer
        #    d. Otherwise, execute the tool and append Observation
        #    e. Continue with the extended prompt
        raise NotImplementedError

# Test the agent on a few examples
agent = ReActAgent(
    model=model_llm,
    tools=[WikiSearchTool(), CalculatorTool()],
    max_steps=8
)

# Simple test
test_questions = [
    "What is the population of France multiplied by 3?",
    "Who wrote the novel '1984', and in what year did that author die?",
    "What is the factorial of 7?",
]

for q in test_questions:
    print(f"Q: {q}")
    answer = agent.run(q)
    print(f"A: {answer}")
    print("-" * 50)

######################################################
#  The following helper code is given to you.
######################################################

class ChatAgent:
    """Simple agent that directly asks the LLM without any tools."""
    def __init__(self, model):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model.model_id)

    def run(self, task):
        messages = [{"role": "user", "content": task + "\nAnswer concisely."}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.model(prompt=prompt, max_tokens=256)

!mkdir -p data
import urllib.request

BASE_URL = "https://raw.githubusercontent.com/ranpox/comp3361-spring2025/refs/heads/main/assignments/A3/data"
for dataset_name in ["math", "gaia"]:
    urllib.request.urlretrieve(f"{BASE_URL}/{dataset_name}.jsonl", f"data/{dataset_name}.jsonl")
    print(f"Downloaded {dataset_name}.jsonl")

# Run agents and save answers to files
all_answer_files = []

for task in ["math", "gaia"]:
    # Vanilla ChatAgent
    vanilla_file = f"output/{model_id.replace('/', '__')}__vanilla__{task}.jsonl"
    chat_agent = ChatAgent(model_llm)
    answer_questions(task, chat_agent, "vanilla", vanilla_file)
    all_answer_files.append(vanilla_file)

    # ReAct Agent
    react_file = f"output/{model_id.replace('/', '__')}__react__{task}.jsonl"
    react_agent = ReActAgent(
        model=model_llm,
        tools=[WikiSearchTool(), CalculatorTool()],
        max_steps=8
    )
    answer_questions(task, react_agent, "react", react_file)
    all_answer_files.append(react_file)

print("All agents finished. Answer files:", all_answer_files)

# Score answers and display results
from eval_utils import score_answers

df = score_answers(all_answer_files)
print(df.to_string())
