import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import ray
import torch
import torch.nn as nn
from tqdm import tqdm

from openrlhf.models.actor import Actor
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean, unpacking_samples
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray

logger = init_logger(__name__)

import requests, json, os
API_try_counter=3
API_url = ""
split_url=API_url+"/split_query"

retrieve_url = "http://your_retriever_host:your_retriever_port"+"/search"

from openai import OpenAI
model_name = "your_check_model_name"
# 首先尝试获取分布式训练的rank
if torch.distributed.is_initialized():
    rank = torch.distributed.get_rank()
else:
    # 如果没有使用分布式训练，则使用进程ID
    pid = os.getpid()
    rank = pid
# print(f"Rank: {rank}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# 根据rank的奇偶性选择客户端
if rank % 2 == 0:  # 偶数进程
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://your_check_model_host:your_check_model_port/v1",
    )
    print(f"进程 {rank} (PID: {os.getpid()}) 使用client_1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
else:  # 奇数进程
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://your_check_model_host:your_check_model_port/v1",
    )
    print(f"进程 {rank} (PID: {os.getpid()}) 使用client_2!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
try_counter=3

num_search_one_attempt=5 # 最大只到9
num_passages_one_retrieval = 2
num_passages_one_split_retrieval = 0
flag_split=False
max_num_passages = 10
try_format_wrong=3
uncorrect_penalty_factor=0.2
correct_bonus_factor=0.6
format_wrong_factor=0.1
fail_penalty_factor=0.1
temperature_factor=0.3
def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor

@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        self.values = to(self.values, device)
        self.attention_mask = to(self.attention_mask, device)
        self.action_mask = to(self.action_mask, device)
        self.kl = to(self.kl, device)
        self.info = {key: to(value, device) for key, value in self.info.items()}
        return self

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        self.values = pin_memory(self.values)
        self.attention_mask = pin_memory(self.attention_mask)
        self.action_mask = pin_memory(self.action_mask)
        self.kl = pin_memory(self.kl)
        self.info = {key: pin_memory(value) for key, value in self.info.items()}
        return self

@dataclass
class Samples:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor

class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: str = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.perf_stats = None
        self.advantage_estimator = strategy.args.advantage_estimator

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str]], **generate_kwargs) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        args = self.strategy.args
        # generate responses
        samples_list, myrewards_list = self.generate_step_samples(all_prompts, **generate_kwargs)
        torch.distributed.barrier()

        experiences = []
        for samples, myrewards in tqdm(
            # samples_list,
            zip(samples_list, myrewards_list),  # Combine samples and rewards
            desc="make_experience",
            disable=not self.strategy.is_rank_0(),
        ):
            experiences.append(self.make_experience(samples, myrewards).to_device("cpu"))

        experiences, rewards = self.process_experiences(experiences)

        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            experience = experience.to_device("cuda")
            reward = reward.to(device="cuda")
            num_actions = experience.info["num_actions"]
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            elif self.advantage_estimator in ["reinforce", "rloo"]:
                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
            experience.to_device("cpu")
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.
        """
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()
        # sample multiple response
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        samples_list = []
        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
            prompts = all_prompts[i : i + args.micro_rollout_batch_size]
            inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
            samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
            )
            samples_list.append(samples)
        return samples_list

    @torch.no_grad()
    def generate_step_samples(self, all_prompts: List[str], **generate_kwargs) -> List[Samples]:
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()

        samples_list = []
        rewards_list = []

        assert args.micro_rollout_batch_size==1

        for i_samples_per_prompt in range(args.n_samples_per_prompt):
            temperature = 0.1 + i_samples_per_prompt * temperature_factor
            generate_kwargs.update({'temperature': temperature})
            pronlem_counter = len(all_prompts["prompt"])
            correct_counter = 0
            wrong_counter = 0
            fail_counter = 0
            for i in range(0, pronlem_counter):
                state = "undo"
                sample_list = []
                reward_list = []
                search_chain=[]
                answer=''
                correctness=False
                correctness_reason=''
                wrong_reason=''
                retrieved_ids = []
                documents = []
                flag=True
                prompts = all_prompts["prompt"][i]
                # print(f"chat template:")
                # print(repr(prompts))
                # 写入文件
                # with open('/remote-home1/yli/Workspace/R3RAG/train/OpenRLHF/output.txt', 'a', encoding='utf-8') as f:
                #     f.write(prompts)
                #     f.write('\n')
                golden_answers = all_prompts["golden_answers"][0][i]
                SystemInput=prompts
                for search_counter in range(num_search_one_attempt):
                    SystemInput_with_chat_template = ApplyChatTemplate(SystemInput)
                    inputs = self.tokenize_fn(SystemInput_with_chat_template, self.prompt_max_len, device="cuda")
                    try_format_wrong_counter=0
                    # 审核格式1:
                    while True:
                        try_format_wrong_counter+=1
                        sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
                        # assert len(attention_mask[0]) == len(action_mask[0])
                        InputandOutputText = self.tokenizer.batch_decode(sequences[0].unsqueeze(0), skip_special_tokens=True)
                        InputandOutputText = InputandOutputText[0]
                        # print(f'SystemInput_with_chat_template: \n {SystemInput_with_chat_template}')
                        # print(f'InputandOutputText: \n {InputandOutputText}')                        
                        check_num = InputandOutputText.count(f"Step {search_counter+1}:\n")
                        if check_num==1:
                            break
                        else:
                            if try_format_wrong_counter==try_format_wrong:
                                wrong_reason = f"Error: The format is wrong after Regenerating {try_format_wrong} times. check_num: {check_num}"+f"Step {search_counter+1}:\n"
                                flag=False
                                break
                    # 审核格式2:
                    if flag:
                        start_index = InputandOutputText.find(f"Step {search_counter+1}:\n")
                        if start_index!=-1:
                            this_step = InputandOutputText[(start_index):]
                            mydict = split_response(this_step)
                        else:
                            flag=False   
                            wrong_reason=f""""Step {search_counter+1}:\n" can't be found."""
                    # 审核格式3:
                    if flag:
                        if not mydict.get('analysis', ""):
                            flag=False
                            # search_chain.append(mydict)
                            wrong_reason="Error: No analysis"
                            print(wrong_reason)
                        elif (not mydict.get('query', "")) and (not mydict.get('answer', "")):
                            flag=False
                            # search_chain.append(mydict)
                            wrong_reason="Error: No query or answer"
                            print(wrong_reason)
                    # 审核格式4+修改attention_mask和action_mask
                    if flag:
                        # print(f"len(sequences[0]):{len(sequences[0])} vs len(attention_mask[0]):{len(attention_mask[0])} vs len(inputs['input_ids'][0]){len(inputs['input_ids'][0])} + len(action_mask[0]){len(action_mask[0])}:{len(inputs['input_ids'][0]) + len(action_mask[0])}")
                        assert len(sequences[0]) == len(attention_mask[0]) == len(inputs['input_ids'][0]) + len(action_mask[0])
                        # "The retrieval query:"注意不包含冒号后面的空格
                        # 785, 56370, 3239, 25,
                        # 数字0到9对应的token的list
                        token_id_number = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
                        # "Step {number}:\n"
                        # \n: 198
                        # "Step": 8304
                        # " ":220
                        # " {number}": token_id_number
                        # ":": 25
                        # ":\n": 510

                        mask_chain = []

                        token_list = sequences[0].cpu().numpy().tolist()

                        # real_start_token_id_list= [785, 56370, 3239, 25] # "The retrieval query:"
                        real_start_token_id_list = [785, 56370, 9293, 25] # "The retrieval documents:" 冒号后面的空格会跟不同的后面的符号组合，而不会形成一个独立的token
                        step_num=search_counter+1
                        step_num_tmp = 1
                        start_index=0
                        real_start_index=0
                        end_index=0
                        while False:
                            if step_num==1:
                                break
                            # find start index: step 前面得\n经常会出现不同的tokenize方式，所以去除\n的匹配
                            if step_num_tmp==1:
                                start_step_token_id_list = [8304, 220]
                            else:
                                # start_step_token_id_list = [198, 8304, 220]
                                start_step_token_id_list = [8304, 220]
                            start_step_token_id_list.append(token_id_number[step_num_tmp])
                            start_step_token_id_list.append(510)
                            start_index = find_sublist_first_occurrence(token_list, start_step_token_id_list, start_index)
                            if start_index==-1:
                                wrong_reason = f"Error: Can't find the start of step {step_num_tmp}."
                                mask_one = {
                                    "find_id_list": start_step_token_id_list,
                                    "text": InputandOutputText,
                                    "token_list": token_list,
                                    "step_num": step_num_tmp,
                                    "start_index": start_index,
                                    "real_start_index": real_start_index,
                                    "end_index": end_index,
                                    "wrong_reason": wrong_reason,
                                }
                                mask_chain.append(mask_one)                            
                                flag=False
                                break
                            # find real_start index
                            real_start_index = find_sublist_first_occurrence(token_list, real_start_token_id_list, start_index)
                            if real_start_index==-1:
                                wrong_reason = f"Error: Can't find the real start of step {step_num_tmp}."
                                mask_one = {
                                    "find_id_list": real_start_token_id_list,
                                    "text": InputandOutputText,
                                    "token_list": token_list,
                                    "step_num": step_num_tmp,
                                    "start_index": real_start_index,
                                    "real_start_index": real_start_index,
                                    "end_index": end_index,
                                    "wrong_reason": wrong_reason,
                                }
                                mask_chain.append(mask_one)
                                flag=False
                                break
                            # find end index 
                            if step_num_tmp==step_num:
                                end_step_token_id_list = []
                                end_index = len(token_list)
                            else:
                                # end_step_token_id_list = [198, 8304, 220]
                                end_step_token_id_list = [8304, 220]
                                end_step_token_id_list.append(token_id_number[step_num_tmp+1])
                                end_step_token_id_list.append(510)
                                end_index = find_sublist_first_occurrence(token_list, end_step_token_id_list, start_index)
                            if end_index==-1:
                                wrong_reason = f"Error: Can't find the end of step {step_num_tmp}."
                                mask_one = {
                                    "find_id_list": end_step_token_id_list,
                                    "text": InputandOutputText,
                                    "token_list": token_list,
                                    "step_num": step_num_tmp,
                                    "start_index": real_start_index,
                                    "real_start_index": real_start_index,
                                    "end_index": end_index,
                                    "wrong_reason": wrong_reason,
                                }
                                mask_chain.append(mask_one)
                                flag=False
                                break
                            # 更改attention_mask
                            attention_mask[0][real_start_index:(end_index)]=0
                            # 以下注释代码用于调试：把sequence中对应的token_id转换为文本输出：
                            # modified_token_list = token_list[real_start_index:(end_index)] # 不能加1，因为是左闭右开，end index刚好是下一个step的开始索引。
                            # text = self.tokenizer.decode(modified_token_list, skip_special_tokens=True)
                            # print(f"Step {step_num_tmp} modified:")
                            # print(repr(text))
                            # 更改action_mask：由于action是输出部分的，输出部分没有涉及到检索内容，所以不需要mask
                            # action_mask[0][real_start_index:end_index]=0
                            mask_one = {
                                "text": InputandOutputText,
                                "token_list": token_list,
                                "step_num": step_num_tmp,
                                "start_index": real_start_index,
                                "real_start_index": real_start_index,
                                "end_index": end_index,
                            }
                            mask_chain.append(mask_one)

                            if step_num_tmp==(step_num-1): # 做完了倒数第二步就可以退出
                                # 最后一步不用做，如果是query，还没检索，所以没有需要mask的内容，如果是answer，无需检索，所以也不需要mask
                                break
                            else:
                                # 更新step_num_tmp
                                step_num_tmp+=1
                        # with open('/remote-home1/yli/Workspace/R3RAG/train/OpenRLHF/mask_chain.json', 'a', encoding='utf-8') as f:
                        #     json.dump(mask_chain, f, ensure_ascii=False, indent=4)
                        #     f.write('\n')
                    # 格式审核结束
                    if not flag:
                        mydict["input"] = SystemInput
                        mydict["input_with_chat_template"] = SystemInput_with_chat_template
                        mydict["output"] = InputandOutputText
                        search_chain.append(mydict)
                        # 有格式问题：存储数据，更新状态
                        samples = Samples(
                            sequences=sequences,
                            attention_mask=attention_mask,
                            action_mask=action_mask,
                            num_actions=action_mask.size(1),
                            packed_seq_lens=None,
                            response_length=action_mask.float().sum(dim=-1),
                            total_length=attention_mask.float().sum(dim=-1),
                        )
                        sample_list.append(samples)
                        reward = -1 * (1-format_wrong_factor)
                        reward_list.append(reward)
                        break
                    
                    # 无格式问题：存储数据，更新状态
                    samples = Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                    )
                    sample_list.append(samples)
                    mydict['input'] = SystemInput
                    mydict['input_with_chat_template'] = SystemInput_with_chat_template
                    mydict['output'] = InputandOutputText
                    mydict['step_output'] = this_step
                    mydict['sample'] = str(samples)
                    token_ids = sequences[0].tolist()
                    mydict['token_texts'] = self.tokenizer.convert_ids_to_tokens(token_ids)
                    if mydict.get('answer', ""):
                        answer = mydict['answer']
                        correctness, correctness_reason = check_correctness(prompts, answer, [golden_answers])
                        mydict["correctness"]=correctness
                        mydict["correctness_reason"]=correctness_reason
                        search_chain.append(mydict)
                        state = "done"
                        break
                    elif mydict.get('query', ""):
                        if flag_split:
                            GetStepbystepRetrievalv2(mydict, retrieved_ids, documents)
                        else:
                            GetStepbystepRetrieval(mydict, retrieved_ids, documents)
                        # 计算reward
                        # 从search_chain的每一个元素的step_output组合出来的字符串，然后计算reward
                        history_steps = [mydict['step_output'] for mydict in search_chain]
                        # reward = calculate_step_score(prompts, history_steps, mydict['analysis'], mydict['query'], mydict['doc'][0:512])
                        reward = calculate_doc_score(prompts, history_steps, mydict['analysis'], mydict['query'], mydict['doc'][0:512])
                        reward_list.append(reward)
                        # 更新SystemInput
                        step = f"Step {search_counter+1}:\nThe problem analysis: {mydict['analysis']}\nThe retrieval query: {mydict['query']}"+"\n"+f"The retrieval documents: {mydict['doc'][0:512]}"
                        SystemInput = SystemInput+"\n"+step
                        # SystemInput = SystemInput+'\n'+this_step+"\n"+f"The retrieval documents: {mydict['doc'][0:512]}"
                        search_chain.append(mydict)


                        if search_counter==num_search_one_attempt-1:
                            print("Warning: The last step doesn't generate answer.")
                            state = "fail"
                            break
                if not flag:
                    state = "wrong"
                if state=="undo":
                    sample_list = []
                    reward_list = []
                elif state=="fail":
                    fail_counter+=1
                    assert len(sample_list)==len(search_chain)==len(reward_list)
                    reward_list = [round(reward_one_tmp*(1-fail_penalty_factor), 3) for reward_one_tmp in reward_list]
                    samples_list.extend(sample_list)
                    rewards_list.extend(reward_list)
                elif state=="wrong":
                    wrong_counter+=1
                    assert len(sample_list)==len(reward_list)==len(search_chain)
                    reward_list = [round(reward_one_tmp, 3) for reward_one_tmp in reward_list]
                    samples_list.extend(sample_list)
                    rewards_list.extend(reward_list)
                elif state=="done":                 
                    samples_list.extend(sample_list)
                    if correctness:
                        correct_counter+=1
                        reward_list = [reward*(1+correct_bonus_factor) for reward in reward_list]
                        reward_list.append(2.0)
                    elif not correctness:
                        reward_list = [reward*(1-uncorrect_penalty_factor) for reward in reward_list]
                        reward_list.append(0.0)
                    # else:
                    #     reward_list = [0 for _ in range(len(search_chain))]
                    # 给每一个元素乘以一个-1的系数
                    # reward_list = [-1*reward for reward in reward_list]
                    assert len(sample_list)==len(reward_list)==len(search_chain)
                    reward_list = [round(reward_one_tmp, 3) for reward_one_tmp in reward_list]
                    rewards_list.extend(reward_list)
                else:
                    raise Exception("Error: Wrong state.")

                print(f"i_samples_per_prompt: {i_samples_per_prompt}, state: {state}, correctness: {correctness}, wrong_reason: {wrong_reason}")
                print(f"reward_list: {reward_list}")
                # data_record = {
                #     "input": prompts,
                #     "golden_answers": golden_answers,
                #     "answer": answer,
                #     "search_chain":search_chain,
                #     "state":state,
                #     "wrong_reason":wrong_reason,
                #     "correctness":correctness,
                #     "reward_list":reward_list,
                #     "correctness_reason":correctness_reason,
                #     "doc_used": [len(retrieved_ids),len(documents)],
                #     "doc_byte": len('\n'.join(documents)),
                # }
                # with open('/remote-home1/yli/Workspace/R3RAG/train/OpenRLHF/record.json', 'a', encoding='utf-8') as f:
                #     json.dump(data_record, f, ensure_ascii=False, indent=4)
                #     f.write('\n')
            print(f"i_samples_per_prompt: {i_samples_per_prompt}, wrong_counter: {wrong_counter}, fail_counter: {fail_counter}, correct_counter: {correct_counter}, pronlem_counter: {pronlem_counter}")
        samples_list_length=len(samples_list)
        rewards_list_length=len(rewards_list)
        question_length = len(all_prompts["prompt"])
        # print(f"Samples_list_length: {samples_list_length}")
        # print(f"Rewards_list_length: {rewards_list_length}")
        assert samples_list_length==rewards_list_length
        # print(f"Question_length: {question_length}")

        total= num_search_one_attempt * question_length * args.n_samples_per_prompt
        remains = total - samples_list_length
        # print(f"Remains: {remains}")
        for i in range(remains):
            samples_list.append(samples_list[i])
            rewards_list.append(rewards_list[i])
        # samples_list_length=len(samples_list)
        # rewards_list_length=len(rewards_list)
        # print(f"Samples_list_length: {samples_list_length}")
        # print(f"Rewards_list_length: {rewards_list_length}")
        print(f"Samples_list_length: {len(samples_list)}")        
        return samples_list, rewards_list

    @torch.no_grad()
    def make_experience(self, samples: Samples, rewards) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        self.actor.eval()
        self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()
        if self.critic is not None:
            self.critic.eval()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask)

        # init log probs
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)

        # values
        if self.critic is not None:
            value = self.critic(sequences, num_actions, attention_mask)
        else:
            value = None

        # rewards
        r = torch.tensor([rewards], dtype=torch.bfloat16, device=action_log_probs.device)
        # if self.remote_rm_url is not None:
            # remote RM
            # queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            # r = remote_rm_fn(self.remote_rm_url, queries=queries).to(device=action_log_probs.device)
        # else:
            # local RM
            # r = self.reward_model(sequences, attention_mask)

        kl = compute_approx_kl(
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
            use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
        )
        # if rewards==1:
        #     print(f"local RM:\n{r}")
        # else:
        #     print(f"local RM(Negative):\n{r}")
        # print(f"local RM:\n{myr}")
        # print(f"KL:\n{kl}")

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }
        # reset model state
        self.actor.train()
        if self.critic is not None:
            self.critic.train()

        experience_one = Experience(
            sequences,
            action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )
        # with open('/remote-home1/yli/Workspace/R3RAG/train/OpenRLHF/output.txt', 'a', encoding='utf-8') as f:
        #     f.write(str(experience_one))
        #     f.write('\n')
        return experience_one
        # return Experience(
        #     sequences,
        #     action_log_probs,
        #     value,
        #     None,
        #     None,
        #     attention_mask,
        #     action_mask,
        #     info,
        #     kl,
        # )

    @torch.no_grad()
    def process_experiences(self, experiences: List[Experience]) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args
        # reward shaping for RLOO
        if args.advantage_estimator == "rloo":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
            rewards = rewards.flatten().to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        # default rewards
        return experiences, [experience.info["reward"] for experience in experiences]

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """

        if isinstance(rewards, list):
            # packing samples
            # TODO: this is slow...
            returns = []
            for r in rewards:
                ret = self.get_cumulative_returns(r.unsqueeze(0), action_mask, gamma)
                returns.append(ret.squeeze(0))
            return returns

        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns

class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, packing_samples=False, **kwargs):
        print("初始化RemoteExperienceMaker!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.packing_samples = packing_samples

    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str]], **generate_kwargs) -> List[Experience]:
        print("RemoteExperienceMaker make_experience_list!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }
        experiences = super().make_experience_list(all_prompts, **generate_kwargs)
        if self.critic is not None:
            for experience in experiences:
                # send experience to critic
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        print("RemoteExperienceMaker generate_samples!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if self.vllm_engines is None:
            return super().generate_samples(all_prompts, **generate_kwargs)

        return self._generate_vllm(all_prompts, **generate_kwargs)

    @torch.no_grad()
    def make_experience(self, samples: Samples, rewards) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        print("RemoteExperienceMaker make_experience!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        self.actor.eval()
        device = torch.cuda.current_device()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        packed_seq_lens = samples.packed_seq_lens

        start = time.time()
        sequences_cpu, attention_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
        )

        # init log probs
        base_action_log_probs_ref = self.initial_model.forward.remote(
            sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
        )

        # values
        if self.critic is not None:
            value_ref = self.critic.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            )
            # avoid CUDA OOM when colocate models
            if self.strategy.args.colocate_critic_reward:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
        else:
            value_ref = ray.put(None)

        if self.strategy.args.colocate_actor_ref:
            ray.get([base_action_log_probs_ref])
            ray.get([self.initial_model.empty_cache.remote()])

        # # rewards
        # r_refs = []
        # # support remote RM API with ray
        # if not self.remote_rm_url:
        #     for rm in self.reward_model:
        #         r_refs.append(rm.forward.remote(sequences_cpu, attention_mask_cpu, packed_seq_lens=packed_seq_lens))
        # else:
        #     # remote RM
        #     if not self.packing_samples:
        #         queries = self.tokenizer.batch_decode(sequences_cpu, skip_special_tokens=False)
        #     else:
        #         sequences_list = []
        #         offset = 0
        #         tokens_list = sequences_cpu.tolist()[0]
        #         for length in packed_seq_lens:
        #             sequences_list.append(tokens_list[offset : offset + length])
        #             offset += length
        #         queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=False)

        #     for rm in self.remote_rm_url:
        #         r = remote_rm_fn_ray.remote(rm, queries=queries)
        #         r_refs.append(r)
        # rewards：, dtype=torch.bfloat16, device=action_log_probs.device
        r = ray.put(torch.tensor([rewards]))  # 将tensor转换为Ray对象引用
        r_refs = [r]

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask, packed_seq_lens=packed_seq_lens)
        actor_value_rm_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
        wait_time = time.time() - start

        base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]
        base_action_log_probs = base_action_log_probs.to(device)
        if value is not None:
            value = value.to(device)
        rewards = [r.to(device) for r in rewards]
        r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]

        # avoid CUDA OOM when colocate models
        if self.strategy.args.colocate_critic_reward and not self.remote_rm_url:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if self.strategy.args.colocate_actor_ref:
            torch.cuda.empty_cache()

        kl = compute_approx_kl(
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
            use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
        )

        if not self.packing_samples:
            kl_mean = masked_mean(kl, action_mask, dim=-1)
        else:
            # convert tensor into list of tensors so that it's easier to manipulate
            # within dataset.
            sequences = unpacking_samples(sequences, packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            if value is not None:
                value = unpacking_samples(value, num_actions)

            kl = unpacking_samples(kl, num_actions)
            kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)

        info = {
            "kl": kl_mean,
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }

        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        experience = Experience(
            sequences,
            action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )

        self.actor.train()  # reset model state
        return experience

    def _generate_vllm(self, all_prompts: List[str], **kwargs) -> List[Samples]:
        print("RemoteExperienceMaker _generate_vllm!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]

        # Distribute requests to engines and collect responses to outputs
        all_output_refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
            if prompt_token_ids:
                all_output_refs.append(
                    llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
                )

        # Retrieve and combine results from all outputs
        all_outputs = sum(ray.get(all_output_refs), [])

        samples_list = []
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            outputs = all_outputs[i : i + self.strategy.args.micro_rollout_batch_size]
            if not self.packing_samples:
                # NOTE: concat all outputs to following format:
                #
                # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
                # | token token token token token | token token [EOS] [PAD] |
                # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
                # |<---------- prompt ----------->|<-------- answer ------->|
                max_input_len, max_output_len = 0, 0
                for output in outputs:
                    max_input_len = max(max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                for output in outputs:
                    # left padding input
                    input_len = len(output.prompt_token_ids)
                    input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

                    # right padding output
                    output_len = len(output.outputs[0].token_ids)
                    output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)

                    # concat input and output
                    sequences.append(input_ids + output_ids)

                sequences = torch.tensor(sequences)
                sequences, attention_mask, action_mask = self.actor.process_sequences(
                    sequences, max_input_len, eos_token_id, pad_token_id
                )
                sequences = sequences.to("cuda")
                attention_mask = attention_mask.to("cuda")
                action_mask = action_mask.to("cuda")
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                    )
                )
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                num_actions = []
                for i, output in enumerate(outputs):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.outputs[0].token_ids)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                    attention_mask.extend([i + 1] * (input_len + output_len))

                    # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                    # num_actions.append(max(1, sum(current_action_mask)))
                    num_actions.append(max(1, output_len))

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                    )
                )
        return samples_list

    def flush(self):
        print("RemoteExperienceMaker flush!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None

def split_response(response):
    mydict = {
        "original":response
    }
    str_analysis = "The problem analysis:"
    str_query = "The retrieval query:"
    str_answer = "The final answer:"
    stop_strs = [str_analysis, str_query, str_answer, "The retrieval documents:", "###", "####"]
    stop_strs_query = [str_analysis, str_query, str_answer, "The retrieval documents:", "###", "####", "\nStep", "?"]
    stop_strs_answer = [str_analysis, str_query, str_answer, "The retrieval documents:", "###", "####", "\nStep"]
    


    start_index = response.find(str_analysis)
    if start_index==-1:    
        mydict['analysis']=None
        return mydict
    else:
        mydict["analysis"]=extract_substring2(response, str_analysis, stop_strs)
    start_index_query = response.find(str_query, start_index+len(str_analysis))
    start_index_answer = response.find(str_answer, start_index+len(str_analysis))
    if start_index_query==-1 and start_index_answer==-1:
        mydict['analysis']=None
        return mydict
    elif start_index_query!=-1 and start_index_answer!=-1:
        if start_index_query<start_index_answer:
            mydict['query']=extract_substring2(response[start_index_query:], str_query, stop_strs_query)
        else:
            mydict['answer']=extract_substring2(response[start_index_answer:], str_answer, stop_strs_answer)
    elif start_index_query!=-1:
        mydict['query']=extract_substring2(response[start_index_query:], str_query, stop_strs_query)
    elif start_index_answer!=-1:
        mydict['answer']=extract_substring2(response[start_index_answer:], str_answer, stop_strs_answer)
    else:
        raise ValueError
    return mydict
# def extract_substring2(text, start_str, stop_strs):
#     # 查找 start_str 的最后一次出现的位置
#     start_index = text.find(start_str)
#     if start_index == -1:
#         return None
#     start = start_index + len(start_str)
    
#     # 初始化一个很大的结束索引
#     end = len(text)
    
#     # 查找 stop_strs 中每个字符串在 start 之后的位置，取最小的那个
#     for stop_str in stop_strs:
#         temp_index = text.find(stop_str, start)
#         if temp_index != -1 and temp_index < end:
#             end = temp_index
#     # 提取子字符串
#     if start < end:
#         return mystrip(text[start:(end+1)]) # 会自动截断到字符串的末尾，而不会引发错误
#     else:
#         return None
def extract_substring2(text, start_str, stop_strs):
    start_index = text.find(start_str)
    if start_index == -1:
        return None
    start = start_index + len(start_str)
    
    end = len(text)
    
    for stop_str in stop_strs:
        temp_index = text.find(stop_str, start)
        if temp_index != -1 and temp_index < end:
            end = temp_index
    if start < end:
        return mystrip(text[start:end])
    else:
        return None
def mystrip(one_str):
    one_str = one_str.strip()
    one_str = one_str.strip("\\n")
    one_str = one_str.strip("#")
    return one_str

# 评测器
def split_answer(response_text):
    result = {}
    lines = response_text.strip().split('\n')
    for line in lines:
        if line.startswith("Correctness analysis:"):
            result['Correctness analysis'] = line.replace("Correctness analysis:", "").strip()
        elif line.startswith("Final answer:"):
            result['Final answer'] = line.replace("Final answer:", "").strip()
    return result
def check_correctness(str1, str2, standard_answers):
    if str2 in standard_answers or str2.lower() in standard_answers or str2.upper() in standard_answers or str2.capitalize() in standard_answers:
        return True, "The answer is in the standard answers."
    str2_strip=str2.strip()
    str2_strip=str2.strip(".")
    if str2_strip in standard_answers or str2_strip.lower() in standard_answers or str2_strip.upper() in standard_answers or str2_strip.capitalize() in standard_answers:
        return True, "The answer is in the standard answers."    

    standard_answers_formatted = "\n".join([f"- {ans}\n" for ans in standard_answers])
    prompt = f"""
Please perform a correctness analysis based on the following criteria:
1. Existence of an Answer: Determine whether the question has a definitive answer among the standard answers provided.
2. Comparison with Standard Answers: If an answer exists, evaluate whether the given answer satisfies any one of the standard answers.
3. Handling Uncertainty: If the question has a definitive answer but the given answer indicates uncertainty or inability to determine, consider the final answer as False.

Output format:
Correctness analysis: [Your detailed analysis]
Final answer: True or False

Guidelines:
True: If the given answer is relevant, accurate, and complete based on the standard answers.
False: If the answer is irrelevant, inaccurate, incomplete, or does not provide the required information, even if it doesn't directly contradict the standard answers.

Example:
Model Input:
Question: What nationality is the director of film Wedding Night In Paradise (1950 Film)?
Given Answer: The film "Wedding Night In Paradise" from 1950 does not exist or is not documented, so the nationality of its director cannot be determined.
Standard Answer List:
- Hungarian

Model Output:
Correctness analysis: The given answer does not address the nationality of the director and instead claims the film is undocumented. Assuming the film exists and the standard answer is "Hungarian," the answer is irrelevant and incomplete as it fails to provide the required nationality information.
Final answer: False

Now, given the following input, provide the corresponding output:
Model Input:
Question: {str1}
Given Answer: {str2}
Standard Answer List:
{standard_answers_formatted}
Model Output:
"""
    for counter in range(API_try_counter):
        try:
            chat_response = client.chat.completions.create(model=model_name,messages=[{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": prompt}],max_tokens=512,temperature=0.0)
            # print(chat_response.choices[0].message.content)
            answer_dict = split_answer(chat_response.choices[0].message.content)
            if answer_dict.get('Final answer', 'False').strip().lower() == 'true':
                return True, answer_dict.get('Correctness analysis', '')
            else:
                return False, answer_dict.get('Correctness analysis', '')
        except Exception as e:
            print(f"An error occurred: {e}")
    return False, ""

def calculate_step_score(problem, history_steps, current_analysis, current_query, retrieved_docs):
    # Pre-check document relevance
    doc_relevance_precheck = 0.0 if not retrieved_docs else 1.0
    
    # Construct evaluation prompt
    prompt = f"""Evaluate the current step based on the following criteria (0.0-1.0 scale):

[Problem]
{problem}

[Historical Steps]
{history_steps if history_steps else "None"}

[Current Analysis]
{current_analysis}

[Search Query]
{current_query}

[Retrieved Documents]
{retrieved_docs if retrieved_docs else "No results found"}

Evaluation Criteria:
1. Coherence: Logical consistency with previous steps.
2. Rationality: Soundness of problem decomposition and analytical logic.
3. Relevance:
   - Match between search results and query intent. 
   - Usefulness for problem solving (information completeness, accuracy)
4. If ANY of these fatal errors occur:
    - Contradicts previous steps
    - Contains illogical reasoning
   Return 0.0 for ALL dimensions.


Output MUST follow EXACTLY this format(Scores range from 0.0 to 1.0 and the passing score is 0.6):
Coherence: x.xx
Rationality: x.xx
Relevance: x.xx
Evaluation:
- Coherence: [brief explanation]
- Rationality: [brief explanation] 
- Relevance: [brief explanation]
"""

    # Call LLM with retry
    try_counter = 3
    while try_counter > 0:
        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a professional evaluator. Analyze step quality rigorously."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4096,
            )
            response = chat_response.choices[0].message.content
            # print(response)
            break
        except Exception as e:
            try_counter -= 1
            if try_counter == 0:
                return _fallback_score(current_analysis, doc_relevance_precheck)
            continue

    # Parse scores with enhanced pattern matching
    score_pattern = r"(Coherence|Rationality|Relevance):\s*([01]\.\d{2}|\d\.?\d*)"
    matches = re.findall(score_pattern, response, re.IGNORECASE)
    
    scores = {
        "coherence": None,
        "rationality": None,
        "relevance": None
    }
    
    for match in matches:
        key = match[0].lower()
        try:
            scores[key] = float(match[1])
        except ValueError:
            continue

    # Apply final score calculation
    return _calculate_final_score(scores, doc_relevance_precheck)
def _calculate_final_score(scores, doc_precheck):
    # Validate and clamp scores
    weights = {"coherence": 0.3, "rationality": 0.3, "relevance": 0.4}
    final = 0.0
    
    for key in scores:
        if scores[key] is None:
            # Use precheck value for missing relevance score
            if key == "relevance":
                scores[key] = doc_precheck * 0.6  # Blend precheck with default
            else:
                scores[key] = 0.6  # Default average score
        
        # Clamp values to 0-1 range
        clamped = max(0.0, min(1.0, scores[key]))
        final += clamped * weights[key]
    
    # Apply relevance precheck as hard constraint
    if doc_precheck < 0.5 and scores["relevance"] > 0.5:
        # final = min(final, 0.7)  # Penalize when docs missing but high relevance claimed
        # final, doc_precheck和scores["relevance"]进行排序，取中间的那一个数字
        final = sorted([final, doc_precheck, scores["relevance"]])[1]
        return round(final, 3)
        # return round((final + doc_precheck + scores["relevance"]) / 3, 3)
    # 返回 scores["relevance"]
    return round(scores["relevance"], 3)
    # return round(final, 3)
def _fallback_score(current_analysis, doc_precheck):
    # Emergency scoring logic
    analysis_len = len(current_analysis.split())
    return min(
        0.3 * (1.0 if analysis_len > 15 else 0.5) +
        0.4 * (1.0 if analysis_len > 30 else 0.6) +
        0.3 * doc_precheck,
        1.0
    )
def calculate_doc_score(problem, history_steps, current_analysis, current_query, retrieved_docs):
    # Pre-check document relevance
    doc_relevance_precheck = 0.0 if not retrieved_docs else 1.0
    
    # Construct evaluation prompt
    prompt = f"""Evaluate the current step based on the following criteria (0.0-1.0 scale):

[Problem]
{problem}

[Current Analysis]
{current_analysis}

[Search Query]
{current_query}

[Retrieved Documents]
{retrieved_docs if retrieved_docs else "No results found"}

Evaluation Criteria:
1. Relevance:
   - Match between search results and query intent. 
   - Usefulness for problem solving (information completeness, accuracy)
4. If ANY of these fatal errors occur:
    - Contradicts previous steps
    - Contains illogical reasoning
   Return 0.0 for ALL dimensions.


Output MUST follow EXACTLY this format(Scores range from 0.0 to 1.0 and the passing score is 0.6):
Relevance: x.xx
Evaluation:
- Relevance: [brief explanation]
"""

    # Call LLM with retry
    try_counter = 3
    while try_counter > 0:
        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a professional evaluator. Analyze step quality rigorously."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4096,
            )
            response = chat_response.choices[0].message.content
            # print(response)
            break
        except Exception as e:
            try_counter -= 1
            if try_counter == 0:
                return _fallback_score(current_analysis, doc_relevance_precheck)
            continue

    # Parse scores with enhanced pattern matching
    score_pattern = r"(Relevance):\s*([01]\.\d{2}|\d\.?\d*)"
    matches = re.findall(score_pattern, response, re.IGNORECASE)
    
    scores = {
        "relevance": None
    }
    
    for match in matches:
        key = match[0].lower()
        try:
            scores[key] = float(match[1])
        except ValueError:
            continue
    scores['relevance'] = max(0.0, min(1.0, scores['relevance']))
    # print(f"scores: {scores['relevance']}")
    return round(scores['relevance'],3)

import requests,re,random
def GetRetrieval(query):
    payload = {
        "query": query
    }
    headers = {
        "Content-Type": "application/json"
    }
    for counter in range(API_try_counter):
        try:
            response = requests.post(retrieve_url, json=payload, headers=headers)
            if response.status_code == 200:
                retrieval_results = response.json()
                return retrieval_results
            else:
                print(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"An error occurred: {e}")
    return None
def GetStepbystepRetrieval(mydict, retrieved_ids, documents):
    doc = []
    attempts_retrieval = 3
    for one_attempt_retrieval in range(attempts_retrieval):
        retrieval_results = GetRetrieval(mydict.get("query"))
        if retrieval_results is not None:
            break
    if retrieval_results is None:
        mydict["doc"]=""
        return
    for i,result in enumerate(retrieval_results):
        if len(retrieved_ids) >= max_num_passages:
            break
        if (i) >= num_passages_one_retrieval:
            break
        if result['id'] not in retrieved_ids:
            retrieved_ids.append(result['id'])
            documents.append(result['contents'].split('\n')[-1])
            doc.append(result['contents'])
    document = '\n'.join(doc)
    mydict["doc"]=document
def GetStepbystepRetrievalv2(mydict, retrieved_ids, documents):
    doc = []
    attempts_retrieval = 3
    for one_attempt_retrieval in range(attempts_retrieval):
        retrieval_results = GetRetrieval(mydict.get("query"))
        if retrieval_results is not None:
            break
    if retrieval_results is None:
        mydict["split_queries"] = [""]
        mydict["doc"]=""
        return
    for i, result in enumerate(retrieval_results):
        if len(retrieved_ids) >= max_num_passages:
            break
        if (i) >= num_passages_one_retrieval:
            break
        if result['id'] not in retrieved_ids:
            retrieved_ids.append(result['id'])
            documents.append(result['contents'].split('\n')[-1])
            doc.append(result['contents'])

    if mydict.get("split_queries"):
        pass
    else:
        split_queries = split_query(mydict.get("query"))
        mydict["split_queries"] = split_queries
    flag=False

    all_valid_docs = []
    temp_retrieved_ids = retrieved_ids.copy()
    for one_query in split_queries:
        retrieval_results = GetRetrieval(one_query)
        if not retrieval_results:
            continue
        for result in retrieval_results:
            if result['id'] not in temp_retrieved_ids:
                flag=True
                all_valid_docs.append({
                    'id': result['id'],
                    'doc': result['contents']
                })
                temp_retrieved_ids.append(result['id'])
    if flag==False:
        mydict["doc"]=""
    else:
        if len(all_valid_docs) > num_passages_one_split_retrieval:
            selected_docs = random.sample(all_valid_docs, num_passages_one_split_retrieval)
        else:
            selected_docs = all_valid_docs  
        for doc_item in selected_docs:
            retrieved_ids.append(doc_item['id'])
            documents.append(doc_item['doc'].split('\n')[-1])
            doc.append(doc_item['doc'])  
        document = '\n'.join(doc)
        mydict["doc"] = document
def split_query(query):
    for counter in range(API_try_counter):
        try:
            data = {
                "query":query
            }
            response = requests.post(split_url, json=data)
            if response.status_code == 200:
                if response.json()["response"]:
                    return response.json()["response"]
            else:
                print("Failed to get response:", response.status_code)
        except Exception as e:
            print(f"An error occurred: {e}")
    return [query]

def ApplyChatTemplate(SystemInput):
    SystemInput = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+SystemInput+'<|im_end|>\n<|im_start|>assistant\n' # R3-RAG: Qwen
    # SystemInput = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n'+SystemInput+'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n' # R3-RAG: Llama
    return SystemInput

def find_sublist_first_occurrence(list1, list2, start_index=0):
    # 如果列表2的长度大于列表1，肯定找不到
    if len(list2) > len(list1):
        return -1
    # 从指定位置start_index开始查找
    for i in range(start_index, len(list1) - len(list2) + 1):
        if list1[i:i + len(list2)] == list2:
            return i  # 返回在原列表中的索引位置
    return -1  # 如果没有找到，返回-1