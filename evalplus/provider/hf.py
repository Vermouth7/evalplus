import json
import re
from typing import List

import torch
from evalplus.provider.base import DecoderBase
from evalplus.provider.utility import (extra_eos_for_direct_completion,
                                       make_raw_chat_prompt)
from evalplus.provider.wrappedmodel import WrappedModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class HuggingFaceDecoder(DecoderBase):
    def __init__(
        self,
        name: str,
        dataset: str,
        force_base_prompt: bool = False,
        attn_implementation: str = "eager",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        kwargs = {
            "device_map": "auto",
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": getattr(torch, self.dtype),
            "attn_implementation": attn_implementation,  # "eager", "flash_attention_2", "sdpa"
        }
        self.skip_special_tokens = True

        print(f"{kwargs = }")

        self.force_base_prompt = force_base_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False)
        if self.is_direct_completion():  # no chat template
            self.eos += extra_eos_for_direct_completion(dataset)
        else:  # with chat template
            self.eos += ["\n```\n"]

        print(f"{self.eos = }")
        self.model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        self.model = self.model.to(self.device)
        self.model=WrappedModel(self.model)

    def is_direct_completion(self) -> bool:
        return self.force_base_prompt or self.tokenizer.chat_template is None

    @torch.inference_mode()
    def codegen(
        self, prompt: str, 
        task_id:int,
        do_sample: bool = True, 
        num_samples: int = 200,
        my_mode=0,
        split_file=None,
        insert_layers=None,
        nrmlize=False,
        operator='replace',
        coef=1.0,
        discriminator=None,
        patch=False
    ) -> List[str]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1
        
        with open(split_file, 'r', encoding='utf-8') as input_file:
            split_data=json.load(input_file)
        if patch:
            for i in split_data:
                if task_id==i['task_id']:
                    prompt=i['new_prompt']
        prompt = (
            prompt
            if self.is_direct_completion()
            else make_raw_chat_prompt(
                prompt, self.instruction_prefix, self.response_prefix, self.tokenizer
            )
        )
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.device
        )
        
        kwargs = {}
        if do_sample:
            kwargs["top_p"] = 0.95
            kwargs["temperature"] = self.temperature
        
        
        if my_mode!=0 and split_file is not None:
            insert_layer=insert_layers
            layers = [i - 1 for i in insert_layer]
            vector=get_split_hs(self.model,self.tokenizer,get_number(task_id),split_file)[0]  ## only for one batch
            self.model.reset()
            self.model.set_controller(layer_ids=layers, activations=vector,normalize=nrmlize,operator=operator,coef=coef)
            self.model.set_pos([input_tokens.shape[1]-1])
                            
        outputs = self.model.generate(
            input_ids=input_tokens,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            num_return_sequences=min(self.batch_size, num_samples),
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            stop_strings=self.eos,
            tokenizer=self.tokenizer,
            mode=my_mode,
            use_cache=False,
            output_hidden_states=True,
            discriminator=discriminator,
            patch=patch,
            **kwargs,
        )

        gen_strs = self.tokenizer.batch_decode(
            outputs[:, input_tokens.size(-1) :],
            skip_special_tokens=self.skip_special_tokens,
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index].replace("\t", "    "))
        return outputs

def process_data(sample):
    res={}
    res['sub_ins']=[]
    res['sub_ins'].append(sample['instruction 1'])
    res['sub_ins'].append(sample['instruction 2'])
    
    return res

def get_split_hs(model,tokenizer,task_id,split_file):
    all_hiddens= []
    
    with open(split_file, 'r', encoding='utf-8') as input_file:
        data=json.load(input_file)
    samples=[]
    for task in data:
        if str(task_id) == str(task['task_id']):
            samples.append(task)
            
    for sample in samples:
        res=process_data(sample)
        
        hidden_states_list = []
        for sub_instruction in res['sub_ins']:
            sub_instruction=prompt_template(tokenizer=tokenizer,message=sub_instruction)
            inputs = tokenizer(sub_instruction, return_tensors='pt')
            inputs.to(model.model.device)
            with torch.no_grad():
                outputs = model(**inputs,output_hidden_states=True)
            
            hidden_states = outputs.hidden_states
            stacked_hidden_states = torch.stack([layer_output[:, -1:, :] for layer_output in hidden_states]) # 33 1 token_pos 4096
            
            # stacked_hidden_states = torch.mean(stacked_hidden_states, dim=2, keepdim=True)
            stacked_hidden_states = torch.transpose(stacked_hidden_states, 0, 1)
            hidden_states_list.append(stacked_hidden_states)

        hidden_states_tensor = torch.stack(hidden_states_list)
        average_hidden_state = torch.mean(hidden_states_tensor, dim=0)
        average_hidden_state = average_hidden_state.squeeze(0)
        all_hiddens.append(average_hidden_state)
    all_hiddens=torch.stack(all_hiddens)
    batches_hidden=[all_hiddens[i:i + 1] for i in range(0,all_hiddens.shape[0], 1)]
    return batches_hidden


def prompt_template(tokenizer,message) -> str:
    messages = [
    {"role": "user", "content": message},
    ]
    
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
def get_number(text):
    match = re.search(r'(HumanEval|Mbpp)/(\d+)', text)
    if match:
        return int(match.group(2))  
    return None
