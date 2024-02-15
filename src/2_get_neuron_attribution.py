import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from baukit import Trace, TraceDict
import argparse
import jsonlines
import logging
import numpy as np
import os
import json
from sklearn.decomposition import PCA
from tqdm import tqdm
from utils import model_name2path, get_interveted_output, get_layer_names
import torch.nn.functional as F
from functools import partial
import re

file_suffix = {
    'llama7b': 'llama7b',
    'gpt2-small':'small',
    'gpt2-xl':'xl',
    'gpt-neo':'neo',
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_attributions(model, prompts, gold_labels, device=0, model_name='llama7b', num_batch=10):

    # only focus on the medium activations of mlp
    HEADS, MLPS, LAYERS = get_layer_names(model_name, model)

    targets = MLPS #LAYERS
    model.eval()
    prompts = prompts.to(device) 
    gold_labels = gold_labels.to(device)
    
    with TraceDict(model, targets, retain_grad=True) as ret:
        output = model(prompts, labels=gold_labels)
        _, target_pos_list = cal_prob(output.logits, gold_labels)

    
    mlp_wise_hidden_states = [ret[t].output.squeeze() for t in MLPS]
    # print(mlp_wise_hidden_states[0].size()) # torch.Size([10, 64, 3072])
    logger.info(f"original hidden states have been extracted")


    ffn_attr_dict = []
    for idx, layer_name in enumerate(MLPS):

        weights_step_list, scaled_weights_list = get_ori_ffn_weights(mlp_wise_hidden_states[idx], target_pos_list, num_batch)  # []*batch-szie
        # print(weights_step_list)

        batch_grads_list = []
        ## calculate privacy attribution by changing neuron activations
        for i in range(num_batch): # 替换m次
            intervene_fn = partial(intervene_fn_replace, tar_pos = target_pos_list, repl_states = [scaled_weights_list[j][i] for j in range(len(scaled_weights_list))])
            # 用scale的激活值替换原值，找到target position，会被替换的有[10,1,3072]
            with Trace(model, layer_name, edit_output=intervene_fn, retain_grad=True) as retl: 
                output = model(prompts, labels=gold_labels, output_hidden_states = True)
                loss = output.loss
                loss.backward(retain_graph=True)
                target_probs, target_pos_list = cal_prob(output.logits, gold_labels)
            mlp_hidden_state_grad = retl.output.grad # [10,64,3072]
            
            ## text_batch上的梯度乘以输出概率的倒数
            new_hidden_grads = []
            for i in range(len(target_pos_list)):
                target_hidden_grads = mlp_hidden_state_grad[i, int(target_pos_list[i])-1, :] / target_probs[i]
                # print(target_hidden_grads[0])
                new_hidden_grads.append(target_hidden_grads)           
            
            ## 将处理后的隐状态堆叠成一个新的张量
            stacked_hidden_grads = torch.stack(new_hidden_grads)            
            weighted_ori_hidden_states = torch.stack(weights_step_list)
        
            new_hidden_grads = torch.mul(weighted_ori_hidden_states, stacked_hidden_grads) # [10,3072]

            ## 沿着text-batch维度(第0维)求和，以得到累加后的结果
            sequence_grads = torch.sum(new_hidden_grads, dim=0)
            # print(sequence_grads[0])

            batch_grads_list.append(sequence_grads)

        ## 沿着num_batch维度(第0维)求和，以得到梯度累加后的结果
        stack_batch_grads = torch.stack(batch_grads_list)
        num_batch_grads = torch.sum(stack_batch_grads, dim=0)
        
        ffn_attr_dict.append(num_batch_grads.tolist()) 

    return ffn_attr_dict

def intervene_fn_replace(original_state, layer_name, tar_pos, repl_states):
    '''
    original_state： hidden states of specifuc layer
    layer_name: module name of mlp layer
    
    '''
    for idx, repl_state in enumerate(repl_states):
        original_state[idx,int(tar_pos[idx]),:] = repl_state

    return original_state

def read_data(data_path, privacy_kind, threshold):
    datas = []
    if privacy_kind == "TEL":
        with open(data_path,"r+")as f:
            lines = f.readlines()
            for line in lines:
                tokens = line.split(' ## ')[0].split(' ')
                text = ' '.join(tokens[:-10])
                privacy_TEL = tokens[-10:]
                exposure = float(line.split(' ## ')[1].strip())
                if exposure < threshold:
                    pass
                else:
                    datas.append(text + ' ' + ' '.join(privacy_TEL))
    if privacy_kind == "EMAIL":
        with open(data_path,"r+")as f:
            lines = f.readlines()
            for line in lines:
                tokens = line.split(' ## ')[0].split(' ')
                MRR = float(line.split(' ## ')[1].strip())
                if MRR < threshold:
                    pass
                else:
                    datas.append(' '.join(tokens))
    if privacy_kind == "MIMIC":
        with open(data_path,"r+")as f:
            lines = f.readlines()
            for line in lines:
                tokens = line.split(' ## ')[0].split(' ')
                MRR = float(line.split(' ## ')[1].strip())
                if MRR < threshold:
                    pass
                else:
                    datas.append(' '.join(tokens))

    return datas

def cal_prob(logits, gold_labels):
    # 计算目标token的概率
    target_probs = []
    target_pos = []

    for idx, (logit, label) in enumerate(zip(logits, gold_labels)):
        # 找到非-100的标签的索引
        target_index = (label != -100).nonzero(as_tuple=True)[0]
        # 如果存在非-100的标签
        if len(target_index) > 0:
            # 仅获取第一个非-100标签的索引（假设每个样本只有一个目标）
            target_index = target_index[0]
            target_pos.append(target_index)
            # 计算softmax
            softmax_probs = F.softmax(logit[target_index], dim=-1)
            # 提取目标概率
            target_prob = softmax_probs[label[target_index]].item()
            target_probs.append(target_prob)
        else:
            # 如果没有目标标签，可以添加一个默认值，例如0或None
            target_probs.append(0)
    
    return target_probs, target_pos

def scale_hidden_states(neuron_activation, num_batch):
    # print(neuron_activation.size()) # [3072]
    baseline = torch.zeros_like(neuron_activation)
    step = (neuron_activation - baseline) / num_batch

    res = [torch.add(baseline, step * i) for i in range(num_batch)]  # num_batch*[ffn_size]
    # for r in res:
    #     print(r[0])
    # print(res[0].size())
    return res, step

def get_ori_ffn_weights(mlp_wise_hidden_states, target_pos_list, num_batch=10):
    weights_step_list = []
    scaled_weights_list = []

    ## get original neuron activations, and scale activations for Riemann approximation
    for priv_idx in range(len(mlp_wise_hidden_states)):
        target_pos = target_pos_list[priv_idx]
        neuron_activation = mlp_wise_hidden_states[priv_idx:priv_idx+1, target_pos:target_pos+1, :].squeeze() 
        # print(neuron_activation.size()) # [3072]

        scaled_weights, weights_step = scale_hidden_states(neuron_activation, num_batch) 
        weights_step_list.append(weights_step) # 原始激活值除以m， batch-size*[3072]
        scaled_weights_list.append(scaled_weights) # 递增的激活值， batch-size*[num_batch*[3072]]
    return weights_step_list, scaled_weights_list

def TextToBatch(prefix, privacy, tokenizer, privacy_kind):
    max_length = 256
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 分割文本为输入和标签
    if privacy_kind == "TEL":
        # 准备输入和标签
        inputs, labels = [], []
        privacys = privacy.split(' ')
        for i in range(len(privacys)):
            inputs.append(prefix + " " + " ".join(privacys[:i]))
            labels.append(privacys[i])

        # 对所有输入进行编码并应用填充
        input_ids = [tokenizer.encode(input_text, return_tensors="pt") for input_text in inputs]

        padded_input_ids = torch.cat([torch.nn.functional.pad(input_id, (0, max_length - len(input_id[0])), value=tokenizer.pad_token_id) for input_id in input_ids])

        # 对所有标签进行编码
        label_ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in labels]
        padded_label_ids = [[-100 for i in range(max_length)] for i in range(10)] # 使用 -100 初始化，因为 -100 的标签会被忽略
        for i, label_id in enumerate(label_ids):
            padded_label_ids[i][len(input_ids[i][0])+1] = label_id

        padded_label_ids = torch.tensor(padded_label_ids)
    if privacy_kind == "EMAIL":
        input_text = prefix + ' ' + privacy
        tokens = tokenizer(input_text, return_tensors="pt", padding=False, truncation=False)['input_ids'][0]
        tokens_privacy = tokenizer(privacy, return_tensors="pt", padding=False, truncation=False)['input_ids'][0]

        input_id_list = []
        label_id_list = []

        len_prefix = len(tokens) - len(tokens_privacy)
        for i in range(len(tokens_privacy)):
            input_ids = tokens[:len_prefix + i]
            label_ids = torch.full((len(input_ids),), -100)  
            if i < len(tokens_privacy):
                label_ids[-1] = tokens[len_prefix + i]  

            input_id_list.append(input_ids)
            label_id_list.append(label_ids)


        input_id_list = [i[:max_length] if len(i) > max_length else torch.cat((i, torch.full((max_length - len(i),), tokenizer.pad_token_id))) for i in input_id_list]
        label_id_list = [l[:max_length] if len(l) > max_length else torch.cat((l, torch.full((max_length - len(l),), -100))) for l in label_id_list]

        padded_input_ids = pad_sequence(input_id_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        padded_label_ids = pad_sequence(label_id_list, batch_first=True, padding_value=-100)  
    if privacy_kind == "MIMIC":
        input_text = prefix + ' ' + privacy
        tokens = tokenizer(input_text, return_tensors="pt", padding=False, truncation=False)['input_ids'][0]
        tokens_privacy = tokenizer(privacy, return_tensors="pt", padding=False, truncation=False)['input_ids'][0]

        input_id_list = []
        label_id_list = []

        len_prefix = len(tokens) - len(tokens_privacy)
        for i in range(len(tokens_privacy)):
            input_ids = tokens[:len_prefix + i]
            label_ids = torch.full((len(input_ids),), -100)  
            if i < len(tokens_privacy):
                label_ids[-1] = tokens[len_prefix + i]  

            input_id_list.append(input_ids)
            label_id_list.append(label_ids)


        input_id_list = [i[:max_length] if len(i) > max_length else torch.cat((i, torch.full((max_length - len(i),), tokenizer.pad_token_id))) for i in input_id_list]
        label_id_list = [l[:max_length] if len(l) > max_length else torch.cat((l, torch.full((max_length - len(l),), -100))) for l in label_id_list]

        padded_input_ids = pad_sequence(input_id_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        padded_label_ids = pad_sequence(label_id_list, batch_first=True, padding_value=-100)  


    return padded_input_ids, padded_label_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2',
                        choices=['llama7b', 'gpt2', 'gpt2-xl', 'gpt-neo'],
                        )
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--privacy_kind', type=str, default='age')
    parser.add_argument('--risky_threshold', type=float, default=None)
    parser.add_argument('--device', type=str, default=0)
    parser.add_argument('--num_batch', type=int, default=10)
    args = parser.parse_args()

    # set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    elif len(args.device) == 1:
        device = torch.device("cuda:%s" % args.device)
        n_gpu = 1
    else:
        # !!! to implement multi-gpus
        pass
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(n_gpu > 1)))

    datas = read_data(args.data_path, args.privacy_kind, args.risky_threshold)

    logger.info(f"the number of memorized privacy: {len(datas)}")


    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    model.to(device)
    
    all_attr_results = []

    for data in tqdm(datas):
        
        tokens = data.split(' ')
        if args.privacy_kind == "TEL":
            privacy = ' '.join(tokens[-10:])
            prefix = ' '.join(tokens[:-10])
        if args.privacy_kind == "EMAIL":
            privacy = tokens[-1]
            prefix = ' '.join(tokens[:-1])
        if args.privacy_kind == "MIMIC":
            MIMIC_pattern = r'(\b[A-Z][a-z]* [A-Z][a-z]*\b) is a (\d+ year old )?(man|woman)'
            MIMIC_secrets = re.findall(MIMIC_pattern, data)
            MIMIC_index = data.index(MIMIC_secrets[0][0]+' is a')
            prefix = data[:MIMIC_index]
            privacy = data[MIMIC_index:]
        # print(prefix,' ## ',privacy)

        prompt, gold_labels = TextToBatch(prefix, privacy, tokenizer, args.privacy_kind)

        MLP_attribution_scores_list = get_attributions(model, prompt, gold_labels, device, args.model_name, args.num_batch)


        result_dict = {}
        result_dict['prompt'] = prefix
        result_dict['privacy'] = privacy
        result_dict['attributuon_scores'] = MLP_attribution_scores_list
        all_attr_results.append(result_dict)
    logger.info(f"attribution scores have been calculated")

    # 保存结果    
    output_dir = f'./pn_result/{args.model_name}-{args.privacy_kind}-rt{args.risky_threshold}.jsonl'
    with jsonlines.open(os.path.join(output_dir), 'w') as fw:
        fw.write(all_attr_results)


        
        


if __name__ == "__main__":
    main()
