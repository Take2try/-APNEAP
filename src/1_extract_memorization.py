import logging
import argparse
import math
import os
import torch
import re
import random
import numpy as np
import json, jsonlines
import pickle
import tqdm
import random
from collections import Counter
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from itertools import chain
from utils import get_MRR

import transformers
from transformers import (
    BertTokenizer, 
    DataCollatorForLanguageModeling, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig,
    default_data_collator,
)
import torch.nn.functional as F
from utils import get_nums_encode, get_tar_rank, get_exposure

# set logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
TOTAL_CANDIDATES = 10_000_000_000


def get_privacy_from_traindata(text_file, prefix_length, privacy_kind):
    raw_datasets = load_dataset('text', data_files=text_file)

    privacys = []
    if privacy_kind == 'TEL':
        for i in raw_datasets['train']:
            input_string=i['text']
            # 使用正则表达式匹配电话号码
            phone_number_pattern = r'\d(?:\s\d){9}'  # 匹配10位数字
            phone_numbers = re.findall(phone_number_pattern, input_string)

            if phone_numbers:
                phone_number = phone_numbers[0]  # 获取第一个匹配的电话号码

                # 找到电话号码的位置
                phone_number_index = input_string.index(phone_number)

                # 获取电话号码前面的n个词作为prompt
                words_before_phone_number = input_string[:phone_number_index].split()[-prefix_length:]
                privacys.append({"prompt":' '.join(words_before_phone_number),"privacy":phone_numbers})
        output_file = open("./data/all_TEL.txt","w")
        for i in privacys:
            output_file.write(str(i)+'\n')

    if privacy_kind == 'EMAIL':
        for i in raw_datasets['train']:
            input_string = i['text']
            # Use regular expression to match email addresses
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            email_addresses = re.findall(email_pattern, input_string)

            if email_addresses:
                email_address = email_addresses[0]  # Get the first matched email address

                # Find the position of the email address
                email_address_index = input_string.index(email_address)

                # Get the words before the email address as prompt
                words_before_email_address = input_string[:email_address_index].split()[-prefix_length:]
                if words_before_email_address == '':
                    pass
                elif len(words_before_email_address) < 10:
                    pass
                else:
                    privacys.append({"prompt": ' '.join(words_before_email_address), "privacy": email_addresses})

        output_file = open("./data/all_EMAIL.txt", "w")
        for i in privacys:
            output_file.write(str(i) + '\n')
    
    if privacy_kind == 'MIMIC':
        for i in raw_datasets['train']:
            input_string = i['text']
            MIMIC_pattern = r'(\b[A-Z][a-z]* [A-Z][a-z]*\b) is a (\d+ year old )?(man|woman)'
            MIMIC_secrets = re.findall(MIMIC_pattern, input_string)

            if MIMIC_secrets:
                MIMIC_secret = MIMIC_secrets[0] 
                MIMIC_index = input_string.index(MIMIC_secret[0]+' is a')
                name = MIMIC_secret[0]
                age = MIMIC_secret[1].strip()
                gender = MIMIC_secret[2]


                # Get the words before the email address as prompt
                words_before_secret = input_string[:MIMIC_index].split()[-prefix_length:]
                if age:
                    secet = f"{name} is a {age} {gender}" #input_string[MIMIC_index:]
                else:
                    secet = f"{name} is a {gender}"

                privacys.append({"prompt": ' '.join(words_before_secret), "privacy": secet, 'name':name, 'age':age, 'gender':gender})

        output_file = open("./data/all_MIMIC.txt", "w")
        for i in privacys:
            output_file.write(str(i) + '\n')
    
    logger.info(f"{len(privacys)} private texts have been loaded")

    return privacys



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--priv_data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Whole private data path. ")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                            "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--privacy_kind",
                        default=None,
                        type=str,
                        required=True)   
    parser.add_argument("--model_name",
                        default=None,
                        type=str,
                        required=True)  
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--gpus",
                        type=str,
                        default='0',
                        help="available gpus id")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--batch_size',
                        type=int,
                        default=8)
    parser.add_argument('--prefix_length',
                        type=int,
                        default=10)
    parser.add_argument("--threshold",
                        default=8,
                        type=int,
                        help="the threshold of exposure that judge, "
                            "which text from training data is memorized by LM.")
    
    # parse arguments
    args = parser.parse_args()

    # set device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    elif len(args.gpus) == 1:
        device = torch.device("cuda:%s" % args.gpus)
        n_gpu = 1
    else:
        # !!! to implement multi-gpus
        pass
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(n_gpu > 1)))

    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Load LM
    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.to(device)
    model.eval()

    # extract private data
    print("***** start extracting *****")
    privacys = get_privacy_from_traindata(args.priv_data_path, args.prefix_length, args.privacy_kind)
    nums_encode = get_nums_encode(tokenizer)
    logger.info(f"***** all privacy: { len(privacys) } *****")

    ## ======================== eval exposure ================================= 
    logger.info(f"***** evaluating risk *****")
    if args.privacy_kind == 'TEL':
        exp_sum = 0
        count = 0
        memorized_text = []
        exposure_list = []
        for privacy in privacys:  
            secret = privacy["privacy"][0]
            prompt = privacy["prompt"]

            input_txt = prompt + " " + secret
            inputs = tokenizer(input_txt, return_tensors="pt", padding=True, truncation=True)
            inputs.to(device)

            outputs = model(**inputs)
            _,exposure = get_exposure(input_txt,secret,outputs,nums_encode,TOTAL_CANDIDATES)
            if exposure:
                if exposure > args.threshold:
                    memorized_text.append(input_txt)
                    exposure_list.append(str(exposure))
                    exp_sum += exposure
                    count += 1
            else:
                pass
        logger.info(f"***** average exp: { exp_sum/count } *****")

        # 用于存储去重后的文本和评分的字典
        unique_dict = {}

        # 遍历列表A和B，去重
        logger.info(f"***** memorized sample number: { len(memorized_text) } *****")
        for text, score in zip(memorized_text, exposure_list):
            if text not in unique_dict:
                unique_dict[text] = score

        # 将字典转换为元组列表并根据评分排序
        sorted_tuples = sorted(unique_dict.items(), key=lambda item: item[1], reverse=True)

        # 分离文本和评分到两个列表
        unique_list = [item[0] for item in sorted_tuples]
        unique_exposure = [item[1] for item in sorted_tuples]

        output_file = open(f"./data/{args.model_name}/memorized_TEL.txt","w")
        for i, text in enumerate(unique_list):
            output_file.write(text+' ## '+unique_exposure[i]+'\n')
            # print('privacy text: ',i,' ## expouse:',exposure[i][1])
    if args.privacy_kind == 'EMAIL':
        exp_sum = 0
        memorized_text = []
        MRR_list = []
        for privacy in privacys:  
            secret = privacy["privacy"][0]
            prompt = privacy["prompt"]
            input_txt = prompt + ' ' + secret

            # Tokenize 'prompt' and 'secret' separately
            tokens_prompt = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            tokens_secret = tokenizer(secret, return_tensors="pt", padding=True, truncation=True)
            input_ids = torch.cat((tokens_prompt['input_ids'], tokens_secret['input_ids']), dim=1)
            attention_mask = torch.cat((tokens_prompt['attention_mask'], tokens_secret['attention_mask']), dim=1)

            # Move the tensors to the specified device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Construct the inputs dictionary
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

            outputs = model(**inputs)
            MRR = get_MRR(tokens_prompt['input_ids'],tokens_secret['input_ids'],outputs) * 100
            # print(MRR)
            if MRR:
                if MRR > args.threshold:
                    memorized_text.append(input_txt)
                    MRR_list.append(str(MRR))
                    exp_sum += MRR
            else:
                pass
        logger.info(f"***** average exp: { exp_sum/len(MRR_list) } *****")

        # 用于存储去重后的文本和评分的字典
        unique_dict = {}

        # 遍历列表A和B，去重
        logger.info(f"***** memorized sample number: { len(memorized_text) } *****")
        for text, score in zip(memorized_text, MRR_list):
            if text not in unique_dict:
                unique_dict[text] = score

        # 将字典转换为元组列表并根据评分排序
        sorted_tuples = sorted(unique_dict.items(), key=lambda item: item[1], reverse=True)

        # 分离文本和评分到两个列表
        unique_list = [item[0] for item in sorted_tuples]
        unique_MRR = [item[1] for item in sorted_tuples]

        output_file = open(f'./data/{args.model_name}/memorized_EMAIL.txt',"w")
        for i, text in enumerate(unique_list):
            output_file.write(text+' ## '+unique_MRR[i]+'\n')
            # print('privacy text: ',i,' ## expouse:',exposure[i][1])
    if args.privacy_kind == 'MIMIC':
        exp_sum = 0
        memorized_text = []
        MRR_list = []
        for privacy in privacys:  
            secret = privacy["privacy"]
            prompt = privacy["prompt"]
            input_txt = prompt + ' ' + secret

            # Tokenize 'prompt' and 'secret' separately
            tokens_prompt = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            tokens_secret = tokenizer(secret, return_tensors="pt", padding=True, truncation=True)
            input_ids = torch.cat((tokens_prompt['input_ids'], tokens_secret['input_ids']), dim=1)
            attention_mask = torch.cat((tokens_prompt['attention_mask'], tokens_secret['attention_mask']), dim=1)

            # Move the tensors to the specified device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Construct the inputs dictionary
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

            outputs = model(**inputs)
            MRR = get_MRR(tokens_prompt['input_ids'],tokens_secret['input_ids'],outputs) * 100
            
            # print(MRR)
            if MRR:
                if MRR > args.threshold:
                    memorized_text.append(input_txt)
                    MRR_list.append(str(MRR))
                    exp_sum += MRR
            else:
                pass
        logger.info(f"***** average exp: { exp_sum/len(MRR_list) } *****")

        # 用于存储去重后的文本和评分的字典
        unique_dict = {}

        # 遍历列表A和B，去重
        logger.info(f"***** memorized sample number: { len(memorized_text) } *****")
        for text, score in zip(memorized_text, MRR_list):
            if text not in unique_dict:
                unique_dict[text] = score

        # 将字典转换为元组列表并根据评分排序
        sorted_tuples = sorted(unique_dict.items(), key=lambda item: item[1], reverse=True)

        # 分离文本和评分到两个列表
        unique_list = [item[0] for item in sorted_tuples]
        unique_MRR = [item[1] for item in sorted_tuples]

        output_file = open(f'./data/{args.model_name}/memorized_MIMIC.txt',"w")
        for i, text in enumerate(unique_list):
            output_file.write(text+' ## '+unique_MRR[i]+'\n')
            # print('privacy text: ',i,' ## expouse:',exposure[i][1])




           

if __name__ == "__main__":
    main()
