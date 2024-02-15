import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from baukit import Trace, TraceDict
import argparse
import jsonlines
import logging
import numpy as np
import os
import json
from sklearn.decomposition import PCA
from tqdm import tqdm
from utils import model_name2path, get_interveted_output
import torch.nn.functional as F
from functools import partial
import json
import numpy as np
from collections import Counter

file_suffix = {
    'llama7b': 'llama7b',
    'gpt2-small':'small',
    'gpt2-xl':'xl',
    'gpt-neo':'neo',
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_data(data_path):
    with open(data_path,'r') as f:
        line = f.readline().strip()
        attrs = json.loads(line)
    return attrs

def get_pn(res, threshold):
    total_counts = Counter()

    for sublist in res:
        sublist_counts = Counter(set(sublist))
        total_counts.update(sublist_counts)

    threshold_frequency = max(1, threshold * len(res))

    frequent_elements = [element for element, count in total_counts.items() if count > threshold_frequency]

    return frequent_elements

def filter(attr, threshold):
    attr_array = np.array(attr)

    max_abs_value = np.max(np.abs(attr_array))
    threshold = threshold * max_abs_value

    res = []
    for i in range(attr_array.shape[0]):
        for j in range(attr_array.shape[1]):
            temp = {}
            if abs(attr_array[i, j]) > threshold:
                # temp[f"{i}:{j}"] = attr_array[i, j]
                # res.append(temp)
                res.append(f"{i}:{j}")

    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2',
                        choices=['llama7b', 'gpt2', 'gpt2-xl', 'gpt-neo'],
                        )
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--privacy_kind', type=str, default='TEL')
    parser.add_argument('--text_threshold', type=float, default=0.1)
    parser.add_argument('--batch_threshold', type=float, default=0.5)

    args = parser.parse_args()

    attr_scores = read_data(args.data_path)
    logger.info(f"the number of memorized privacy: {len(attr_scores)}")

    all_res = []
    ## 筛选每个sample的神经元
    for attr in attr_scores:
        res = filter(attr['attributuon_scores'], args.text_threshold)
        all_res.append(res)
    
    pn = get_pn(all_res, args.batch_threshold)
    logger.info(f"the number of privacy neurons: {len(pn)}")

    import matplotlib.pyplot as plt


    # Split the data into individual neurons and parse them
    layers, positions = zip(*[map(int, neuron.split(':')) for neuron in pn])

    layers_np = np.array(layers)

    # Determine the unique number of layers to define the y-axis
    unique_layers = np.unique(layers_np)

    # Count the number of neurons in each layer
    neuron_counts_per_layer = {layer: np.sum(layers_np == layer) for layer in unique_layers}

    # Create a bar plot for the number of neurons per layer
    plt.figure(figsize=(10, 6))
    plt.bar(neuron_counts_per_layer.keys(), neuron_counts_per_layer.values(), color='skyblue')
    plt.title('Number of Special Neurons per Layer in a Language Model')
    plt.xlabel('Layer Index')
    plt.ylabel('Number of Special Neurons')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the figure
    plt.savefig(f'{args.model_name}-{args.privacy_kind}-neurons-distribution.png')

    output_dir = f'filtered-pn-{args.model_name}-{args.privacy_kind}.txt'
    with open(os.path.join(output_dir), 'w') as fw:
        for i in pn:
            fw.write(i+'\n')




if __name__ == "__main__":
    main()
