from splitters import langchain_recursive_chinese_split
from recall_eval import evalute_recall_batch
from model import Reranker
from pathlib import Path
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import json

text:str = (Path.cwd() / 'data' / 'data.txt').open('r').read()
image_save_path = Path('images')
eval_save_path = Path('eval_runs')

def plot_output_chunk_sizes(text:str) -> None:
    """Plot the distribution of output chunk sizes for different input chunk sizes."""
    chunk_sizes = [80, 120, 150, 180]
    fig, axes = plt.subplots(2, 2, figsize=(14, 5))

    for ax, chunk_size in zip(axes.flatten(), chunk_sizes):
        text_pieces = langchain_recursive_chinese_split(text, chunk_size=chunk_size)
        ax.hist([len(piece) for piece in text_pieces], bins=20, edgecolor='white', linewidth=1)
        ax.grid(linestyle='--')
        ax.set_title(f"Chunk length distribution (input size = {chunk_size})", fontsize=10)
        ax.axvline(x=chunk_size, color='#8B0000', linestyle='--', label='Chunk size')

    plt.tight_layout()
    fp = image_save_path/ 'chunk_size_distribution.png'
    plt.savefig(fp, dpi=300)
    print(f"Chunking experiment image saved at {fp}")

def recall_fn(query:str, topn:int = 10) -> List[str]:
    """If used for evaluation, recall_fn has to return more than 10 chunks such that mrr@10 and ndcg@10 can be computed."""
    reranker = Reranker()
    chunks = langchain_recursive_chinese_split(text)
    scores = np.array(reranker.rerank(query, chunks))
    sorted_chunks = np.array(chunks)[np.argsort(scores)[::-1]][:topn].tolist()
    return sorted_chunks
    
def show_recalls(query:str):
    sorted_chunks = recall_fn(query)    
    print("Recall from text file: ")
    for idx, c in enumerate(sorted_chunks):
        print(f"Recall {idx+1}: {c[:50]}... ...")
 
if __name__ == '__main__':
    # plot_output_chunk_sizes(text)
    # show_recalls("钢琴的起源是怎样的？")
    scores = evalute_recall_batch(
        queries = ["钢琴的起源是怎样的？", "股票的分类是怎样的？", "介绍一下足球这项运动？"],
        recall_fn=partial(recall_fn, topn=10)
    )
    
    with open(eval_save_path / 'eval_results.json', 'w') as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
        
    print("Evaluation resutls saved to ", str(eval_save_path / 'eval_results.json'))