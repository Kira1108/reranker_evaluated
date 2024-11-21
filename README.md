# ReRank test

当你想说服别人，Rerank的效果很好，部署一个rerank服务的时候，或者当你想更换一个新的rerank模型的时候，你要battle。这个仓库就是你battle的依据-评估。     
- Documents are splitted into chunks for recall. 
- Chunks are sorted with a single rerank model, retrieving the top 10 chunks.
- The top 10 chunks are then evaluted via a LLM classifier. 
- Finally the evaluation scores are computed for each query against their
top 10 recall results

1. Chunking size distribution
![ChunksizeDistribution](images/chunk_size_distribution.png)

2. Evaluation recall quality
```bash
python main.py
```

3. View results at
`./eval_runs/eval_results.json`

```json
[
    {....
    "document_scores": [1,1,1,1,1,1,1,1,1,0],
    "accuracy_3": 1,
    "accuracy_5": 1,
    "mrr_10": 1.0,
    "ndgc_10": 4.254494511770457}
    ...
]
```