import json
import math
from typing import Callable, List

from nice_car.llms import Qwen2
from retry import retry
from tqdm import tqdm


@retry(tries=3, delay=0.5)
def call_llm(prompt:str) -> str:
    """调用大模型"""
    qwen = Qwen2()
    return qwen(prompt)

def classify_recall_single(query:str, doc:str) -> int:
    """使用大模型判断单条召回结果是否准确"""
    
    prompt = """
    Given a user query and a search result against a knowledge base.
    Determine if the search result is helpful to answer the user query.
    
    if the search result is helpful to answer user query or is relevant to user query, result = 1
    if the search result if unhelpful to answer user query or is unrelevant to user query, result = 0
    
    ! Note that you should output a valid json object without any explaination.
    
    Example Output 
    {{
        "result": 1
    }}
    
    User Query: {query}
    Search Result: {doc}
    
    Output:
    """

    try:
        prompt = prompt.format(
            query = query,
            doc = doc
        )
        
        result = call_llm(prompt).strip("```").strip("json")
        result = json.loads(result)
    
        return result['result']
    except Exception as e:
        print("Unable to parse the output from Qwen.")
        print(e)
        return 0

def classify_recall(query:str, docs:List[str]) -> List[int]:
    """使用大模型判断多条召回结果是否准确"""
    return [classify_recall_single(query, d) for d in docs]

def pad_zero(l:List[int], n = 10):
    """召回条数不够的时候需要pad 0"""
    return l + [0] * (n - len(l))


def acc_n(scores:List[int], n:int = 3):
    """Compute acc@n score"""
    if len(scores) < n:
        scores = pad_zero(scores, n = n)
    return int(any(scores[:n]))


def mrr_n(scores:List[int], n:int = 10):
    """Compute mrr@n score"""
    if len(scores) < n:
        scores = pad_zero(scores, n = n) 
    for i, s in enumerate(scores):
        if s > 0:
            return 1 / (i + 1)
    return 0

def ndgc(scores:List[int], n:int = 10):
    """Compute ndgc score"""
    if len(scores) < n:
        scores = pad_zero(scores, n = n)
    return sum([s / math.log2(i+2) for i, s in enumerate(scores)])

def evalute_recall_single(
    query:str, 
    recall_fn:Callable[[str], List[str]]) -> dict:
    
    print("Evaluating query: ", query)

    documents = recall_fn(query)
    scores = classify_recall(query, documents)

    accuracy_3 = acc_n(scores)
    accuracy_5 = acc_n(scores, n = 5)
    mrr_10 = mrr_n(scores, n = 10)
    ndgc_10 = ndgc(scores, n = 10)
    
    return {
        "query": query,
        "documents:": documents,
        "document_scores":scores,
        "accuracy_3": accuracy_3,
        "accuracy_5": accuracy_5,
        "mrr_10": mrr_10,
        "ndgc_10": ndgc_10
    }
    
def evalute_recall_batch(
    queries:List[str], 
    recall_fn:Callable[[str], List[str]]) -> List[dict]:
    return [evalute_recall_single(q, recall_fn) for q in tqdm(queries)]