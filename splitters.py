from functools import partial
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

def langchain_recursive_split(text:str, *args, **kwargs) -> List[str]:
    
    """
    Splits the given text into a list of strings using the RecursiveCharacterTextSplitter from langchain_text_splitters.
    Args:
        text (str): The text to be split.
        *args: Additional positional arguments to be passed to RecursiveCharacterTextSplitter.
        **kwargs: Additional keyword arguments to be passed to RecursiveCharacterTextSplitter.
    Returns:
        List[str]: A list of strings resulting from the split operation.
    """
    
    splitter = RecursiveCharacterTextSplitter(
        *args, **kwargs
    )

    return splitter.split_text(text)

def langchain_recursive_chinese_split(text:str, chunk_size:int = 250) -> List[str]:
    
    """
    Splits the given Chinese text into chunks using a recursive splitting function.
    This function uses a partial application of `langchain_recursive_split` with specific
    separators and parameters tailored for Chinese text. The text is split based on 
    various punctuation marks and newlines, ensuring that each chunk is of a specified 
    size and overlap.
    Args:
        text (str): The Chinese text to be split.
    Returns:
        List[str]: A list of text chunks.
    """
    
    split_fn = partial(
        langchain_recursive_split, 
        separators=["\n\n", "\n", "。","!","？","！","；",";"],
        length_function=len,
        chunk_size=chunk_size,
        chunk_overlap=0,
        keep_separator=True
    )
    
    return split_fn(text)