# import libraries
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

class ChunkPipeline:
    """Returns the chunked documents
    """
    def __init__(self, filepath: str):
        self.filepath = filepath

        with open(filepath, 'r', encoding="utf-8") as file:
            self.text = file.read()

    def clean_text(self):
        """removes the repetitive headers that appears on every page, returns a cleaned string
        """
        sample_text_cleaned = re.sub(r'## Employment Act 1968 2020 Ed.', '', self.text)
        sample_text_cleaned = re.sub(r'## 2020 Ed. Employment Act 1968', '', sample_text_cleaned)

        return sample_text_cleaned
    
    def splitter(self, cleaned_text: str):
        """using the RecursiveCharacterTextSplitter from langchain.text_splitter, we generate the chunks of 
        document by sections 
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            length_function=len,
            separators=['(?<=\n)##', '(?<=\n)###'],
            is_separator_regex=True)

        all_splits = text_splitter.create_documents([cleaned_text])

        return all_splits