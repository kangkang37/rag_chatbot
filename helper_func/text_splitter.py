
# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,SentenceTransformersTokenTextSplitter
from transformers import AutoTokenizer


'''
achieve text splitter in two ways: 
1. RecursiveCharacterTextSplitter,Use len and token as length_function respectively,
use split_text and split_docuements functions respectively,

2. SentenceTransformersTokenTextSplitter,

To use split_text, you need to convert the PDF into a string after reading it;
Docuemtns requires the PDF to be a list after reading it, and each element is in the form of <class 'langchain_core.documents.base.Document'>.

'''

def text_splitter_len_text(file_path):
    ##### use split_text
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    pages_texts=[page.page_content for page in pages]
    print(f'type of pages_texts is: {type(pages_texts)}') # <class 'list'>
    print(f'type of pages_texts[0] is: {type(pages_texts[0])}') # str
    print(f'the second pages_texts is: {pages_texts[1]}')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=10,
        length_function=len,  # use the number of characters to compute the length of chunks
        add_start_index=True,
    )
    pages_texts_str='\n\n'.join(pages_texts)
    texts = text_splitter.split_text(pages_texts_str)
    return texts

def text_splitter_len_documents(file_path):

    #### use split_documents
    # Use the PyPDFLoader to load and parse the PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    print(f'type of pages is: {type(pages)}') # <class 'list'>
    print(f'type of pages[0] is: {type(pages[0])}') # <class 'langchain_core.documents.base.Document'>
    print(f'the second pages is: {pages[1]}')

    print(f'Loaded {len(pages)} pages from the PDF')

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap  = 10,
        length_function = len, # 用字符计算chunk长度
        add_start_index = True,
    )

    texts = text_splitter.split_documents(pages)

    return texts


# use token to compute the length of chunks
def text_splitter_token(file_path):
    # Use the PyPDFLoader to load and parse the PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    print(f'type of pages is: {type(pages)}')
    print(f'Loaded {len(pages)} pages from the PDF')

    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    def tokens(text: str) -> int:
        return len(tokenizer.encode(text))


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap  = 10,
        length_function = tokens, # 用字符计算chunk长度
        add_start_index = True,
    )

    texts = text_splitter.split_documents(pages)
    return texts

def token_splitter(file_path):
    # Use the PyPDFLoader to load and parse the PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    print(f'Loaded {len(pages)} pages from the PDF')
    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)

    token_split_texts = token_splitter.split_documents(pages)
    # texts = text_splitter.split_documents(pages)
    return token_split_texts

if __name__=="__main__":
    pdf_path="./pdf_files/resume.pdf" # input your own pdf file
    texts = text_splitter_len_text(pdf_path)

    # texts=text_splitter_len_documents(pdf_path)
    # texts = text_splitter_token(pdf_path)
    # texts = token_splitter(pdf_path)
    print(f'Split the pages in {len(texts)} chunks')
    print('--'*10)
    print(texts[0])
    # print('--'*10)
    # print(texts[0].page_content)
    # print('--'*10)
    # print(texts[0].metadata)
    # print('--'*10)
    # print(texts[0].metadata.keys())
    # print('--'*10)

    print(type(texts)) # list
    print(type(texts[0])) # 'langchain_core.documents.base.Document'
