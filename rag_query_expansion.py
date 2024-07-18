'''
-- Date: 5/28/2024
-- Author: Kangyi Qiu

Use Gradio interface, Chroma database, Google Gemini API and rag(Retrieval augmented generation) method to achieve Domain-specific information retrieval.
Add Query Expansion Method into this code.

we achieved:
basic functions:
    Implement a user-interactive chatbot based on a web page, and implement basic functions such as
    file uploading, dialogue question and answer, PDF file segmentation and storage in the Chroma vector database.

Query expansion: there are two ways to achieve query expansion
1, Including generating hypothetical answers based on the original query using LLM,
    jointed query = original_query + hypothetical_answer,
    push the jointed_query into the LLM with relative information from Chroma, and getting a more comprehensive answer;
2, Generate other questions related to this original_query,
    input a series of questions to Chroma to find related chunks, and then generate answers based on these chunks and query.

refer: https://learn.deeplearning.ai/courses/advanced-retrieval-for-ai/lesson/4/query-expansion
paper: https://arxiv.org/abs/2305.03653
'''

import random
import gradio as gr
import time
from pypdf import PdfReader
import google.generativeai as genai

from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


def helper_read_pdf(pdf_path):
    '''
    pdf_path is the path of upload file,
    get text information from this pdf,
    use text_splitter to split the informatio into some chunks,
    return list, list[i] is string, the i-th chunk string.

    :param pdf_path:
    :return:
    '''
    reader = PdfReader(pdf_path)
    pdf_texts = [p.extract_text().strip() for p in
                 reader.pages]  # length of pdf_texts is the number of pages of upload pdf.
    pdf_texts = [text for text in pdf_texts if text]

    # # test
    # print(f"the pdf text length: {len(pdf_texts)}, '\n', the content of pdf_texts[0]: {pdf_texts[0][:100]}")
    # time.sleep(1)

    # text splitter
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=0
    )
    character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))
    # print(f"Character split into {len(character_split_texts)} chunks.") ##

    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
    token_split_texts = [token_split for text in character_split_texts for token_split in
                         token_splitter.split_text(text)]
    # print(f"Token split into {len(token_split_texts)} chunks.") ##

    # print('the following are the chunks from this upload pdf: ') ##
    # for i in range(len(token_split_texts)):
    #     print(token_split_texts[i])
    #     print('\n')

    return token_split_texts


def augment_example_answer(query):
    '''
    query expansion method 1 helper function
    :param query: string, input original query
    :return: string, get hypothetical answer from this query, using Gemini LLM,
    '''
    messages = [
        {
            "role": "model",
            "parts": "I have gived you a query about resume pdf files. Provide an example answer to the given question, that might be found in a document like a resume. "
        },
        {"role": "user", "parts": query}
    ]

    response = model.generate_content(messages)
    contents = response.text
    # print('contents from augment_example_answer: \n ', contents)
    return contents


def augment_multiple_query(query):
    '''
    query helper function, method 2.
    :param query: string, input original query
    :return: contents is a list. contents[i] is string, the related question string from this query.
    '''
    messages = [
        {
            "role": "model",
            "parts": "You are an experienced recruiter. Your users are asking questions about resumes. "
                     "Suggest up to five additional related questions to help them find the information they need, for the provided question. "
                     "Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic."
                     "Make sure they are complete questions, and that they are related to the original question."
                     "Output one question per line. Do not number the questions."
        },
        {"role": "user", "parts": query}
    ]

    response = model.generate_content(messages)
    contents = response.text
    # print('contents from augment_multiple_query: ', contents)
    contents = contents.split("\n")  # return list
    return contents



def count_files(message, history):
    """
    the main func of gr.ChatInterface(fn=count_files)
    :param message: dict.
        message={
            'text': #this is input query string#,
            'files': # this is list, store upload file paths string#
        }
    :param history: 2d list, history[i] is the i-th question-answer,
            history[i][0] is the user's input query, history[i][1] is the answer string from system.
        e.g. [
            ['hi', 'I cannot answer the question as the provided context does not contain any relevant information.'],
            ['who are you?', 'I am xxx']
            ]
    :return: string, answer about query, will be show in web-based Gradio demo in chatbot box.
    """
    text_message = message['text']  # get text from message
    file_paths = message['files']
    num_files = len(file_paths)

    # yield f"You uploaded {num_files} files, the message is: {message}." ##
    # time.sleep(1)
    if not chroma_collection.get()['ids'] and not message['files']:
        # yield f'there are no data in chroma collection. \n {chroma_collection.get()}, and no upload pdf!'
        yield f'Please upload the relevant pdfs.'
        return

    # print('the ids of chroma_collection: ', chroma_collection.get()['ids'])

    # read the first file
    # if upload file from message:
    if message['files']:
        pdf_path = file_paths[0]

        token_split_texts = helper_read_pdf(pdf_path)

        max_id = max(map(int, chroma_collection.get()['ids']), default=0)
        ids = [str(i) for i in range(max_id + 1, max_id + 1 + len(token_split_texts))]
        chroma_collection.add(ids=ids, documents=token_split_texts)
        # print(f"Added {len(ids)} chunks to Chroma collection.") ##

    ids_str = str(chroma_collection.get()['ids'])  # get all ids from collection

    original_query = text_message

    # # method 1: get hypothetical answer from the original query
    # hypothetical_answer = augment_example_answer(original_query)  # get example answer
    # # print('\n hypothetical answer: ', hypothetical_answer) ##
    #
    # joint_query = f"{original_query} {hypothetical_answer}"
    # ##

    # method 2: get multi-querys from original query
    augmented_queries = augment_multiple_query(original_query)
    # print('\n multi_querys: ', augmented_queries) ##

    joint_query = [original_query] + augmented_queries
    ##

    results = chroma_collection.query(query_texts=joint_query, n_results=5)
    # print(results)

    retrieved_documents = results['documents'][0]

    information = "\n\n".join(retrieved_documents)

    prompt = [
        {
            "role": "model",
            "parts": "answer the question based on the information."
        },
        {"role": "user", "parts": f"Question: {original_query}. \n Information: {information}"}
    ]
    response = model.generate_content(prompt)
    contents = response.text
    # yield f"the ids_str: {ids_str},\n the original_query is: {original_query}, \n the information is: {information}, \n\n the contents: {contents}"
    # time.sleep(3)
    yield contents


def create_collections(coll_name):
    """
    search the coll_name whether exist in the chroma collections or not.
    if not exist, create a new collection with the name=coll_name;
    if does exist, get the collection from chroma client.
    :param coll_name: string, the collection name we are searching in the chroma.collections
    :return: chroma collection
    """
    embedding_function = SentenceTransformerEmbeddingFunction()
    chroma_client = chromadb.Client()

    collections = chroma_client.list_collections()
    collection_names = [collection.name for collection in collections]

    # if coll_name not exist in the collection_names
    if coll_name not in collection_names:
        chroma_coll = chroma_client.create_collection(coll_name, embedding_function=embedding_function)
        # print(f"Collection {coll_name} created.")
    else:  # already exists this collection
        chroma_coll = chroma_client.get_collection(coll_name)
        # print(f"Collection {coll_name} already exists.")

    return chroma_coll


if __name__ == "__main__":
    Google_API = input("Please input the Google Gemini API: ")
    # print(Google_API)
    genai.configure(api_key=Google_API)

    # chroma
    collection_name = "test1"
    chroma_collection = create_collections(collection_name)

    # chroma_collection.query()

    model = genai.GenerativeModel(model_name="gemini-pro")

    demo = gr.ChatInterface(fn=count_files, examples=[{"text": "Hello", "files": []}], title="RAG Chat Bot",
                            multimodal=True)

    demo.launch()
