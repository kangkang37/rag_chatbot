'''
Use Gradio interface, Chroma database, Google Gemini API and rag(Retrieval augmented generation) method to achieve Domain-specific information retrieval.

-- 5/27/2024
-- Author: Kangyi Qiu

this is a RAG project using Gradio to achieve web application, and using Chroma to build vector database.
it achieves:
* web chatbot based on Gadio.
* using Google Api key, the Gemini model as the LLM
* using Chroma database, upload pdf file and stored in chroma.collection
* input question, find the closest chunks from Chroma, and using Gemini to answer this question based on the chunks.

we achieve:
Implement user-interactive Gradio-based web interface
You can upload pdf or text,
Call Google's large language model to implement intelligent robot question answering
Use chroma vector database to store incoming pdf data and form your own vector database
Every time a question is entered, the chunks closest to the question can be found based on the existing vector database data, and then used as references in turn to output the answer through the large model.

'''

import random
import gradio as gr
import time
from pypdf import PdfReader
import google.generativeai as genai

from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction,DefaultEmbeddingFunction


# The following code implements multiple pdf uploading, reading pdf content and outputting.
def count_files(message, history):
    text_message = message['text']  # get text from message
    file_paths = message['files']
    num_files = len(file_paths)

    # yield f"You uploaded {num_files} files, the message is: {message}." ##
    # time.sleep(1)
    if not chroma_collection.get()['ids'] and not message['files']:  # if No file was uploaded, and Chroma does not store the previously uploaded PDF
        # yield f'there are no data in chroma collection. \n {chroma_collection.get()}, and no upload pdf!'
        yield f'Please upload the relevant pdfs.'
        return

    # read the first file
    # if upload file from message:
    if message['files']:
        pdf_path = file_paths[0]
        reader = PdfReader(pdf_path)
        pdf_texts = [p.extract_text().strip() for p in
                     reader.pages]  # length of pdf_texts is the number of pages of upload pdf.
        pdf_texts = [text for text in pdf_texts if text]
        # yield f"the pdf text length: {len(pdf_texts)}, '\n', the content of pdf_texts[0]: {pdf_texts[0][:100]}"
        # time.sleep(1)

        # text splitter
        character_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=1000,
            chunk_overlap=0
        )
        character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))
        # print(f"Character split into {len(character_split_texts)} chunks.")

        token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
        token_split_texts = [token_split for text in character_split_texts for token_split in
                             token_splitter.split_text(text)]
        # print(f"Token split into {len(token_split_texts)} chunks.")

        max_id = max(map(int, chroma_collection.get()['ids']), default=0)
        ids = [str(i) for i in range(max_id + 1, max_id + 1 + len(token_split_texts))]
        chroma_collection.add(ids=ids, documents=token_split_texts)
        # print(f"Added {len(ids)} documents to Chroma collection.")

    ids_str = str(chroma_collection.get()['ids'])

    query = text_message

    results = chroma_collection.query(query_texts=[query], n_results=5)
    retrieved_documents = results['documents'][0]

    information = "\n\n".join(retrieved_documents)
    # information = pdf_texts

    prompt = [
        {
            "role": "model",
            "parts": "answer the question based on the information."
        },
        {"role": "user", "parts": f"Question: {query}. \n Information: {information}"}
    ]
    response = model.generate_content(prompt)
    contents = response.text
    # yield f"the ids_str: {ids_str},\n the query is: {query}, \n the information is: {information}, \n the content: {contents}"
    # time.sleep(3)
    yield contents


def create_collections(coll_name):

    default_emb_func=DefaultEmbeddingFunction()
    embedding_function = SentenceTransformerEmbeddingFunction()
    chroma_client = chromadb.Client()
    collections = chroma_client.list_collections()
    collection_names = [collection.name for collection in collections]

    if coll_name not in collection_names:
        chroma_coll = chroma_client.create_collection(coll_name, embedding_function=embedding_function)
        # print(f"Collection {coll_name} created.")
    else:
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

    model = genai.GenerativeModel(model_name="gemini-pro")

    demo = gr.ChatInterface(fn=count_files, examples=[{"text": "Hello", "files": []}], title="RAG Chat Bot",
                            multimodal=True)

    demo.launch()


