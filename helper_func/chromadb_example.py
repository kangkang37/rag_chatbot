
'''
chromadb
'''


import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,SentenceTransformersTokenTextSplitter


def getDataFromPDF(file_path) -> list:
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        pages_texts = [page.page_content for page in pages]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=10,
            length_function=len,
            add_start_index=True,
        )
        pages_texts_str = '\n\n'.join(pages_texts)
        texts = text_splitter.split_text(pages_texts_str)
        return texts

        # with open('../test_data/data.txt', 'r', encoding="utf-8") as file:
        #     content: str = file.read()
        #     splited_data: list = content.split("\n\n\n")
        #     return splited_data
    except Exception as e:
        print("Read data from text failed : ", e)

def addVectorDataToDb(file_path) -> None:
    embeddings: list = []
    metadatas: list = []
    documents: list = []
    ids: list = []
    splited_data = getDataFromPDF(file_path)
    print(len(splited_data))
    print(splited_data[:3])

    # emb = embedding_model(splited_data) #
    # print(len(emb))
    # print(len(emb[0])) #
    try:
        emb = embedding_model(splited_data)
        print(len(emb))
        print(len(emb[0]))  # emb size 28*384
        for index in range(len(splited_data)):
            embeddings.append(emb[index])
            metadatas.append({"Chapter": str(index+1)})
            documents.append(splited_data[index])
            ids.append(str(index+1))
        collection.add(
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            ids=ids
        )
        print("Data added to collection")
    except Exception as e:
        print("Add data to db failed : ", e)


def searchDataByVector(query: str):
    try:
        query_vector = embedding_model([query])
        print(len(query_vector)) # 1
        print(len(query_vector[0])) # 384
        res = collection.query(
            query_embeddings=query_vector,
            n_results=2,
            include=['distances','embeddings', 'documents', 'metadatas'],
        )
        print("Query", "\n--------------")
        print(query)
        print("Result", "\n--------------")
        print(res['documents']) # 1*n_results
        print("Vector", "\n--------------")
        print(len(res['embeddings']), len(res['embeddings'][0])) # 1*n_results*384
        print("")
        print("Complete Response","\n-------------------------")
        print(res)

    except Exception as e:
        print("Vector search failed : ", e)


if __name__=='__main__':
    file_path= "./pdf_files/resume.pdf"
    embedding_model = SentenceTransformerEmbeddingFunction()
    chroma_client=chromadb.Client()

    # client = chromadb.PersistentClient(path="my_chroma_db_1")
    collection = chroma_client.get_or_create_collection(name="test1", embedding_function=embedding_model)
    print(collection)
    print(collection.get())
    addVectorDataToDb(file_path)
    print(collection.get())

    query = "what is MoCo?"
    searchDataByVector(query=query)



