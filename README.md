# RAG project

## env
```
conda create --name rag python=3.10.12
conda install jupyter
Pip install panel==1.3.8
pip install langchain
pip install -U langchain-community
Pip install chromed
Pip install google.generativeai
pip install sentence_transformers
Pip install pypdf
```

## run
note that you need to enter the Google Gemini API to run the code. 

```
python rag_basic.py
```

```
python rag_query_expansion.py
```

use colab or jupyter notebook to run the rag_adaptor.ipynb.  


## introduce the project 
there are three main files in this project: 
### 1. rag_basic.py  
this is a basic RAG project using Gradio to achieve web application, and using Chroma to build vector database.
it achieves:
* web chatbot based on Gadio.
* using Google Api key, the Gemini model as the LLM
* using Chroma database, upload pdf file and stored in chroma.collection
* input question, find the closest chunks from Chroma, and using Gemini to answer this question based on the chunks.

### 2. rag_query_expansion.py
based on the rag_basic code, I add query expansion method into it.  
basic functions:  
- Implement a user-interactive chatbot based on a web page, and implement basic functions such as
  file uploading, dialogue question and answer, PDF file segmentation and storage in the Chroma vector database.

**Query expansion:**   

there are two ways to achieve query expansion   

  a) generating hypothetical answers based on the original query using LLM,  
      `jointed query = original_query + hypothetical_answer`  
      push the jointed_query into the LLM with relative information from Chroma, and getting a more comprehensive answer.  
  
  b) Generate other questions related to this original_query,  
      input a series of questions to Chroma to find related chunks, and then generate answers based on these chunks and query.

### 3. rag_adaptor.ipynb
Based on query expansion, an embedding adapter method is also added.  

**Adding Embedding Adaptor**:  
  An adaptor matrix was trained to make the retrieval of relevant knowledge more accurate.
