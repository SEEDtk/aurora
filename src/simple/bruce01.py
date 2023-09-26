from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

loader = TextLoader("./83332.12.txt")   ### just the one file

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    separators = ["\n"],
    keep_separator = False,
    chunk_size = 0,    # just splits on lines (separators)
    chunk_overlap  = 0,
    length_function = len,
    is_separator_regex = False,
    # add_start_index = True,
)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=texts, embedding=embeddings,) # no persist

query = "Which genome has ID 83332.12 ?"  # Mycobacterium tuberculosis H37Rv

retriever = vectordb.as_retriever(search_kwargs={"k": 2}) # k override 4 with 2

llm = ChatOpenAI(model_name="gpt-4", temperature=0.0, max_tokens = 128,)
# llm = ChatOpenAI(model_name="gpt-3.5-turbo")  # cheaper model should work fine

qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type="stuff", # stuff all in at once
                                       retriever=retriever,
                                       # return_source_documents=True) # we know :-)
                                      )

llm_response = qa_chain(query)
print("RESPONSE")
print(llm_response['result'])
