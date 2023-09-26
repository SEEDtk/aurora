from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader

# glob lists the files to import, loader_cls indicates this is text, not PDF
loader = DirectoryLoader('./', glob="./txt/*.txt", loader_cls=TextLoader)
# loader = TextLoader('./one_file.txt')
# actually load it
documents = loader.load()
# Split into 1000-byte chunks with 100-char overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# split the documents into the chunks
texts = text_splitter.split_documents(documents)
# embeddings are floating-point vectors used by the OpenAI; there are many formats.  This class ensures it is the format for GPT
embeddings = OpenAIEmbeddings()
# Create a vector database from our text.  It will be saved in the named directory for later loading
vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embeddings,
                                 persist_directory="SAVED_CHROMA_DB")
# Here we put in the query
query = "What did Abraham Lincoln say our fathers had brought forth on this continent?"

## first, prove we can obtain the relevant docs. This section tests the database without using the OpenAI model.
# The retriever looks for documents relevant to the query, returns the 4 most relevant documents.
retriever = vectordb.as_retriever()  # k docs to return; default 4
docs = retriever.get_relevant_documents(query)
print("RELEVANT DOCS:")
for (i,doc) in enumerate(docs):
    print("relevant doc --------",i)
    print(doc)
print("-" * 70)

# next, use a chain to retrieve and process.. This is another sanity test.  Here we want only 2 documents,
# hence the k=2
retriever = vectordb.as_retriever(search_kwargs={"k": 2}) # k override 4 with 2
print("SEARCH TYPE",retriever.search_type,"SEARCH KWARGS",retriever.search_kwargs)
print("-" * 70)
# Here we ask for GPT-4.  Temp is 0 to 1.  Higher temperatures use more randomizing in the search.  0.7
# is commonly used for fooling around.  Max_Tokens limits the number of output tokens, reducing the size
# of the output.  There are usually around 1.7 tokens per word, but longer words have more.  You get
# charged per token. $$$$$ <----
llm = ChatOpenAI(model_name="gpt-4", temperature=0.0, max_tokens = 128,)

# load_qa_chain uses all texts and accepts multiple documents;
# RetrievalQA uses load_qa_chain under the hood but retrieves relevant text chunks first
# chain_types are: stuff, map-reduce, refine, etc; stuff means all in at once which may
#  exceed token limit, so may have to use another type for queries with large context.
# Note we still have k=2 on the retriever.  Remember, a document is one of the chunk things
# above.
qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type="stuff", # stuff all in at once
                                       retriever=retriever,
                                       return_source_documents=True)

llm_response = qa_chain(query)
print("LLM_RESPONSE")
print(llm_response)
print("\nPROCESSED OUTPUT (including sources)")
print("RESULT",llm_response['result'])
print('\nSOURCES:')
for source in llm_response["source_documents"]:
    print(source.metadata['source'])
