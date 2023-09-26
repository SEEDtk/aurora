from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain

filename = "txt/gettysburg.txt"  # sys.argv[1]

llm = ChatOpenAI(model_name="gpt-4", temperature=0.0, max_tokens = 128,)

embeddings = OpenAIEmbeddings()

# if split by hand, we can create a list of metadata dicts giving the source
#   for each chunk, e.g.:   [ { "source": "name_of_source_doc" }, ... ]
# This is the "metadatas" described below.
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 100,
    length_function = len,
)

# Note here we slurp the whole file into a single string and split it once
with open(filename) as textfile:
    text = textfile.read()
chunks = text_splitter.split_text(text)
# Here we are using Meta's vector database instead of CHROMA, just for fun
indexer = FAISS.from_texts(chunks,embeddings)  # metadatas can be 3rd arg

# load_qa_chain uses all texts and accepts multiple documents;
# RetrievalQA uses load_qa_chain under the hood but retrieves relevant text chunks first
# chain_types are: stuff, map-reduce, refine, etc; stuff means all in at once which may
#  exceed token limit, so may have to use another type for queries with large context
# Note that we aren't using a retriever here, we are just creating a model-only chain
chain = load_qa_chain(llm) #  default chain_type="stuff" (all in at once)

query = "What did Lincoln say had been brought forth on this continent?"
# This indexer thing does the retrieval step that RetrievalQA did for us.
docs = indexer.similarity_search(query,k=3)
# Now we ask for a response using the llm-only chain, giving it just the relevant documents
# computed by the indexer.
response = chain.run(input_documents=docs,question=query)
print(response)
