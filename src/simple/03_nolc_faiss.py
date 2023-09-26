import os, time
import numpy as np
import openai

import faiss   # actually using faiss-cpu on ash
# from HF docs:  faiss is a library for dense retrieval. It means that it
#   retrieves documents based on their vector representations, by doing a
#   nearest neighbors search

## make sure your env has OPENAI_API_KEY
## This finds the articles on disk, splits then into word-chunks, and builds passages
text_dir = "./txt"
articles = []
for filename in os.listdir(text_dir):
    with open(text_dir + "/" + filename) as f:
        content = ""
        lines = f.readlines()
        for line in lines:
            line = line.strip() + " "
            content += line
        articles.append(content)
# note no overlapping this time
passages = []
num_words_per_passage = 1000
for artidx in range(len(articles)):
    article = articles[artidx]
    words = article.split()
    for i in range(0, len(words), num_words_per_passage):
        passage = " ".join(words[i:i+num_words_per_passage])
        passages.append(passage)

# --------
# The embed model determines how to convert text to floating-point vectors.  This one is old,
# but is generally the one used by langchain.  The query model is the LLM.
EMBED_MODEL = "text-embedding-ada-002"  # typically used for embeddings
QUERY_MODEL = "gpt-4"
# Here we map the passage text to the vectors created.  This is used for debugging
embeds_all = []
embed2passage = {}
for passage in passages:
    # res is a big data structure with extra stuff.  We extract just the embedding
    res = openai.Embedding.create(input=passage, engine=EMBED_MODEL)
    # The key to a dictionary has to be a tuple (immutable) not a list (mutable)
    embed = tuple( res['data'][0]['embedding'] )
    embeds_all.append(embed)
embed2passage = { embed : passages[i]  for (i,embed) in enumerate(embeds_all) }

## Here we want to set up similarity searches.  In this case we use inner product to
## determine what is close.
m = 32  # num neighbors to each vertex
dim = len(embeds_all[0])
# Create the vector store
faiss_index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
for tuple_embed in embeds_all:
    # Convert embedding from a tuple to a 2-dimensional array with 1 row and N columns
    # THis is for FAISS compatibility.
    array_embed = np.array(tuple_embed).reshape( (1,len(tuple_embed)) )
    # print("ARRAYEMBED_SHAPE",array_embed.shape)  # (1,1536)
    # Add embedding to a vector store.  As they are added, FAISS connects each one to its similar neighbors
    faiss_index.add( array_embed )
print("LENPASSAGES",len(passages),"LENEMBEDSALL",len(embeds_all))

stime = time.time()

query = "What did Lincoln say had been brought forth on this continent?"
# Ask openai (this is a static method) to create an embedding for the query.
# Above we created embeddings for the documents.
res = openai.Embedding.create (input=[query], engine=EMBED_MODEL)
q_embed = res['data'][0]['embedding']
q_embed = np.array(q_embed).reshape( (1,len(q_embed)) )
print("QEMBED_SHAPE",q_embed.shape)  # (1,1536) <-- 1536 is the OpenAI norm

# find the k=2 most similar passages to the query embeds
(D,I) = faiss_index.search(q_embed, k=2)

# print the indexes and simlarity scores
print ('Closest matching indexes:', I)
print ('Inner Products:', D)

# use the closest passage text as context for the query.  (I) was built above from the index search
context = ""
for i in I[0]:
    embed = embeds_all[i]
    passage = embed2passage[ tuple(embed) ]
    context += passage + " "
# The messages are the parameters to the chat completion.  What does the query (user) mean
# in the context of the closest passages (assistant).
msgs = [ {"role": "system",    "content": ""},
         {"role": "assistant", "content": context},
         {"role": "user",      "content": query} ]
# ChatCompletion means we want an answer to the query.  We are not having a conversation, just asking for a response.
# Generally, assistant stuff and system stuff are true, but system is more global.  Note that we don't use the vectors.
# The vectors are for searching, not for the LLM.
response = openai.ChatCompletion.create(
    model=QUERY_MODEL,
    temperature=0.0,
    max_tokens=100,
    messages=msgs,
)

print("--------------------------------")
print(f"Query: {query}")
print("--------------------------------")
print("RESPONSE")
print(response['choices'][0]['message']['content'])
print(f"time local index and remote query {time.time()-stime:0.2f}")
