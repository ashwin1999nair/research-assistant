from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

## Text Splitting
splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

## Embedding Model
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## Chrome DB Client
chroma_client=chromadb.Client()

def store_chunks(raw_texts: list, topic: str)-> chromadb.Collection:
    """Takes raw texts, chunks them, embeds them, stores in ChromaDB. Returns the collection for querying."""

    all_chunks=[]
    for text in raw_texts:
        chunks=splitter.split_text(text)
        all_chunks.extend(chunks)

    ## Get embeddings for all chunks
    embedded_chunks=embeddings.embed_documents(all_chunks)

    ## Create a fresh collection for this topic
    collection_name="research"
    try:
        chroma_client.delete_collection(collection_name)

    except:
        pass

    collection=chroma_client.create_collection(collection_name)

    ## Store chunks with their embeddings
    collection.add(documents= all_chunks, embeddings=embedded_chunks, ids=[str(i) for i in range (len(all_chunks))])

    return collection

def query_chunks(collection: chromadb.Collection, topic: str, n_results: int=10) ->list:
    """Queries ChromaDB for most relevant chunks.Returns list of relevant text chunks."""
    query_embedding = embeddings.embed_query(topic)
    results=collection.query(query_embeddings=[query_embedding], n_results=n_results)

    return results["documents"][0]


