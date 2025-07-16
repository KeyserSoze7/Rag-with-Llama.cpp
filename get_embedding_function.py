# from langchain_community.embeddings.ollama import OllamaEmbeddings
#from langchain_ollama import OllamaEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings

#from langchain_community.embeddings.bedrock import BedrockEmbeddings


# def get_embedding_function():
#     # embeddings = BedrockEmbeddings(
#     #     credentials_profile_name="default", region_name="us-east-1"
#     # )

#     #embeddings = OllamaEmbeddings(model="nomic-embed-text")
#     embeddings = SentenceTransformer("BAAI/bge-small-en-v1.5")
#     return embeddings


def get_embedding_function():
    model_name = "BAAI/bge-small-en-v1.5"
    return HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={"normalize_embeddings": True})
