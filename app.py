from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Ler o texto
with open("texto.txt", "r", encoding="utf-8") as arquivo:
    texto = arquivo.read()

# Dividir em chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=100,
    chunk_overlap=20
)

chunks = text_splitter.split_text(texto)

print("Chunks criados:")
print(chunks)

# Criar embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Criar banco vetorial FAISS
db = FAISS.from_texts(chunks, embeddings)

print("\nBanco vetorial criado com sucesso!")

# Fazer perguntas
while True:
    pergunta = input("\nDigite sua pergunta: ")

    if pergunta.lower() == "sair":
        break

    docs = db.similarity_search(pergunta)

    print("\nResposta encontrada:\n")
    print(docs[0].page_content)