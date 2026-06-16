from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSequence, RunnableLambda
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ── Schema ────────────────────────────────────────────────────────────────────
class AnswerSchema(BaseModel):
    question: str       = Field(description="The question asked by the user")
    pointers: list[str] = Field(description="Exactly 5 answer pointers based on the video context, or politely say I dont know if not found in context")

# ── Parser ────────────────────────────────────────────────────────────────────
parser = PydanticOutputParser(pydantic_object=AnswerSchema)

# ── Prompt ────────────────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer strictly from the video context only. Return valid JSON only."),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nGive exactly 5 pointers.\n{format_instructions}")
]).partial(format_instructions=parser.get_format_instructions())

# ── Model & Embeddings ────────────────────────────────────────────────────────
model      = ChatGoogleGenerativeAI(model="gemini-2.5-flash", max_tokens=2000)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")

# ── Input ─────────────────────────────────────────────────────────────────────
video_id = "6W92_t9FveA"
question = "Who is speaking in this video what is the channel name and topic?"

# ── Step 1: Load Transcript ───────────────────────────────────────────────────
print("Loading transcript...")
ytt             = YouTubeTranscriptApi()
transcript_list = ytt.fetch(video_id, languages=["hi"])
transcript      = " ".join([entry.text for entry in transcript_list])

# ── Step 2: Split ─────────────────────────────────────────────────────────────
print("Splitting transcript...")
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).create_documents([transcript])

# ── Step 3: Embed & Store ─────────────────────────────────────────────────────
print("Embedding and storing in FAISS...")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever   = vectorstore.as_retriever(search_kwargs={"k": 4})

# ── Step 4: Concat Docs ───────────────────────────────────────────────────────
def concat_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# ── Step 5: Parallel Chain ────────────────────────────────────────────────────
parallel_chain = RunnableParallel(
    question = RunnablePassthrough(),
    context  = retriever | RunnableLambda(concat_docs)
)

# ── Step 6: Answer Chain ──────────────────────────────────────────────────────
answer_chain = RunnableSequence(prompt, model, parser)

# ── Step 7: Run ───────────────────────────────────────────────────────────────
print("Running RAG pipeline...\n")
parallel_result = parallel_chain.invoke(question)

result = answer_chain.invoke({
    "question" : parallel_result["question"],
    "context"  : parallel_result["context"]
})

# ── Print ─────────────────────────────────────────────────────────────────────
print("=" * 55)
print(f"  QUESTION: {result.question}")
print("=" * 55)
print("\n 5 POINTERS:")
for i, p in enumerate(result.pointers, 1):
    print(f"  {i}. {p}")
