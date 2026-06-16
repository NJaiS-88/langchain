from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", max_tokens=2000)

# ── Schema 1: full explanation ──────────────────────────────────────────────
class ExplanationSchema(BaseModel):
    topic: str = Field(description="The topic being explained")
    explanation: str = Field(description="A detailed explanation of the topic")

# ── Schema 2: 5-pointer summary ─────────────────────────────────────────────
class SummarySchema(BaseModel):
    topic: str = Field(description="The topic being summarized")
    pointers: list[str] = Field(description="Exactly 5 short summary bullet points")

# ── Parsers ──────────────────────────────────────────────────────────────────
parser1 = PydanticOutputParser(pydantic_object=ExplanationSchema)
parser2 = PydanticOutputParser(pydantic_object=SummarySchema)

# ── Prompts ──────────────────────────────────────────────────────────────────
prompt1 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Always return valid complete JSON only."),
    ("human", "Explain {topic}.\n{format_instructions}")
]).partial(format_instructions=parser1.get_format_instructions())

prompt2 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Always return valid complete JSON only."),
    ("human", "Summarize this in exactly 5 short pointers:\n{topic}\n{format_instructions}")
]).partial(format_instructions=parser2.get_format_instructions())

# ── Chain 1: get full explanation ─────────────────────────────────────────────
chain1 = prompt1 | model | parser1
explanation_result: ExplanationSchema = chain1.invoke({"topic": "unemployment in India"})

# ── Chain 2: summarize the explanation ───────────────────────────────────────
chain2 = prompt2 | model | parser2
summary_result: SummarySchema = chain2.invoke({"topic": explanation_result.explanation})

# ── Print results ─────────────────────────────────────────────────────────────
print("=" * 50)
print(f"TOPIC: {explanation_result.topic}")
print("=" * 50)

print("\nFULL EXPLANATION:")
print(explanation_result.explanation)

print("\n5-POINT SUMMARY:")
for i, point in enumerate(summary_result.pointers, 1):
    print(f"  {i}. {point}")
