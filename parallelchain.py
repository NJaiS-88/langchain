from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", max_tokens=2000)

# ── Schemas ───────────────────────────────────────────────────────────────────
class ExplanationSchema(BaseModel):
    explanation: str = Field(description="A detailed explanation of the topic")

class SummarySchema(BaseModel):
    pointers: list[str] = Field(description="Exactly 5 short summary bullet points")

class FinalSchema(BaseModel):
    topic: str = Field(description="The topic")
    explanation: str = Field(description="Full explanation")
    pointers: list[str] = Field(description="5 summary bullet points")
    questions: list[str] = Field(description="5 practice questions")

# ── Parsers ───────────────────────────────────────────────────────────────────
parser_exp   = PydanticOutputParser(pydantic_object=ExplanationSchema)
parser_sum   = PydanticOutputParser(pydantic_object=SummarySchema)
parser_final = PydanticOutputParser(pydantic_object=FinalSchema)

# ── Prompts ───────────────────────────────────────────────────────────────────
prompt_exp = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Return valid JSON only."),
    ("human", "Explain {topic} in detail.\n{format_instructions}")
]).partial(format_instructions=parser_exp.get_format_instructions())

prompt_sum = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Return valid JSON only."),
    ("human", "Give exactly 5 bullet point summary about {topic}.\n{format_instructions}")
]).partial(format_instructions=parser_sum.get_format_instructions())

prompt_final = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Return valid JSON only."),
    ("human", """Using the info below, fill all fields including 5 practice questions:

Topic: {topic}
Explanation: {explanation}
Pointers: {pointers}

{format_instructions}""")
]).partial(format_instructions=parser_final.get_format_instructions())

# ── Step 1: Run parallel chains ───────────────────────────────────────────────
parallel_chain = RunnableParallel(
    explanation = prompt_exp | model | parser_exp,
    summary     = prompt_sum | model | parser_sum,
)

topic = "unemployment in India"

parallel_result = parallel_chain.invoke({"topic": topic})

# ── Step 2: Manually merge outputs ────────────────────────────────────────────
explanation_text = parallel_result["explanation"].explanation
pointers_text    = parallel_result["summary"].pointers

# ── Step 3: Run final chain ───────────────────────────────────────────────────
final_chain = prompt_final | model | parser_final

result: FinalSchema = final_chain.invoke({
    "topic"       : topic,
    "explanation" : explanation_text,
    "pointers"    : pointers_text,
})

# ── Print ─────────────────────────────────────────────────────────────────────
print("=" * 55)
print(f"  TOPIC: {result.topic}")
print("=" * 55)

print("\n EXPLANATION:\n", result.explanation)

print("\n SUMMARY:")
for i, p in enumerate(result.pointers, 1):
    print(f"  {i}. {p}")

print("\n PRACTICE QUESTIONS:")
for i, q in enumerate(result.questions, 1):
    print(f"  {i}. {q}")
