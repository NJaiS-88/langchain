from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", max_tokens=1000)

# ── Schemas ───────────────────────────────────────────────────────────────────
class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Sentiment of the text: either 'positive' or 'negative'")

class ResponseSchema(BaseModel):
    sentiment: Literal["positive", "negative"]
    response: str = Field(description="Appropriate response based on sentiment")

# ── Parsers ───────────────────────────────────────────────────────────────────
parser_sentiment = PydanticOutputParser(pydantic_object=SentimentSchema)
parser_response  = PydanticOutputParser(pydantic_object=ResponseSchema)

# ── Prompts ───────────────────────────────────────────────────────────────────
prompt_sentiment = ChatPromptTemplate.from_messages([
    ("system", "You are a sentiment analysis expert. Return valid JSON only."),
    ("human", "Identify the sentiment of this text:\n{text}\n{format_instructions}")
]).partial(format_instructions=parser_sentiment.get_format_instructions())

prompt_positive = ChatPromptTemplate.from_messages([
    ("system", "You are an enthusiastic and uplifting assistant. Return valid JSON only."),
    ("human", "User said something positive:\n{text}\nRespond with energy and encouragement.\n{format_instructions}")
]).partial(format_instructions=parser_response.get_format_instructions())

prompt_negative = ChatPromptTemplate.from_messages([
    ("system", "You are an empathetic and supportive assistant. Return valid JSON only."),
    ("human", "User said something negative:\n{text}\nRespond with empathy and support.\n{format_instructions}")
]).partial(format_instructions=parser_response.get_format_instructions())

# ── Branch Conditions ─────────────────────────────────────────────────────────
def is_positive(x):
    return x["sentiment"] == "positive"

def is_negative(x):
    return x["sentiment"] == "negative"

# ── Chains ────────────────────────────────────────────────────────────────────
sentiment_chain = prompt_sentiment | model | parser_sentiment

branch_chain = RunnableBranch(
    (is_positive, prompt_positive | model | parser_response),
    prompt_negative | model | parser_response,   # default → negative
)

# ── Run ───────────────────────────────────────────────────────────────────────
text = "I just got promoted today, I am so happy!"

sentiment_result = sentiment_chain.invoke({"text": text})

final_result = branch_chain.invoke({
    "text"      : text,
    "sentiment" : sentiment_result.sentiment
})

# ── Print ─────────────────────────────────────────────────────────────────────
print("=" * 55)
print(f"  SENTIMENT : {final_result.sentiment.upper()}")
print("=" * 55)
print(f"\n  RESPONSE  : {final_result.response}")
