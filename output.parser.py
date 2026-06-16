from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 1. Define schema
class MySchema(BaseModel):
    definition: str = Field(description="A concise definition of the topic")
    example: str = Field(description="An example illustrating the topic")

# 2. Build parser
parser = PydanticOutputParser(pydantic_object=MySchema)
format_instructions = parser.get_format_instructions()

# 3. Prompt with partial variables
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Always respond with valid complete JSON only."),
    ("human", "Explain {topic}.\n{format_instructions}")
]).partial(format_instructions=format_instructions)

# 4. Model — increased max_tokens so JSON isn't cut off
load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", max_tokens=500)

# 5. Chain and final result
chain = prompt | model | parser
result = chain.invoke({"topic": "recursion"})

print("Definition:", result.definition)
print("Example:", result.example)
