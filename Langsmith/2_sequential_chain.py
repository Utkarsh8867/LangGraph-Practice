from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# LangSmith project
os.environ["LANGCHAIN_PROJECT"] = "langsmith"

load_dotenv()

# Prompt 1
prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)

# Prompt 2
prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text:\n{text}",
    input_variables=["text"]
)

# First LLM
model1 = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# Second LLM (FIXED)
model2 = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

parser = StrOutputParser()

# LangSmith config
config = {
    "run_name": "sequential-chain-groq",
    "tags": ["sequential-chain", "groq"],
    "metadata": {
        "model1": "llama-3.1-8b-instant",
        "model2": "llama-3.1-8b-instant",
        "parser": "StrOutputParser"
    }
}

# Sequential chain
chain = prompt1 | model1 | parser | prompt2 | model2 | parser

# Run
result = chain.invoke(
    {"topic": "Unemployment in India"},
    config=config
)

print(result)
