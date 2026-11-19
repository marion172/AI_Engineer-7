"""
app package
Inicializa el entorno y las variables de observabilidad de LangSmith
antes de que se importe cualquier componente de LangChain o LangGraph.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Asegurar variables de entorno globales para LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
langchain_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_key and langchain_key != "tu_langsmith_api_key_aqui":
    os.environ["LANGCHAIN_API_KEY"] = langchain_key

project_name = os.getenv("LANGCHAIN_PROJECT", "pre-entrega-7-api")
os.environ["LANGCHAIN_PROJECT"] = project_name
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

__version__ = "1.0.0"
