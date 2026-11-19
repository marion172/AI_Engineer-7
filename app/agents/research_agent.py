"""
app/agents/research_agent.py
============================
Especialista en Investigación y Recolección de Fuentes.
Busca hechos objetivos, reportes oficiales y cifras comprobables.
"""

import logging
from typing import Any, Dict
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from app.tools.search_tools import search_external_sources
from app.agents.supervisor import get_llm
from app.state import AgentState
from app.observability import trace_llm_call

logger = logging.getLogger("api_agent.research")

RESEARCHER_PROMPT = """Sos el INVESTIGADOR del equipo multi-agente.
Tu misión es buscar hechos, cifras oficiales y fuentes verificables en base a lo que te pide el Supervisor.
No inventes datos ni des opiniones personales: concentrate en los números y las fuentes.

INSTRUCCIÓN DEL SUPERVISOR:
{directive}

PEDIDO ORIGINAL:
{query}

RESULTADOS DE BÚSQUEDA OBTENIDOS:
{search_results}

TAREA:
1. Extraé los hechos clave, las cifras concretas y de dónde salieron (las fuentes).
2. Armá un resumen claro y ordenado para que el Agente de Análisis pueda usar esos números para calcular.
"""


@trace_llm_call(step_name="research_agent_step")
def research_agent_node(state: AgentState) -> dict:
    """
    Nodo de investigación en LangGraph.
    Ejecuta búsqueda de información y consolida 'research_data'.
    """
    directive = state.get("supervisor_feedback", "Investigar la consulta del usuario.")
    query = state.get("user_query", "")

    logger.info("[RESEARCH AGENT] Ejecutando búsqueda de fuentes y datos...")

    # Herramienta de búsqueda externa (Tavily, Wikipedia o Base de Respaldo)
    search_output = search_external_sources.invoke({"query": f"{query} {directive}"})
    items = search_output.get("items", [])

    extracted_metrics: Dict[str, Any] = {}
    sources = []

    for item in items:
        if isinstance(item, dict):
            if "metrics" in item and item["metrics"]:
                extracted_metrics[item.get("id", item.get("title", "metric"))] = item["metrics"]
            if "source" in item:
                sources.append(item["source"])

    logger.info(f"[RESEARCH AGENT] Encontrados {len(items)} registros en fuentes: {list(set(sources))}")

    # Síntesis estructurada mediante LLM
    prompt_formatted = RESEARCHER_PROMPT.format(
        directive=directive,
        query=query,
        search_results=str(items),
    )

    summary_text = ""
    try:
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content="Sos un investigador metódico y objetivo."),
            HumanMessage(content=prompt_formatted),
        ])
        summary_text = response.content
    except Exception as exc:
        logger.warning(f"[RESEARCH AGENT] Usando resumen estructurado de respaldo ({exc}).")
        summary_text = (
            f"Se consolidaron {len(items)} fuentes clave para '{query}'.\n"
            f"Fuentes: {', '.join(set(sources)) or 'Fuentes financieras/tecnológicas'}.\n"
            f"Métricas detectadas: {list(extracted_metrics.keys())}."
        )

    updated_research_data = {
        "items": items,
        "extracted_metrics": extracted_metrics,
        "sources": list(set(sources)),
        "summary": summary_text,
    }

    return {
        "research_data": updated_research_data,
        "messages": [
            AIMessage(
                content=f"[Research Agent] Hallados {len(items)} registros. Métricas listas para análisis.\n\n{summary_text}"
            )
        ],
    }

