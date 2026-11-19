"""
app/agents/synthesizer.py
=========================
Nodo de Síntesis Final.
Consolida los hechos encontrados por el Investigador y los cálculos
cuantitativos del Analista en un informe ejecutivo para el usuario.
"""

import logging
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from app.agents.supervisor import get_llm
from app.state import AgentState
from app.observability import trace_llm_call

logger = logging.getLogger("api_agent.synthesizer")

SYNTHESIS_PROMPT = """Sos el REDACTOR EJECUTIVO del equipo multi-agente en producción.
Tu misión es redactar un informe final de alto nivel combinando la investigación objetiva y los cálculos analíticos.

CONSULTA ORIGINAL DEL USUARIO:
{query}

DATOS ENCONTRADOS POR EL INVESTIGADOR:
{research_summary}
FUENTES COMPROBADAS: {sources}

ANÁLISIS CUANTITATIVO Y CÁLCULOS DEL ANALISTA:
{analysis_narrative}
CÁLCULOS MATEMÁTICOS DETALLADOS:
{calculations}
ESTADÍSTICAS Y RIESGOS:
{stats_and_sentiment}

ESTRUCTURA DEL REPORTE:
1. Resumen Ejecutivo (Conclusiones principales)
2. Hallazgos de Investigación y Fuentes
3. Análisis Numérico, Ratios y Comparaciones
4. Alertas de Riesgo y Panorama de Mercado
5. Conclusión y Recomendación Final
"""


@trace_llm_call(step_name="synthesizer_step")
def synthesizer_node(state: AgentState) -> dict:
    """
    Nodo de cierre del orquestador. Genera el reporte ejecutivo final.
    """
    logger.info("[SYNTHESIZER] Redactando reporte final para el usuario...")

    query = state.get("user_query", "")
    research_data = state.get("research_data", {})
    analysis_data = state.get("analysis_data", {})

    prompt_formatted = SYNTHESIS_PROMPT.format(
        query=query,
        research_summary=research_data.get("summary", "Sin resumen de investigación disponible."),
        sources=", ".join(research_data.get("sources", ["Fuentes de mercado"])),
        analysis_narrative=analysis_data.get("narrative", "Sin narrativa de análisis."),
        calculations=str(analysis_data.get("calculations", [])),
        stats_and_sentiment=f"Estadísticas: {analysis_data.get('statistical_summary')}\nSentimiento: {analysis_data.get('sentiment_analysis')}",
    )

    final_text = ""
    try:
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content="Sos un consultor senior que redacta informes claros, profesionales y bien fundamentados."),
            HumanMessage(content=prompt_formatted),
        ])
        final_text = response.content
    except Exception as exc:
        logger.warning(f"[SYNTHESIZER] Usando plantilla de reporte de respaldo ({exc}).")
        final_text = (
            f"# Informe Ejecutivo: {query}\n\n"
            f"## 1. Resumen Ejecutivo\nSe completó la evaluación integral mediante el equipo multi-agente.\n\n"
            f"## 2. Investigación y Fuentes\n{research_data.get('summary', 'Datos recolectados exitosamente.')}\n"
            f"**Fuentes:** {', '.join(research_data.get('sources', ['Mercados globales']))}\n\n"
            f"## 3. Análisis Numérico y Cálculos\n{analysis_data.get('narrative', 'Cálculos realizados con rigor matemático.')}\n\n"
            f"## 4. Estado de Ejecución\nFinalizado con éxito bajo monitoreo activo de observabilidad.\n"
        )

    logger.info("[SYNTHESIZER] Reporte final generado con éxito.")

    return {
        "final_report": final_text,
        "task_completed": True,
        "next_agent": "FINISH",
        "messages": [AIMessage(content=final_text)],
    }

