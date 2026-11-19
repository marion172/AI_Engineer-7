"""
app/agents/supervisor.py
========================
Supervisor jerárquico del sistema multi-agente.
Coordina a los especialistas, evalúa la suficiencia de los datos, evita bucles
infinitos y detecta si una tarea requiere intervención humana (HITL).
"""

import os
import logging
from typing import Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.state import AgentState, AgentTarget, RubricEvaluation
from app.observability import trace_llm_call

load_dotenv()
logger = logging.getLogger("api_agent.supervisor")


class SupervisorRoutingDecision(BaseModel):
    """
    Decisión estructurada devuelta por el modelo para controlar el grafo.
    """
    next_agent: Literal["research_agent", "analyst_agent", "synthesizer", "human_approval"] = Field(
        description="A qué nodo le toca intervenir ahora."
    )
    research_sufficiency: float = Field(
        ge=0.0,
        le=1.0,
        description="Del 0 al 1, qué tan completos están los datos encontrados hasta ahora.",
    )
    analysis_rigor: float = Field(
        ge=0.0,
        le=1.0,
        description="Del 0 al 1, qué tan bien calculados y analizados están los números.",
    )
    synthesis_ready: bool = Field(
        description="True si ya tenemos suficiente información para armar el reporte final.",
    )
    is_critical_operation: bool = Field(
        default=False,
        description="True si la tarea o siguiente paso involucra operaciones financieras críticas, órdenes o alto impacto.",
    )
    critique_and_reasoning: str = Field(
        description="Por qué tomaste esta decisión.",
    )
    directive_for_agent: str = Field(
        description="Instrucción corta y clara para el especialista de turno.",
    )


SUPERVISOR_SYSTEM_PROMPT = """Sos el SUPERVISOR de un equipo de trabajo multi-agente en producción.
Tu rol es coordinar especialistas para responder la consulta del usuario de forma completa, precisa y eficiente:

Tus especialistas son:
1. 'research_agent': Busca hechos, balances, cifras y fuentes confiables.
2. 'analyst_agent': Hace cálculos matemáticos sobre esos datos (variaciones porcentuales, comparativas, ratios).
3. 'human_approval': Nodo de interrupción y aprobación obligatoria cuando la tarea implique operaciones críticas (ej. órdenes de compra/venta, rebalanceo de cartera, transferencias o alto impacto/riesgo).
4. 'synthesizer': Nodo final que arma el informe para el usuario cuando ya tenemos la investigación y los cálculos listos.

REGLAS DE DECISIÓN:
- Si no hay datos buscados o falta información clave -> mandá a 'research_agent'.
- Si ya tenemos datos pero faltan números procesados o cálculos -> mandá a 'analyst_agent'.
- Si la tarea contiene operaciones críticas (comprar/vender acciones, transferir fondos, costo muy alto) y NO ha sido aprobada previamente -> marcá is_critical_operation=True y mandá a 'human_approval'.
- Si ya investigamos y calculamos lo necesario -> marcá synthesis_ready=True y mandá a 'synthesizer'.
- Evitá bucles: si la información básica ya responde la consulta, cerrá la tarea de inmediato."""


def get_llm():
    """
    Instancia el cliente LLM configurado en las variables de entorno.
    Por defecto utiliza OpenRouter con el modelo nvidia/nemotron-3.5-lightning:free.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("OPENAI_MODEL", "nvidia/nemotron-3.5-lightning:free")

    kwargs = {"temperature": 0.1, "model": model}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)


def _heuristic_fallback_decision(state: AgentState, reason: str) -> SupervisorRoutingDecision:
    """
    Ruta de respaldo heurística ante fallos de conexión o indisponibilidad del LLM.
    """
    has_research = bool(state.get("research_data") and state["research_data"].get("items"))
    has_analysis = bool(state.get("analysis_data") and (
        state["analysis_data"].get("calculations") or state["analysis_data"].get("sentiment_analysis")
    ))
    iterations = state.get("iteration_count", 0)
    requires_approval = state.get("requires_human_approval", False)
    approval_status = state.get("approval_status", "NOT_REQUIRED")

    # Detección heurística de criticidad
    query_lower = state.get("user_query", "").lower()
    critical_keywords = ["comprar", "vender", "orden", "transferir", "liquidar", "ejecutar transacción", "alta criticidad"]
    is_critical = any(kw in query_lower for kw in critical_keywords) or requires_approval

    if is_critical and approval_status not in ("APPROVED", "REJECTED") and has_analysis:
        return SupervisorRoutingDecision(
            next_agent="human_approval",
            research_sufficiency=0.8,
            analysis_rigor=0.8,
            synthesis_ready=False,
            is_critical_operation=True,
            critique_and_reasoning=f"Operación crítica detectada en consulta ('{query_lower[:30]}...'). Se requiere aprobación humana previa.",
            directive_for_agent="Solicitar autorización al operador humano antes de proceder.",
        )

    if not has_research:
        return SupervisorRoutingDecision(
            next_agent="research_agent",
            research_sufficiency=0.1,
            analysis_rigor=0.0,
            synthesis_ready=False,
            is_critical_operation=False,
            critique_and_reasoning=f"Fallback ({reason}): Falta investigar hechos y números.",
            directive_for_agent="Investigar fuentes confiables y cifras relevantes sobre la consulta.",
        )
    elif not has_analysis:
        return SupervisorRoutingDecision(
            next_agent="analyst_agent",
            research_sufficiency=0.8,
            analysis_rigor=0.1,
            synthesis_ready=False,
            is_critical_operation=False,
            critique_and_reasoning=f"Fallback ({reason}): Datos encontrados, falta procesar cálculos.",
            directive_for_agent="Calcular crecimientos, ratios comparativos y estadísticas de los datos hallados.",
        )
    else:
        return SupervisorRoutingDecision(
            next_agent="synthesizer",
            research_sufficiency=0.9,
            analysis_rigor=0.9,
            synthesis_ready=True,
            is_critical_operation=False,
            critique_and_reasoning=f"Fallback ({reason}): Información suficiente o límite de pasos alcanzado.",
            directive_for_agent="Sintetizar el reporte final para el usuario.",
        )


@trace_llm_call(step_name="supervisor_decision")
def supervisor_node(state: AgentState) -> dict:
    """
    Nodo de orquestación central. Evalúa el estado actual y decide el siguiente paso.
    """
    iteration = state.get("iteration_count", 0) + 1
    max_iters = state.get("max_iterations", 5)

    logger.info(f"[SUPERVISOR] Vuelta {iteration}/{max_iters}")

    # Salvavidas: si alcanzamos el máximo de iteraciones, forzamos cierre
    if iteration >= max_iters:
        logger.info("[SUPERVISOR] Límite de iteraciones alcanzado. Derivando a síntesis final.")
        return {
            "iteration_count": iteration,
            "next_agent": "synthesizer",
            "supervisor_feedback": "Tiempo límite alcanzado: consolidar la mejor respuesta posible con lo disponible.",
            "rubric_scores": {
                "research_sufficiency": 0.7,
                "analysis_rigor": 0.7,
                "synthesis_ready": True,
                "critique": "Finalización forzada por límite de pasos para evitar bucle infinito.",
            },
        }

    # Contexto para el LLM
    query = state.get("user_query", "")
    research_summary = state.get("research_data", {}).get("summary", "Sin investigación previa.")
    analysis_narrative = state.get("analysis_data", {}).get("narrative", "Sin cálculos previos.")
    approval_status = state.get("approval_status", "NOT_REQUIRED")

    user_context = (
        f"CONSULTA DEL USUARIO: {query}\n"
        f"ESTADO DE APROBACIÓN HITL: {approval_status}\n"
        f"RESUMEN INVESTIGACIÓN ACTUAL:\n{research_summary}\n\n"
        f"RESUMEN ANÁLISIS ACTUAL:\n{analysis_narrative}\n\n"
        f"VUELTA ACTUAL: {iteration} de {max_iters}.\n"
    )

    decision: SupervisorRoutingDecision
    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(SupervisorRoutingDecision)
        decision = structured_llm.invoke([
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            {"role": "user", "content": user_context},
        ])
    except Exception as exc:
        logger.warning(f"[SUPERVISOR] Fallo al invocar structured output ({exc}). Usando fallback heurístico.")
        decision = _heuristic_fallback_decision(state, str(exc))

    logger.info(f"[SUPERVISOR] Decisión: {decision.next_agent} | Razón: {decision.critique_and_reasoning}")

    return {
        "iteration_count": iteration,
        "next_agent": decision.next_agent,
        "supervisor_feedback": decision.directive_for_agent,
        "is_critical": decision.is_critical_operation,
        "rubric_scores": {
            "research_sufficiency": decision.research_sufficiency,
            "analysis_rigor": decision.analysis_rigor,
            "synthesis_ready": decision.synthesis_ready,
            "critique": decision.critique_and_reasoning,
        },
        "messages": [
            AIMessage(
                content=f"[Supervisor] Próximo paso: '{decision.next_agent}'. Directiva: {decision.directive_for_agent}"
            )
        ],
    }

