"""
app/graph.py
============
Orquestador Multi-Agente con Persistencia en Redis y Nodo Human-In-The-Loop.
Integra los especialistas (Supervisor, Investigador, Analista, Síntesis)
junto con el nodo de aprobación obligatoria ante operaciones críticas.
"""

import os
import logging
from typing import Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.state import AgentState
from app.agents.supervisor import supervisor_node
from app.agents.research_agent import research_agent_node
from app.agents.analyst_agent import analyst_agent_node
from app.agents.synthesizer import synthesizer_node
from app.hitl import human_approval_node

logger = logging.getLogger("api_agent.graph")

_COMPILED_GRAPH = None
_CHECKPOINTER = None


def supervisor_router(state: AgentState) -> Literal["research_agent", "analyst_agent", "synthesizer", "human_approval"]:
    """
    Ruteo condicional del Supervisor.
    Determina si la ejecución se deriva a investigación, cómputo, aprobación humana o síntesis.
    """
    next_step = state.get("next_agent", "research_agent")
    is_critical = state.get("is_critical", False)
    approval_status = state.get("approval_status", "NOT_REQUIRED")

    # Si se identificó criticidad y aún no se autorizó, derivar al nodo HITL
    if (is_critical or next_step == "human_approval") and approval_status == "PENDING":
        return "human_approval"

    if next_step == "FINISH" or next_step == "synthesizer":
        return "synthesizer"
    elif next_step == "analyst_agent":
        return "analyst_agent"
    elif next_step == "human_approval":
        return "human_approval"
    else:
        return "research_agent"


def hitl_router(state: AgentState) -> Literal["synthesizer", "end"]:
    """
    Ruteo condicional posterior a la intervención humana (HITL).
    Si la operación fue autorizada continúa hacia síntesis final.
    Si fue rechazada, aborta el flujo derivando a END.
    """
    status = state.get("approval_status")
    if status == "REJECTED":
        logger.info("[HITL ROUTER] Operación rechazada por el operador. Finalizando flujo.")
        return "end"
    logger.info("[HITL ROUTER] Operación aprobada. Procediendo a síntesis final.")
    return "synthesizer"


def build_orchestrator_graph() -> StateGraph:
    """
    Construye la topología del grafo multi-agente con nodo HITL:
    - START -> supervisor
    - supervisor -> (research_agent | analyst_agent | human_approval | synthesizer)
    - research_agent -> supervisor
    - analyst_agent -> supervisor
    - human_approval -> (synthesizer si APPROVED | END si REJECTED)
    - synthesizer -> END
    """
    workflow = StateGraph(AgentState)

    # 1. Definición de Nodos
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research_agent", research_agent_node)
    workflow.add_node("analyst_agent", analyst_agent_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # 2. Punto de entrada
    workflow.add_edge(START, "supervisor")

    # 3. Aristas condicionales del Supervisor
    workflow.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "research_agent": "research_agent",
            "analyst_agent": "analyst_agent",
            "human_approval": "human_approval",
            "synthesizer": "synthesizer",
        },
    )

    # 4. Retorno de especialistas al Supervisor
    workflow.add_edge("research_agent", "supervisor")
    workflow.add_edge("analyst_agent", "supervisor")

    # 5. Aristas de salida del nodo Human-in-the-loop
    workflow.add_conditional_edges(
        "human_approval",
        hitl_router,
        {
            "synthesizer": "synthesizer",
            "end": END,
        },
    )

    # 6. Finalización
    workflow.add_edge("synthesizer", END)

    return workflow


def get_checkpointer():
    """
    Instancia el checkpointer de LangGraph para persistir el estado entre pasos e interrupciones.
    Prioridad:
    1. RedisSaver (si REDIS_URL está activa y la librería está disponible).
    2. SqliteSaver persistente local (si SQLite checkpoint está disponible).
    3. MemorySaver en memoria (fallback seguro para pruebas rápidas).
    """
    global _CHECKPOINTER
    if _CHECKPOINTER is not None:
        return _CHECKPOINTER

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # 1. Intentar RedisSaver
    try:
        from langgraph.checkpoint.redis import RedisSaver
        # Si RedisSaver está disponible en langgraph.checkpoint.redis
        checkpointer = RedisSaver.from_conn_info(redis_url)
        logger.info(f"[Checkpointer] RedisSaver configurado exitosamente contra {redis_url}")
        _CHECKPOINTER = checkpointer
        return _CHECKPOINTER
    except Exception as exc_redis:
        try:
            # Alternativa de importación según versión de paquete
            from langgraph_checkpoint_redis import RedisSaver
            checkpointer = RedisSaver.from_conn_info(redis_url)
            logger.info(f"[Checkpointer] RedisSaver configurado exitosamente contra {redis_url}")
            _CHECKPOINTER = checkpointer
            return _CHECKPOINTER
        except Exception:
            logger.info(f"[Checkpointer] RedisSaver no inicializado ({exc_redis}). Evaluando persistencia de respaldo.")

    # 2. Intentar SqliteSaver local como fallback persistente
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        import sqlite3
        db_path = os.path.join(os.getcwd(), "agent_checkpoints.db")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        logger.info(f"[Checkpointer] Usando SqliteSaver persistente en '{db_path}'.")
        _CHECKPOINTER = checkpointer
        return _CHECKPOINTER
    except Exception as exc_sqlite:
        logger.info(f"[Checkpointer] SqliteSaver no disponible ({exc_sqlite}). Usando MemorySaver.")

    # 3. Fallback en memoria
    _CHECKPOINTER = MemorySaver()
    logger.info("[Checkpointer] MemorySaver inicializado como fallback.")
    return _CHECKPOINTER


def get_graph():
    """
    Obtiene la instancia compilada del grafo con su checkpointer activo.
    """
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        checkpointer = get_checkpointer()
        workflow = build_orchestrator_graph()
        _COMPILED_GRAPH = workflow.compile(checkpointer=checkpointer)
        logger.info("[Graph] Grafo multi-agente compilado y listo para ejecución.")
    return _COMPILED_GRAPH

