
import logging
from typing import Literal

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END

from app.core.config import get_settings
from app.rag.state import AgentState, RouteDecision, EvidenceGrade
from app.rag.vectorstore import get_retriever


logger = logging.getLogger(__name__)
settings = get_settings()

_llm = None
_web_search = None


def llm():
    global _llm

    if _llm is None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing")

        _llm = ChatGroq(
            model=settings.groq_model,
            temperature=0,
            api_key=settings.groq_api_key,
        )

    return _llm


def web_search_tool():
    global _web_search

    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")

        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )

    return _web_search


def add_trace(state: AgentState, message: str):
    return [*state.get("trace", []), message]


# ---------------------------------------------------------
# ROUTER
# ---------------------------------------------------------

def route_question(state: AgentState):
    router = llm().with_structured_output(
        RouteDecision,
        method="json_mode"
    )

    decision = router.invoke(f"""
You route messages for an enterprise IT support assistant.

Use "kb" for questions about:
- company IT policies
- VPN
- password reset
- MFA
- laptop setup
- software access
- security
- email
- devices
- troubleshooting
- technology support
- internal company procedures

Use "direct" ONLY for:
- greetings
- thanks
- casual conversation
- questions that clearly require no company knowledge

IMPORTANT:
If the question could possibly be answered using the company's
private knowledge base, ALWAYS choose "kb".

Do not choose web search here.
Web search is only a fallback after the private KB has been checked.

Question:
{state['question']}

Return valid JSON:
{{"route":"kb"}}
or
{{"route":"direct"}}
""")

    return {
        "source_used": decision.route,
        "trace": add_trace(
            state,
            f"Router → {decision.route.upper()}"
        ),
    }


def route_after_router(
    state: AgentState,
) -> Literal["retrieve_kb", "direct_answer"]:

    return (
        "retrieve_kb"
        if state["source_used"] == "kb"
        else "direct_answer"
    )


# ---------------------------------------------------------
# PRIVATE KNOWLEDGE BASE RETRIEVAL
# ---------------------------------------------------------

def retrieve_kb(state: AgentState):
    docs = get_retriever().invoke(
        state["current_query"]
    )

    return {
        "kb_docs": docs,
        "trace": add_trace(
            state,
            f"Private KB retrieval → {len(docs)} chunks"
        ),
    }


# ---------------------------------------------------------
# PRIVATE KB EVIDENCE GRADING
# ---------------------------------------------------------

def grade_kb(state: AgentState):
    grader = llm().with_structured_output(
        EvidenceGrade,
        method="json_mode"
    )

    context = "\n\n".join(
        f"Source: {d.metadata.get('source', 'unknown')}\n"
        f"{d.page_content}"
        for d in state["kb_docs"]
    )

    grade = grader.invoke(f"""
You are evaluating whether a private company knowledge base
contains enough information to answer an enterprise IT support question.

Question:
{state['question']}

Private company KB evidence:
{context}

Evaluate the evidence using these rules:

1. Return "good" if the evidence directly answers the question.

2. Return "good" if the evidence contains the relevant instructions,
   troubleshooting steps, policy, or information needed to answer.

3. Do NOT require the document to use the exact same wording as the question.

4. Do NOT require every possible detail.

5. If the KB provides the main answer or useful troubleshooting steps,
   return "good".

6. Return "weak" only when the retrieved evidence is genuinely irrelevant,
   unrelated, or insufficient to safely answer the question.

7. Prefer the private company KB over external information.

Return valid JSON:

{{"grade":"good"}}

or

{{"grade":"weak"}}
""")

    return {
        "kb_grade": grade.grade,
        "trace": add_trace(
            state,
            f"KB evidence grade → {grade.grade.upper()}"
        ),
    }


# ---------------------------------------------------------
# DECIDE WHAT TO DO AFTER KB GRADING
# ---------------------------------------------------------

def after_kb(
    state: AgentState,
) -> Literal[
    "generate_from_kb",
    "rewrite_query",
    "search_web",
]:

    # KB contains enough information
    if state["kb_grade"] == "good":
        return "generate_from_kb"

    # KB did not contain enough information.
    # Try improving the query before going to the web.
    if state["retry_count"] < settings.max_retries:
        return "rewrite_query"

    # Only use web search after KB retrieval attempts fail.
    return "search_web"


# ---------------------------------------------------------
# WEB SEARCH
# ---------------------------------------------------------

def search_web(state: AgentState):
    result = web_search_tool().invoke(
        {
            "query": state["current_query"]
        }
    )

    lines = []
    citations = []

    if isinstance(result, dict):

        if result.get("answer"):
            lines.append(
                "Search answer: " + result["answer"]
            )

        for item in result.get("results", []):

            title = item.get("title", "")
            url = item.get("url", "")
            content = item.get("content", "")

            lines.append(
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Content: {content}"
            )

            citations.append(
                {
                    "title": title or url,
                    "url": url,
                    "type": "web",
                }
            )

    else:
        lines.append(str(result))

    return {
        "web_results": "\n\n".join(lines),
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(
            state,
            "Web fallback → Tavily search"
        ),
    }


# ---------------------------------------------------------
# WEB EVIDENCE GRADING
# ---------------------------------------------------------

def grade_web(state: AgentState):
    grader = llm().with_structured_output(
        EvidenceGrade,
        method="json_mode"
    )

    grade = grader.invoke(f"""
You evaluate web evidence for an enterprise IT support assistant.

Question:
{state['question']}

Web evidence:
{state['web_results']}

Return "good" if the evidence is:
- directly relevant
- sufficient to answer the question
- technically useful

Return "weak" if the evidence is:
- irrelevant
- contradictory
- insufficient
- unreliable

Return valid JSON:

{{"grade":"good"}}

or

{{"grade":"weak"}}
""")

    return {
        "web_grade": grade.grade,
        "trace": add_trace(
            state,
            f"Web evidence grade → {grade.grade.upper()}"
        ),
    }


# ---------------------------------------------------------
# DECIDE WHAT TO DO AFTER WEB SEARCH
# ---------------------------------------------------------

def after_web(
    state: AgentState,
) -> Literal[
    "generate_from_web",
    "rewrite_query",
    "insufficient",
]:

    if state["web_grade"] == "good":
        return "generate_from_web"

    if state["retry_count"] < settings.max_retries:
        return "rewrite_query"

    return "insufficient"


# ---------------------------------------------------------
# QUERY REWRITE
# ---------------------------------------------------------

def rewrite_query(state: AgentState):

    rewritten = llm().invoke(f"""
Rewrite the following enterprise IT support question into
a better search query for retrieving information from a
private company knowledge base.

Rules:
- Preserve the original intent.
- Add useful technical keywords where appropriate.
- Make the query specific.
- Do not answer the question.
- Return ONLY the rewritten search query.

Question:
{state['question']}
""").content.strip()

    return {
        "current_query": rewritten,
        "retry_count": state["retry_count"] + 1,
        "trace": add_trace(
            state,
            f"Query rewrite → {rewritten}"
        ),
    }


# ---------------------------------------------------------
# GENERATE ANSWER FROM PRIVATE KB
# ---------------------------------------------------------

def generate_from_kb(state: AgentState):

    context = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n"
        f"{d.page_content}"
        for d in state["kb_docs"]
    )

    answer = llm().invoke(f"""
You are an enterprise IT support copilot.

Answer ONLY from the private company knowledge base below.

Rules:
- Use the company KB as the source of truth.
- Do not use outside knowledge.
- Do not invent policies, procedures, or technical details.
- Be concise and actionable.
- If there are steps, present them clearly.
- If the KB does not provide a detail, do not make it up.
- Mention that the answer is based on the company's private knowledge base.

Question:
{state['question']}

Private KB:
{context}
""").content

    citations = []
    seen = set()

    for d in state["kb_docs"]:

        src = d.metadata.get(
            "source",
            "Private KB"
        )

        if src not in seen:

            seen.add(src)

            citations.append(
                {
                    "title": src.split("/")[-1],
                    "url": "",
                    "type": "private_kb",
                }
            )

    return {
        "answer": answer,
        "source_used": "private_kb",
        "citations": citations,
        "trace": add_trace(
            state,
            "Answer generation → PRIVATE KB"
        ),
    }


# ---------------------------------------------------------
# GENERATE ANSWER FROM WEB
# ---------------------------------------------------------

def generate_from_web(state: AgentState):

    answer = llm().invoke(f"""
You are an enterprise IT support copilot.

The private company knowledge base was insufficient.

Answer ONLY from the external web evidence below.

Rules:
- Clearly state that this is external web information.
- Do not pretend that the information comes from the company KB.
- Mention that company IT validation may be required before
  changing company-managed systems.
- Be concise and actionable.
- Do not invent information.

Question:
{state['question']}

Web evidence:
{state['web_results']}
""").content

    return {
        "answer": answer,
        "source_used": "web_search",
        "trace": add_trace(
            state,
            "Answer generation → WEB SEARCH"
        ),
    }


# ---------------------------------------------------------
# DIRECT ANSWER
# ---------------------------------------------------------

def direct_answer(state: AgentState):

    answer = llm().invoke(
        f"Respond briefly and naturally to: {state['question']}"
    ).content

    return {
        "answer": answer,
        "source_used": "direct",
        "trace": add_trace(
            state,
            "Direct response → no retrieval"
        ),
    }


# ---------------------------------------------------------
# INSUFFICIENT EVIDENCE
# ---------------------------------------------------------

def insufficient(state: AgentState):

    return {
        "answer": (
            "I couldn't find enough reliable evidence in the "
            "company knowledge base or external search to answer "
            "confidently. Please contact the IT help desk or "
            "provide more details."
        ),
        "source_used": "insufficient_evidence",
        "trace": add_trace(
            state,
            "Stopped → insufficient reliable evidence"
        ),
    }


# ---------------------------------------------------------
# BUILD LANGGRAPH
# ---------------------------------------------------------

def build_graph():

    graph = StateGraph(AgentState)

    for name, fn in {
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb,
        "search_web": search_web,
        "grade_web": grade_web,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "insufficient": insufficient,
    }.items():

        graph.add_node(name, fn)

    # START → Router
    graph.add_edge(
        START,
        "route_question"
    )

    # Router → KB or Direct
    graph.add_conditional_edges(
        "route_question",
        route_after_router,
        {
            "retrieve_kb": "retrieve_kb",
            "direct_answer": "direct_answer",
        },
    )

    # KB retrieval → KB grading
    graph.add_edge(
        "retrieve_kb",
        "grade_kb"
    )

    # KB grading → answer / retry / web
    graph.add_conditional_edges(
        "grade_kb",
        after_kb,
        {
            "generate_from_kb": "generate_from_kb",
            "rewrite_query": "rewrite_query",
            "search_web": "search_web",
        },
    )

    # Query rewrite → KB retrieval again
    graph.add_edge(
        "rewrite_query",
        "retrieve_kb"
    )

    # Web search → Web grading
    graph.add_edge(
        "search_web",
        "grade_web"
    )

    # Web grading → answer / retry / insufficient
    graph.add_conditional_edges(
        "grade_web",
        after_web,
        {
            "generate_from_web": "generate_from_web",
            "rewrite_query": "rewrite_query",
            "insufficient": "insufficient",
        },
    )

    # End nodes
    graph.add_edge(
        "generate_from_kb",
        END
    )

    graph.add_edge(
        "generate_from_web",
        END
    )

    graph.add_edge(
        "direct_answer",
        END
    )

    graph.add_edge(
        "insufficient",
        END
    )

    return graph.compile()


agent_graph = build_graph()


# ---------------------------------------------------------
# MAIN ASK FUNCTION
# ---------------------------------------------------------

def ask(question: str):
    initial: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }

    result = agent_graph.invoke(initial)

    print("\n========== AGENT TRACE ==========")

    for step in result.get("trace", []):
        print(step)

    print("=================================\n")

    return result