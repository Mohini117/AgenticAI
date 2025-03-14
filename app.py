import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field 
from typing import Annotated, List, Dict
from typing_extensions import TypedDict
import operator
import streamlit as st

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph, END
from langchain_groq import ChatGroq
from langgraph.constants import Send

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(model="Gemma2-9b-It", temperature=0.7)

# Schema for structured output
class Section(BaseModel):
    name: str = Field(description="Name for this section of the various queries.")
    description: str = Field(description="Answering questions to the users various queries.")

class Sections(BaseModel):
    sections: List[Section] = Field(description="Sections of the various queries.")

# Augment the LLM with structured output schema
planner = llm.with_structured_output(Sections)

# Graph State
class State(TypedDict):
    topic: str  # Report topic
    sections: List[Dict]  # Store sections as dictionaries
    completed_sections: Annotated[List, operator.add]  # Workers write here
    final_report: str  # Final report

# Worker State
class WorkerState(TypedDict):
    section: Dict  # Dictionary format instead of Pydantic model
    completed_sections: Annotated[List, operator.add]

def q_and_a_manager():
    # Node: Orchestrator – decides which worker to assign a query about a product.
    def orchestrator(state: State):
        report_sections = planner.invoke([
            SystemMessage(content="Decide which worker to assign tasks about a particular query of a particular product."),
            HumanMessage(content=f"Here is the product query: {state['topic']}")
        ])
        if not report_sections or not report_sections.sections:
            raise ValueError("No sections were generated for the report.")
        # Convert sections to a list of dictionaries
        sections_list = [section.dict() for section in report_sections.sections]
        # st.write("Generated Sections:", sections_list)
        return {"sections": sections_list}

    # Node: llm_call1 – handles Compatibility searches.
    def llm_call1(state: WorkerState):
        section = llm.invoke([
            SystemMessage(content="Answer the query about compatibility searches. Use markdown formatting."),
            HumanMessage(content=f"Query type: {state['section']['name']}, Answer: {state['section']['description']}")
        ])
        return {"completed_sections": [section.content]}

    # Node: llm_call2 – handles Non-product searches.
    def llm_call2(state: WorkerState):
        section = llm.invoke([
            SystemMessage(content="Answer the query about non-product searches. Use markdown formatting."),
            HumanMessage(content=f"Query type: {state['section']['name']}, Answer: {state['section']['description']}")
        ])
        return {"completed_sections": [section.content]}

    # Node: llm_call3 – handles Symptom/problem searches.
    def llm_call3(state: WorkerState):
        section = llm.invoke([
            SystemMessage(content="Answer the query about symptom/problem searches. Use markdown formatting."),
            HumanMessage(content=f"Query type: {state['section']['name']}, Answer: {state['section']['description']}")
        ])
        return {"completed_sections": [section.content]}

    # Node: llm_call4 – handles general product queries when no other worker applies.
    def llm_call4(state: WorkerState):
        section = llm.invoke([
            SystemMessage(content="Answer the query about the product. Use markdown formatting."),
            HumanMessage(content=f"Query type: {state['section']['name']}, Answer: {state['section']['description']}")
        ])
        return {"completed_sections": [section.content]}

    # Node: Synthesizer – combines responses into a final report.
    def synthesizer(state: State):
        completed_sections = state["completed_sections"]
        final_report = "\n\n---\n\n".join(completed_sections)
        return {"final_report": final_report}

    # Conditional edge function: assigns a worker based on section name.
    def assign_workers(state: State):
        worker_mapping = {
            "Compatibility": "llm_call1",
            "Non-product": "llm_call2",
            "Symptom/problem": "llm_call3",
            "General Product": "llm_call4"
        }
        return [Send(worker_mapping.get(s["name"], "llm_call4"), {"section": s}) for s in state["sections"]]

    # Build the workflow
    orchestrator_worker_builder = StateGraph(State)
    orchestrator_worker_builder.add_node("orchestrator", orchestrator)
    orchestrator_worker_builder.add_node("llm_call1", llm_call1)
    orchestrator_worker_builder.add_node("llm_call2", llm_call2)
    orchestrator_worker_builder.add_node("llm_call3", llm_call3)
    orchestrator_worker_builder.add_node("llm_call4", llm_call4)
    orchestrator_worker_builder.add_node("synthesizer", synthesizer)

    # Define the edges between nodes
    orchestrator_worker_builder.add_edge(START, "orchestrator")
    orchestrator_worker_builder.add_conditional_edges(
        "orchestrator", assign_workers, ["llm_call1", "llm_call2", "llm_call3", "llm_call4"]
    )
    orchestrator_worker_builder.add_edge("llm_call1", "synthesizer")
    orchestrator_worker_builder.add_edge("llm_call2", "synthesizer")
    orchestrator_worker_builder.add_edge("llm_call3", "synthesizer")
    orchestrator_worker_builder.add_edge("llm_call4", "synthesizer")
    orchestrator_worker_builder.add_edge("synthesizer", END)

    orchestrator_worker = orchestrator_worker_builder.compile()

    # Try to display the workflow diagram using st.image (if possible)
    try:
        graph_img = orchestrator_worker.get_graph().draw_mermaid_png()
        st.image(graph_img, caption="Workflow Diagram")
    except Exception as e:
        st.write("Could not display workflow diagram:", e)
    
    return orchestrator_worker

# Initialize the agent (workflow)
agent = q_and_a_manager()

# --- Streamlit Interface ---
st.title("🤖 Customer Support Chatbot")

query = st.text_area("Ask me anything about our products, policies, or support:")

if st.button("Get Answer"):
    if query.strip():
        try:
            # Prepare initial state
            state = {
                "topic": query,
                "sections": [],
                "completed_sections": [],
                "final_report": ""
            }
            # Run the workflow
            result = agent.invoke(state)
            st.markdown("### 📌 Final Answer:")
            st.markdown(result["final_report"])
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a query before submitting.")
