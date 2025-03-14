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

# Function to load local CSS
def local_css(file_name: str):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load custom CSS file
local_css("styles.css")

# Schema for structured output
class Section(BaseModel):
    name: str = Field(description="Teacher role name for this section (e.g., Math, Physics, Chemistry, Science).")
    description: str = Field(description="Details of the subject query that need to be answered.")

class Sections(BaseModel):
    sections: List[Section] = Field(description="Sections representing the teacher roles for the query.")

# Augment the LLM with structured output schema
planner = llm.with_structured_output(Sections)

# Graph State
class State(TypedDict):
    topic: str  # Subject query
    sections: List[Dict]  # Store sections as dictionaries
    completed_sections: Annotated[List, operator.add]  # Teacher responses
    final_report: str  # Final combined answer

# Worker State
class WorkerState(TypedDict):
    section: Dict  # Dictionary format instead of Pydantic model
    completed_sections: Annotated[List, operator.add]

def q_and_a_manager():
    # Node: Orchestrator – decides which teacher should answer the subject query.
    def orchestrator(state: State):
        report_sections = planner.invoke([
            SystemMessage(content="Decide which teacher should be assigned to answer the following subject query. The available teachers are for Math, Physics, Chemistry, and Science. Also, decide if a basic or advanced explanation is needed based on the query details."),
            HumanMessage(content=f"Subject query: {state['topic']}")
        ])
        if not report_sections or not report_sections.sections:
            raise ValueError("No sections were generated for the query.")
        # Use model_dump() instead of dict() per Pydantic V2
        sections_list = [section.model_dump() for section in report_sections.sections]
        return {"sections": sections_list}

    # Node: math_teacher – handles math queries.
    def math_teacher(state: WorkerState):
        section = llm.invoke([
            SystemMessage(content="You are a math teacher. Provide a clear explanation tailored to the user's query, noting if a basic or advanced explanation is needed. Use markdown formatting."),
            HumanMessage(content=f"Teacher role: {state['section']['name']}. Query details: {state['section']['description']}")
        ])
        return {"completed_sections": [section.content], "section": state["section"]}  # Preserve section key

    # Node: physics_teacher – handles physics queries.
    def physics_teacher(state: WorkerState):
        section = llm.invoke([
            SystemMessage(content="You are a physics teacher. Provide a clear and concise explanation tailored to the query. Use markdown formatting."),
            HumanMessage(content=f"Teacher role: {state['section']['name']}. Query details: {state['section']['description']}")
        ])
        return {"completed_sections": [section.content], "section": state["section"]}  # Preserve section key

    # Node: chemistry_teacher – handles chemistry queries.
    def chemistry_teacher(state: WorkerState):
        section = llm.invoke([
            SystemMessage(content="You are a chemistry teacher. Provide a clear explanation with relevant chemical concepts and equations where necessary. Use markdown formatting."),
            HumanMessage(content=f"Teacher role: {state['section']['name']}. Query details: {state['section']['description']}")
        ])
        return {"completed_sections": [section.content], "section": state["section"]}  # Preserve section key

    # Node: science_teacher – handles general science queries.
    def science_teacher(state: WorkerState):
        section = llm.invoke([
            SystemMessage(content="You are a science teacher. Provide an explanation that covers the relevant scientific concepts in a clear manner. Use markdown formatting."),
            HumanMessage(content=f"Teacher role: {state['section']['name']}. Query details: {state['section']['description']}")
        ])
        return {"completed_sections": [section.content], "section": state["section"]}  # Preserve section key

    # Node: Teacher Reviewer – reviews the teacher's response for alignment with the topic.
    def teacher_reviewer(state: WorkerState):
        # Ensure the section and completed_sections are available
        section_description = state["section"]["description"]
        teacher_response = state["completed_sections"][-1]  # Get the latest response

        # Review the response
        review = llm.invoke([
            SystemMessage(content="You are a reviewer. Review the teacher's response and ensure it is aligned with the topic question. If it is aligned, return the response as-is. If not, provide feedback for improvement."),
            HumanMessage(content=f"Topic: {section_description}\nTeacher Response: {teacher_response}")
        ])
        return {"completed_sections": [review.content], "section": state["section"]}  # Preserve section key

    # Node: Synthesizer – combines teacher responses into a final answer.
    def synthesizer(state: State):
        completed_sections = state["completed_sections"]
        final_report = "\n\n---\n\n".join(completed_sections)
        return {"final_report": final_report}

    # Conditional edge function: assigns a teacher based on section name.
    def assign_workers(state: State):
        worker_mapping = {
            "Math": "math_teacher",
            "Physics": "physics_teacher",
            "Chemistry": "chemistry_teacher",
            "Science": "science_teacher"
        }
        return [Send(worker_mapping.get(s["name"], "science_teacher"), {"section": s}) for s in state["sections"]]

    # Build the workflow
    orchestrator_worker_builder = StateGraph(State)
    orchestrator_worker_builder.add_node("orchestrator", orchestrator)
    orchestrator_worker_builder.add_node("math_teacher", math_teacher)
    orchestrator_worker_builder.add_node("physics_teacher", physics_teacher)
    orchestrator_worker_builder.add_node("chemistry_teacher", chemistry_teacher)
    orchestrator_worker_builder.add_node("science_teacher", science_teacher)
    orchestrator_worker_builder.add_node("teacher_reviewer", teacher_reviewer)
    orchestrator_worker_builder.add_node("synthesizer", synthesizer)

    # Define the edges between nodes
    orchestrator_worker_builder.add_edge(START, "orchestrator")
    orchestrator_worker_builder.add_conditional_edges(
        "orchestrator", assign_workers, ["math_teacher", "physics_teacher", "chemistry_teacher", "science_teacher"]
    )
    orchestrator_worker_builder.add_edge("math_teacher", "teacher_reviewer")
    orchestrator_worker_builder.add_edge("physics_teacher", "teacher_reviewer")
    orchestrator_worker_builder.add_edge("chemistry_teacher", "teacher_reviewer")
    orchestrator_worker_builder.add_edge("science_teacher", "teacher_reviewer")
    orchestrator_worker_builder.add_edge("teacher_reviewer", "synthesizer")
    orchestrator_worker_builder.add_edge("synthesizer", END)

    orchestrator_worker = orchestrator_worker_builder.compile()
    return orchestrator_worker

# Initialize the agent (workflow)
agent = q_and_a_manager()

# --- Streamlit Interface ---

# Custom Header
st.markdown("""
    <div class="main-header">
        <h1>Advanced Subject Teacher AI Agent</h1>
        <p class="subtitle">Your ultimate source for in-depth subject explanations</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar: Display the workflow diagram in a larger original form.
with st.sidebar:
    st.subheader("Workflow Diagram")
    try:
        graph_img = agent.get_graph().draw_mermaid_png()
        st.image(graph_img, caption="Workflow Diagram", use_container_width=True)
    except Exception as e:
        st.write("Could not display workflow diagram:", e)

# Main chatbot interface.
st.subheader("Chatbot")
query = st.text_area("Ask me anything about Math, Physics, Chemistry, or Science:")

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