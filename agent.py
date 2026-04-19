import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# 1. State Definition
class AgentState(TypedDict):
    messages: List[str]
    current_logic_step: str

# 2. Model Initialization
# Supervisor (Local SLM - tiny to save your RAM)
supervisor = ChatOllama(model="phi3:mini") 
# Reasoner (Cloud LLM - the heavy lifter)
reasoner = ChatGroq(model="llama-3.1-8b-instant")

# 3. Node: Socratic Reasoning (Cloud)
def socratic_node(state: AgentState):
    sys_msg = SystemMessage(content="""
        You are a Socratic Tutor. 
        1. Identify the 'atomic steps' of the user's problem.
        2. Ask ONE leading question to test their understanding of the first step.
        3. DO NOT give the answer.
    """)
    response = reasoner.invoke([sys_msg] + [HumanMessage(content=m) for m in state['messages']])
    return {"messages": state['messages'] + [response.content]}

# 4. Node: Local Check (Ollama)
def guardrail_node(state: AgentState):
    # This runs locally on your Mac to ensure the cloud didn't leak the answer
    last_msg = state['messages'][-1]
    check_prompt = f"Did this message give away the final answer? Answer only YES or NO: {last_msg}"
    response = supervisor.invoke(check_prompt)
    if "YES" in response.content.upper():
        return "tutor" # Loop back to regenerate if it failed
    return END

# 5. Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("tutor", socratic_node)
workflow.set_entry_point("tutor")
workflow.add_conditional_edges("tutor", guardrail_node, {"tutor": "tutor", END: END})

import sys

# Compile the graph
app = workflow.compile()


if __name__ == "__main__":
    # We start with an empty list of messages
    current_messages = []
    print("--- Socratic Sentinel Active (Type 'exit' to quit) ---")

    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        # We append the NEW user message to our history list
        current_messages.append(user_input)
        
        # Prepare the state with the FULL conversation history
        state = {
            "messages": current_messages, 
            "current_logic_step": "ongoing_dialogue"
        }
        
        print("\n--- Thinking... ---")
        final_state = app.invoke(state)
        
        # Get the AI's response
        ai_response = final_state["messages"][-1]
        
        # IMPORTANT: Append the AI's response to the history so the NEXT turn knows it
        current_messages.append(ai_response)
        
        print(f"\nSocratic Assistant: {ai_response}")