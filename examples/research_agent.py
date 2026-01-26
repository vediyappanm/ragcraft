from typing import Dict
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate

class ResearchAgent:
    """Agent that plans, retrieves, and synthesizes information."""
    
    def __init__(self, tools: list, llm):
        self.agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=self._create_agent_prompt()
        )
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=tools,
            max_iterations=5,
            early_stopping_method="generate",
            verbose=True
        )
    
    def research(self, question: str) -> Dict:
        """Execute research task with planning and tool use."""
        try:
            result = self.executor.invoke({
                "input": f"Research this question: {question}",
                "chat_history": []
            })
            return {
                "answer": result["output"],
                "steps": result["intermediate_steps"],
                "sources": self._extract_sources(result)
            }
        except Exception as e:
            # Fallback to standard RAG
            return self.fallback_to_simple_rag(question)
    
    def _create_agent_prompt(self) -> PromptTemplate:
        """Create the agent prompt template."""
        template = """You are a research assistant that helps answer complex questions by using available tools.
        
        You have access to the following tools:
        {tools}
        
        Use the following format:
        
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        
        Begin!
        
        Question: {input}
        Thought:{agent_scratchpad}"""
        
        return PromptTemplate.from_template(template)
    
    def _extract_sources(self, result: Dict) -> list:
        """Extract sources from agent execution result."""
        sources = []
        for step in result.get("intermediate_steps", []):
            action, observation = step
            if hasattr(action, 'tool') and hasattr(observation, 'page_content'):
                sources.append({
                    "tool": action.tool,
                    "content": observation.page_content[:200] + "...",
                    "metadata": getattr(observation, 'metadata', {})
                })
        return sources
    
    def fallback_to_simple_rag(self, question: str) -> Dict:
        """Fallback method when agent fails."""
        return {
            "answer": f"I encountered an error while researching '{question}'. Please try again or rephrase your question.",
            "steps": [],
            "sources": [],
            "error": "Agent execution failed, used fallback response"
        }
