from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from pathlib import Path
import os

from crewai.knowledge.source.crew_docling_source import CrewDoclingSource

from .init_llm import _default_llm  # Provides the configured Gemini LLM


@CrewBase
class CreateWsiKlS:
    """WSI Cancer Description Single-Agent System"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # ---------------------------------------------------------------------
    # Agent definition (single agent only)
    # ---------------------------------------------------------------------
    @agent
    def wsi_cancer_agent(self) -> Agent:  # type: ignore[override]
        return Agent(
            config=self.agents_config["wsi_cancer_agent"],  # type: ignore[index]
            verbose=True,
            llm=_default_llm,
        )

    # ---------------------------------------------------------------------
    # Task definitions – executed sequentially by the single agent
    # ---------------------------------------------------------------------
    @task
    def planning_task(self) -> Task:  # type: ignore[override]
        return Task(config=self.tasks_config["planning_task"])  # type: ignore[index]

    @task
    def description_generation_task(self) -> Task:  # type: ignore[override]
        return Task(config=self.tasks_config["description_generation_task"])  # type: ignore[index]

    @task
    def description_evaluation_task(self) -> Task:  # type: ignore[override]
        return Task(config=self.tasks_config["description_evaluation_task"])  # type: ignore[index]

    @task
    def finalization_task(self) -> Task:  # type: ignore[override]
        return Task(
            config=self.tasks_config["finalization_task"],  # type: ignore[index]
            output_file="wsi_cancer_description.md",
        )

    # ---------------------------------------------------------------------
    # Crew definition
    # ---------------------------------------------------------------------
    @crew
    def crew(self) -> Crew:  # type: ignore[override]
        """Create the WSI Cancer Description Single-Agent crew"""

        # Initialize knowledge sources (reuse those available, if present)
        knowledge_sources = []
        _abs_knowledge_dir = Path(__file__).resolve().parents[2] / "knowledge"
        knowledge_files = [
            "Pathoma 2021 - Kidney.pdf",
        ]

        for fname in knowledge_files:
            abs_path = _abs_knowledge_dir / fname
            if abs_path.exists():
                knowledge_sources.append(
                    CrewDoclingSource(
                        file_paths=[str(fname)]
                    )  # relative path handled internally
                )

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            knowledge_sources=knowledge_sources,
            embedder={
                "provider": "google",
                "config": {
                    "model": "models/text-embedding-004",
                    "api_key": os.getenv("GEMINI_API_KEY"),
                },
            },
        )
