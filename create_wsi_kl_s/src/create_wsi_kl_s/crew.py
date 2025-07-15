from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List, Dict, Any, Optional
from pathlib import Path
import os
import json

from crewai.knowledge.source.crew_docling_source import CrewDoclingSource
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

from .init_llm import _default_llm  # Provides the configured Gemini LLM


@CrewBase
class CreateWsiKlS:
    """WSI Cancer Description Single-Agent System"""

    agents: List[BaseAgent]
    tasks: List[Task]
    
    def __init__(self, use_json_source: bool = False, json_file_path: Optional[str] = None):
        """Initialize the crew with optional JSON data source
        
        Args:
            use_json_source: If True, use JSON file as knowledge source instead of PDFs
            json_file_path: Path to JSON file containing cancer descriptions
        """
        super().__init__()
        self.use_json_source = use_json_source
        self.json_file_path = json_file_path
        
    def load_json_descriptions(self, file_path: str) -> Dict[str, Any]:
        """Load cancer descriptions from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"JSON file not found: {file_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file: {e}")
            return {}
            
    def validate_cancer_descriptions(self, descriptions: Dict[str, Any], cancer_type: str) -> List[str]:
        """Validate and select appropriate descriptions for the given cancer type
        
        Args:
            descriptions: Dictionary of cancer descriptions
            cancer_type: The cancer type to find descriptions for
            
        Returns:
            List of validated descriptions
        """
        if not descriptions:
            return []
            
        validated_descriptions = []
        
        # Direct match
        if cancer_type in descriptions:
            if isinstance(descriptions[cancer_type], list):
                validated_descriptions.extend(descriptions[cancer_type])
            elif isinstance(descriptions[cancer_type], str):
                validated_descriptions.append(descriptions[cancer_type])
                
        # Fuzzy matching for similar cancer types
        cancer_type_lower = cancer_type.lower()
        for key, value in descriptions.items():
            if key.lower() != cancer_type_lower and cancer_type_lower in key.lower():
                if isinstance(value, list):
                    validated_descriptions.extend(value)
                elif isinstance(value, str):
                    validated_descriptions.append(value)
                    
        return validated_descriptions

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
    def description_validation_task(self) -> Task:  # type: ignore[override]
        return Task(config=self.tasks_config["description_validation_task"])  # type: ignore[index]

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

        knowledge_sources = []
        
        if self.use_json_source and self.json_file_path:
            # Scenario 2: Use JSON file as knowledge source
            json_descriptions = self.load_json_descriptions(self.json_file_path)
            if json_descriptions:
                # Convert JSON descriptions to string format for knowledge source
                descriptions_text = ""
                for cancer_type, descriptions in json_descriptions.items():
                    descriptions_text += f"\n\n=== {cancer_type} ===\n"
                    if isinstance(descriptions, list):
                        for desc in descriptions:
                            descriptions_text += f"- {desc}\n"
                    else:
                        descriptions_text += f"{descriptions}\n"
                
                knowledge_sources.append(
                    StringKnowledgeSource(
                        content=descriptions_text,
                        metadata={"source": "json_cancer_descriptions"}
                    )
                )
        else:
            # Scenario 1: Use PDF files as knowledge source
            _abs_knowledge_dir = Path(__file__).resolve().parents[2] / "knowledge"
            knowledge_files = [
                "camelyon16.pdf",
                "tcga_lung.pdf", 
                "tcga_renal.pdf",
                "Pathoma 2021 - Kidney.pdf",
            ]

            for fname in knowledge_files:
                abs_path = _abs_knowledge_dir / fname
                if abs_path.exists():
                    knowledge_sources.append(
                        CrewDoclingSource(
                            file_paths=[str(abs_path)]
                        )
                    )

        # Select appropriate tasks based on data source
        if self.use_json_source:
            # For JSON source: validate existing descriptions instead of generating new ones
            selected_tasks = [
                self.planning_task(), 
                self.description_validation_task(),  # Use validation instead of generation
                self.description_evaluation_task(), 
                self.finalization_task()
            ]
        else:
            # For PDF source: generate new descriptions
            selected_tasks = [
                self.planning_task(), 
                self.description_generation_task(), 
                self.description_evaluation_task(), 
                self.finalization_task()
            ]

        return Crew(
            agents=self.agents,
            tasks=selected_tasks,
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
