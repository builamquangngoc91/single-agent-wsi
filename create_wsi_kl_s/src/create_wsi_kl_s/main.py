#!/usr/bin/env python
import sys
import warnings
from datetime import datetime
import argparse

from create_wsi_kl_s.crew import CreateWsiKlS

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """
    Run the WSI Cancer Description Single-Agent System.
    """
    parser = argparse.ArgumentParser(description="WSI Cancer Description Analysis")
    parser.add_argument("cancer_type", nargs="?", default="No Tumor (Negative Lymph Nodes)", 
                       help="Cancer type to analyze")
    parser.add_argument("--json-source", type=str, 
                       help="Path to JSON file containing cancer descriptions")
    
    args = parser.parse_args()
    
    cancer_type = args.cancer_type
    use_json_source = args.json_source is not None
    
    inputs = {
        "cancer_type": cancer_type,
        "current_year": str(datetime.now().year),
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
    }

    print(f"Starting single-agent WSI Cancer Description Analysis for: {cancer_type}")
    print(f"Analysis Date: {inputs['analysis_date']}")
    print(f"Data Source: {'JSON file' if use_json_source else 'PDF files'}")
    if use_json_source:
        print(f"JSON Source: {args.json_source}")
    print("-" * 50)

    try:
        crew_instance = CreateWsiKlS(
            use_json_source=use_json_source,
            json_file_path=args.json_source
        )
        result = crew_instance.crew().kickoff(inputs=inputs)
        print("\nWSI Cancer Description completed successfully!")
        print("Output saved to: wsi_cancer_description.md")
        return result
    except Exception as e:
        raise Exception(
            f"An error occurred while running the WSI cancer description system: {e}"
        )


def train():
    """Train the single-agent crew (fine-tuning)."""
    if len(sys.argv) < 3:
        print(
            "Usage: python -m create_wsi_kl_s.main train <n_iterations> <training_file> [cancer_type] [--json-source JSON_FILE]"
        )
        sys.exit(1)

    cancer_type = "Clear Cell Renal Cell Carcinoma (ccRCC)"
    json_source = None
    
    # Parse additional arguments
    if len(sys.argv) > 3:
        for i, arg in enumerate(sys.argv[3:], 3):
            if arg == "--json-source" and i + 1 < len(sys.argv):
                json_source = sys.argv[i + 1]
            elif not arg.startswith("--") and cancer_type == "Clear Cell Renal Cell Carcinoma (ccRCC)":
                cancer_type = arg

    inputs = {
        "cancer_type": cancer_type,
        "current_year": str(datetime.now().year),
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
    }

    crew_instance = CreateWsiKlS(
        use_json_source=json_source is not None,
        json_file_path=json_source
    )
    crew_instance.crew().train(
        n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs
    )


def replay():
    """Replay a previous execution from a specific task."""
    if len(sys.argv) < 2:
        print("Usage: python -m create_wsi_kl_s.main replay <task_id>")
        sys.exit(1)

    CreateWsiKlS().crew().replay(task_id=sys.argv[1])


def test():
    """Test the crew with evaluation LLM."""
    if len(sys.argv) < 3:
        print(
            "Usage: python -m create_wsi_kl_s.main test <n_iterations> <eval_llm> [cancer_type] [--json-source JSON_FILE]"
        )
        sys.exit(1)

    cancer_type = "Clear Cell Renal Cell Carcinoma (ccRCC)"
    json_source = None
    
    # Parse additional arguments
    if len(sys.argv) > 3:
        for i, arg in enumerate(sys.argv[3:], 3):
            if arg == "--json-source" and i + 1 < len(sys.argv):
                json_source = sys.argv[i + 1]
            elif not arg.startswith("--") and cancer_type == "Clear Cell Renal Cell Carcinoma (ccRCC)":
                cancer_type = arg

    inputs = {
        "cancer_type": cancer_type,
        "current_year": str(datetime.now().year),
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
    }

    crew_instance = CreateWsiKlS(
        use_json_source=json_source is not None,
        json_file_path=json_source
    )
    crew_instance.crew().test(
        n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs
    )


if __name__ == "__main__":
    # Allow modes similar to multi-agent script
    if len(sys.argv) > 1 and sys.argv[1] in {"train", "replay", "test"}:
        mode = sys.argv[1]
        sys.argv = sys.argv[1:]  # shift args for functions above
        if mode == "train":
            train()
        elif mode == "replay":
            replay()
        elif mode == "test":
            test()
    else:
        run()
