import argparse
import sys
import os
from green_evaluator import GreenEvaluator, get_task_list_by_benchmark, A2A_AVAILABLE

def main():
    parser = argparse.ArgumentParser(description="Run Benchmark Suite")
    parser.add_argument("--agent_path", type=str, required=True, help="Path to the agent file")
    parser.add_argument("--benchmark", type=str, required=True, choices=["miniwob", "workarena", "webarena", "visualwebarena"], help="Benchmark suite to run")
    parser.add_argument("--max_steps", type=int, default=50, help="Max steps per task")
    parser.add_argument("--max_tasks", type=int, default=None, help="Limit number of tasks (useful for WebArena)")
    
    args = parser.parse_args()

    # Initialize evaluator
    evaluator = GreenEvaluator()
    
    # Load agent
    try:
        agent = evaluator.load_agent(args.agent_path)
    except Exception as e:
        print(f"Failed to load agent: {e}")
        sys.exit(1)

    # Get tasks
    try:
        tasks = get_task_list_by_benchmark(args.benchmark)
        if args.max_tasks:
            tasks = tasks[:args.max_tasks]
            print(f"Limited to first {args.max_tasks} tasks.")
    except Exception as e:
        print(f"Error getting task list: {e}")
        sys.exit(1)

    print(f"Running {args.benchmark} suite with {len(tasks)} tasks...")
    
    # Run suite
    results = evaluator.evaluate_agent_on_benchmark_suite(agent, tasks)
    
    # Print summary
    print(f"\nBenchmark Suite Results ({args.benchmark}):")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Successful Tasks: {results['successful_tasks']}/{results['total_tasks']}")

if __name__ == "__main__":
    main()