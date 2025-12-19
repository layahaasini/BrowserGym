import os
import pathlib
import pytest
import tempfile
import time
from unittest.mock import Mock, patch

from green_evaluator import GreenEvaluator, get_miniwob_task_list, get_workarena_task_list, get_webarena_task_list
from browsergym.experiments.agent import Agent
from browsergym.core.action.highlevel import HighLevelActionSet


@pytest.fixture
def temp_results_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def evaluator(temp_results_dir):
    return GreenEvaluator(results_dir=temp_results_dir)


@pytest.fixture
def simple_agent_file(temp_results_dir):
    agent_path = pathlib.Path(temp_results_dir) / "test_agent.py"
    agent_code = '''
from browsergym.experiments.agent import Agent
from browsergym.core.action.highlevel import HighLevelActionSet
class TestAgent(Agent):
    def __init__(self):
        super().__init__()
        self.action_set = HighLevelActionSet(subsets=["chat", "bid"], strict=False, multiaction=False)
        self.steps = 0
    
    def obs_preprocessor(self, obs):
        return obs
    
    def get_action(self, obs):
        self.steps += 1
        if self.steps == 1:
            return 'click("5")', {}
        return 'send_msg_to_user("Done")', {}
'''
    with open(agent_path, 'w') as f:
        f.write(agent_code)
    return str(agent_path)


class TestInitialization:
    def test_evaluator_initialization(self, evaluator):
        assert evaluator.results_dir.exists()
        assert evaluator.logger is not None
        assert evaluator.evaluation_metrics is not None
    
    def test_results_dir_created(self, temp_results_dir):
        evaluator = GreenEvaluator(results_dir=temp_results_dir)
        assert pathlib.Path(temp_results_dir).exists()


class TestAgentLoading:
    def test_load_agent_from_file(self, evaluator, simple_agent_file):
        agent = evaluator.load_agent(simple_agent_file)
        assert agent is not None
        assert hasattr(agent, 'get_action')
        assert hasattr(agent, 'action_set')
    
    def test_load_demo_agent(self, evaluator):
        agent = evaluator.load_agent("agents/agent.py")
        assert agent is not None
    
    def test_load_nonexistent_agent_raises_error(self, evaluator):
        with pytest.raises(Exception):
            evaluator.load_agent("/nonexistent/agent.py")


class TestTaskEvaluation:
    @pytest.mark.slow
    def test_evaluate_miniwob_task(self, evaluator, simple_agent_file):
        if not os.getenv('MINIWOB_URL'):
            pytest.skip("MINIWOB_URL not set")
        
        agent = evaluator.load_agent(simple_agent_file)
        result = evaluator.evaluate_agent_on_task(agent, "miniwob.click-dialog", max_steps=10)
        
        assert "task_name" in result
        assert result["task_name"] == "miniwob.click-dialog"
        assert "success" in result
        assert "steps_taken" in result
        assert "total_reward" in result
        assert result["steps_taken"] <= 10
    
    @pytest.mark.slow
    @pytest.mark.workarena
    def test_evaluate_workarena_task(self, evaluator, simple_agent_file):
        if not all([os.getenv('SNOW_INSTANCE_URL'), os.getenv('SNOW_INSTANCE_UNAME')]):
            pytest.skip("WorkArena environment not configured")
        
        agent = evaluator.load_agent(simple_agent_file)
        result = evaluator.evaluate_agent_on_task(agent, "workarena.servicenow.order-standard-laptop", max_steps=10)
        
        assert "task_name" in result
        assert "success" in result
        assert "steps_taken" in result
    
    @pytest.mark.slow
    @pytest.mark.webarena
    def test_evaluate_webarena_task(self, evaluator, simple_agent_file):
        if not os.getenv('WA_SHOPPING'):
            pytest.skip("WebArena environment not configured")
        
        agent = evaluator.load_agent(simple_agent_file)
        result = evaluator.evaluate_agent_on_task(agent, "webarena.4", max_steps=10)
        
        assert "task_name" in result
        assert "success" in result
        assert "steps_taken" in result


class TestBackendPreparation:
    @pytest.mark.workarena
    def test_workarena_backend_preparation(self, evaluator, simple_agent_file):
        if not all([os.getenv('SNOW_INSTANCE_URL'), os.getenv('SNOW_INSTANCE_UNAME')]):
            pytest.skip("WorkArena environment not configured")
        
        agent = evaluator.load_agent(simple_agent_file)
        result = evaluator.evaluate_agent_on_task(agent, "workarena.servicenow.order-standard-laptop", max_steps=5)
        
        # Backend preparation should not cause errors
        assert isinstance(result, dict)
    
    @pytest.mark.webarena
    def test_webarena_backend_preparation(self, evaluator, simple_agent_file):
        if not os.getenv('WA_SHOPPING'):
            pytest.skip("WebArena environment not configured")
        
        agent = evaluator.load_agent(simple_agent_file)
        result = evaluator.evaluate_agent_on_task(agent, "webarena.4", max_steps=5)
        
        # Backend preparation should not cause errors
        assert isinstance(result, dict)
    
    def test_backend_preparation_failure_logged(self, evaluator, simple_agent_file, caplog):
        if not os.getenv('MINIWOB_URL'):
            pytest.skip("MINIWOB_URL not set")
        
        # MiniWoB tasks don't require backend preparation, but should handle gracefully
        agent = evaluator.load_agent(simple_agent_file)
        result = evaluator.evaluate_agent_on_task(agent, "miniwob.click-test", max_steps=5)
        
        assert isinstance(result, dict)


class TestErrorHandling:
    def test_invalid_task_name(self, evaluator, simple_agent_file):
        agent = evaluator.load_agent(simple_agent_file)
        result = evaluator.evaluate_agent_on_task(agent, "invalid.task.name", max_steps=5)
        
        assert "error" in result or "task_name" in result
    
    def test_agent_exception_during_action(self, evaluator):
        if not os.getenv('MINIWOB_URL'):
            pytest.skip("MINIWOB_URL not set")
        
        class BrokenAgent(Agent):
            def __init__(self):
                super().__init__()
                self.action_set = HighLevelActionSet(subsets=["chat"], strict=False, multiaction=False)
            
            def obs_preprocessor(self, obs):
                return obs
            
            def get_action(self, obs):
                raise ValueError("Test exception")
        
        agent = BrokenAgent()
        result = evaluator.evaluate_agent_on_task(agent, "miniwob.click-test", max_steps=5)
        
        assert isinstance(result, dict)
    
    def test_agent_returns_none(self, evaluator):
        if not os.getenv('MINIWOB_URL'):
            pytest.skip("MINIWOB_URL not set")
        
        class NoneAgent(Agent):
            def __init__(self):
                super().__init__()
                self.action_set = HighLevelActionSet(subsets=["chat"], strict=False, multiaction=False)
            
            def obs_preprocessor(self, obs):
                return obs
            
            def get_action(self, obs):
                return None, {}
        
        agent = NoneAgent()
        result = evaluator.evaluate_agent_on_task(agent, "miniwob.click-test", max_steps=5)
        
        assert isinstance(result, dict)
        assert result["steps_taken"] >= 0
    
    def test_invalid_action_format(self, evaluator):
        if not os.getenv('MINIWOB_URL'):
            pytest.skip("MINIWOB_URL not set")
        
        class InvalidActionAgent(Agent):
            def __init__(self):
                super().__init__()
                self.action_set = HighLevelActionSet(subsets=["bid", "chat"], strict=False, multiaction=False)
            
            def obs_preprocessor(self, obs):
                return obs
            
            def get_action(self, obs):
                return 'click("999999")', {}
        
        agent = InvalidActionAgent()
        result = evaluator.evaluate_agent_on_task(agent, "miniwob.click-test", max_steps=5)
        
        assert isinstance(result, dict)


class TestEvaluationMetrics:
    @pytest.mark.slow
    def test_tracks_total_tasks(self, evaluator, simple_agent_file):
        if not os.getenv('MINIWOB_URL'):
            pytest.skip("MINIWOB_URL not set")
        
        agent = evaluator.load_agent(simple_agent_file)
        task_list = ["miniwob.click-test", "miniwob.click-dialog"]
        results = evaluator.evaluate_agent_on_benchmark_suite(agent, task_list)
        
        assert results["total_tasks"] == len(task_list)
    
    @pytest.mark.slow
    def test_tracks_successful_and_failed_tasks(self, evaluator, simple_agent_file):
        if not os.getenv('MINIWOB_URL'):
            pytest.skip("MINIWOB_URL not set")
        
        agent = evaluator.load_agent(simple_agent_file)
        task_list = ["miniwob.click-test"]
        results = evaluator.evaluate_agent_on_benchmark_suite(agent, task_list)
        
        assert "successful_tasks" in results
        assert "failed_tasks" in results
        assert results["successful_tasks"] + results["failed_tasks"] == results["total_tasks"]
    
    @pytest.mark.slow
    def test_calculates_success_rate(self, evaluator, simple_agent_file):
        if not os.getenv('MINIWOB_URL'):
            pytest.skip("MINIWOB_URL not set")
        
        agent = evaluator.load_agent(simple_agent_file)
        task_list = ["miniwob.click-test"]
        results = evaluator.evaluate_agent_on_benchmark_suite(agent, task_list)
        
        assert "success_rate" in results
        assert 0 <= results["success_rate"] <= 1
        expected_rate = results["successful_tasks"] / results["total_tasks"]
        assert results["success_rate"] == expected_rate
    
    @pytest.mark.slow
    def test_calculates_average_steps(self, evaluator, simple_agent_file):
        if not os.getenv('MINIWOB_URL'):
            pytest.skip("MINIWOB_URL not set")
        
        agent = evaluator.load_agent(simple_agent_file)
        task_list = ["miniwob.click-test"]
        results = evaluator.evaluate_agent_on_benchmark_suite(agent, task_list)
        
        assert "average_steps" in results
        assert results["average_steps"] >= 0
    
    @pytest.mark.slow
    def test_calculates_average_reward(self, evaluator, simple_agent_file):
        if not os.getenv('MINIWOB_URL'):
            pytest.skip("MINIWOB_URL not set")
        
        agent = evaluator.load_agent(simple_agent_file)
        task_list = ["miniwob.click-test"]
        results = evaluator.evaluate_agent_on_benchmark_suite(agent, task_list)
        
        assert "average_reward" in results
    
    @pytest.mark.slow
    def test_saves_task_results(self, evaluator, simple_agent_file, temp_results_dir):
        if not os.getenv('MINIWOB_URL'):
            pytest.skip("MINIWOB_URL not set")
        
        agent = evaluator.load_agent(simple_agent_file)
        task_list = ["miniwob.click-test"]
        results = evaluator.evaluate_agent_on_benchmark_suite(agent, task_list)
        
        assert "task_results" in results
        assert len(results["task_results"]) == len(task_list)
        
        for task_result in results["task_results"]:
            assert "task_name" in task_result
            assert "success" in task_result
            assert "steps_taken" in task_result
            assert "total_reward" in task_result


class TestReproducibility:
    @pytest.mark.slow
    def test_same_agent_same_task_identical_results(self, evaluator, simple_agent_file):
        if not os.getenv('MINIWOB_URL'):
            pytest.skip("MINIWOB_URL not set")
        
        agent1 = evaluator.load_agent(simple_agent_file)
        result1 = evaluator.evaluate_agent_on_task(agent1, "miniwob.click-test", max_steps=10)
        
        time.sleep(0.1)
        
        agent2 = evaluator.load_agent(simple_agent_file)
        result2 = evaluator.evaluate_agent_on_task(agent2, "miniwob.click-test", max_steps=10)
        
        assert result1["task_name"] == result2["task_name"]
        assert result1["success"] == result2["success"]
        assert result1["steps_taken"] == result2["steps_taken"]
        assert result1["total_reward"] == result2["total_reward"]


class TestIntegration:
    @pytest.mark.slow
    def test_full_evaluation_pipeline(self, evaluator, simple_agent_file, temp_results_dir):
        if not os.getenv('MINIWOB_URL'):
            pytest.skip("MINIWOB_URL not set")
        
        # Load agent
        agent = evaluator.load_agent(simple_agent_file)
        
        # Evaluate on tasks
        task_list = ["miniwob.click-test"]
        results = evaluator.evaluate_agent_on_benchmark_suite(agent, task_list)
        
        # Check results structure
        assert results["total_tasks"] == len(task_list)
        assert len(results["task_results"]) == len(task_list)
        
        # Check results file created
        results_files = list(pathlib.Path(temp_results_dir).glob("benchmark_evaluation_*.json"))
        assert len(results_files) > 0
    
    @pytest.mark.slow
    def test_multiple_tasks_in_sequence(self, evaluator, simple_agent_file):
        if not os.getenv('MINIWOB_URL'):
            pytest.skip("MINIWOB_URL not set")
        
        agent = evaluator.load_agent(simple_agent_file)
        task_list = ["miniwob.click-test", "miniwob.click-dialog"]
        results = evaluator.evaluate_agent_on_benchmark_suite(agent, task_list)
        
        assert results["total_tasks"] == len(task_list)
        assert len(results["task_results"]) == len(task_list)


class TestPerformance:
    @pytest.mark.slow
    def test_evaluation_completes_within_time(self, evaluator, simple_agent_file):
        if not os.getenv('MINIWOB_URL'):
            pytest.skip("MINIWOB_URL not set")
        
        agent = evaluator.load_agent(simple_agent_file)
        
        start_time = time.time()
        result = evaluator.evaluate_agent_on_task(agent, "miniwob.click-test", max_steps=10)
        end_time = time.time()
        
        duration = end_time - start_time
        assert duration < 60  # Should complete within 60 seconds


class TestTaskLists:
    def test_get_miniwob_task_list(self):
        task_list = get_miniwob_task_list()
        assert isinstance(task_list, list)
        assert len(task_list) > 0
        assert all(task.startswith("miniwob.") for task in task_list)
    
    def test_get_workarena_task_list(self):
        task_list = get_workarena_task_list()
        assert isinstance(task_list, list)
    
    def test_get_webarena_task_list(self):
        task_list = get_webarena_task_list()
        assert isinstance(task_list, list)


class TestA2AServer:
    @pytest.mark.skipif(not __import__('green_evaluator').A2A_AVAILABLE, reason="A2A dependencies not available")
    def test_agent_card_endpoint(self, evaluator):
        from green_evaluator import A2AServer
        from fastapi.testclient import TestClient
        
        server = A2AServer(evaluator, card_url="http://localhost:8000")
        client = TestClient(server.app)
        
        response = client.get("/card")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "capabilities" in data
        assert "miniwob_benchmark" in data["capabilities"]
        assert "workarena_benchmark" in data["capabilities"]
        assert "webarena_benchmark" in data["capabilities"]
