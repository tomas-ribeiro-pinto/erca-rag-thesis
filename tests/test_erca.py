import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric, ContextualPrecisionMetric, ContextualRelevancyMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
import time

from tests.evaluation_chatbot import EvaluationChatbot

dataset = EvaluationDataset()
dataset.add_test_cases_from_csv_file(
    # file_path is the absolute path to you .csv file
    file_path="./tests/erca-test-datasets/test_dataset_gpt-5.csv",
    input_col_name="input",
    actual_output_col_name="actual_output",
    expected_output_col_name="expected_output",
    context_col_name="context",
    context_col_delimiter= ";",
    retrieval_context_col_name="retrieval_context",
    retrieval_context_col_delimiter= ";"
)

print(f"Loaded {len(dataset.test_cases)} test cases for evaluation")

@pytest.mark.parametrize(
    "test_case",
    dataset.test_cases,
    ids=[f"test_case_{i+1:03d}" for i in range(len(dataset.test_cases))]
)
def test_rag(test_case: LLMTestCase):
    """Test individual RAG test case with progress tracking and rate limiting"""
    
    # Get the test case index for progress tracking
    test_index = dataset.test_cases.index(test_case) + 1
    total_tests = len(dataset.test_cases)
    
    print(f"\n=== Processing test case {test_index}/{total_tests} ===")
    print(f"Input: {test_case.input[:100]}...")
    
    try:
        # Add delay every 10 tests to prevent rate limiting
        if test_index > 1 and test_index % 10 == 1:
            print(f"Pausing for 5 seconds to avoid rate limiting...")
            time.sleep(5)
        
        # Add small delay between each test to be conservative
        if test_index > 1:
            time.sleep(1)
        
        print(f"Initializing evaluation model...")
        # Use OpenAI GPT-4o-mini through OpenRouter for reliable JSON evaluation
        evaluation_model = EvaluationChatbot(model="openai/gpt-4o-mini")

        print(f"Creating metrics...")
        answer_relevancy_metric = AnswerRelevancyMetric(model=evaluation_model)
        faithfulness_metric = FaithfulnessMetric(model=evaluation_model)
        contextual_relevancy_metric = ContextualRelevancyMetric(model=evaluation_model)
        contextual_precision_metric = ContextualPrecisionMetric(model=evaluation_model)
        contextual_recall_metric = ContextualRecallMetric(model=evaluation_model)
        hallucination_metric = HallucinationMetric(model=evaluation_model)

        metrics = [answer_relevancy_metric, faithfulness_metric, contextual_relevancy_metric, 
                  contextual_precision_metric, contextual_recall_metric, hallucination_metric]
        
        print(f"Running evaluation for test case {test_index}...")
        assert_test(test_case, metrics)
        print(f"✓ Test case {test_index}/{total_tests} completed successfully")
        
    except AttributeError as e:
        if "'NoneType' object has no attribute 'find'" in str(e):
            print(f"✗ Test case {test_index}/{total_tests} failed: Model returned None")
            print(f"This suggests the evaluation model isn't responding properly")
            print(f"Test input was: {test_case.input}")
            # Skip this test instead of failing the entire suite
            pytest.skip(f"Model returned None for test case {test_index}")
        else:
            raise
    except Exception as e:
        print(f"✗ Test case {test_index}/{total_tests} failed with error: {str(e)}")
        print(f"Test input was: {test_case.input}")
        # Re-raise the exception so pytest marks this test as failed but continues with others
        raise

