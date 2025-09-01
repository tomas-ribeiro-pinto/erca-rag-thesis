import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval.dataset import EvaluationDataset, Golden

from tests.evaluation_chatbot import EvaluationChatbot

dataset = EvaluationDataset()
dataset.pull(alias="erca_qa_dataset")

chatbot = EvaluationChatbot(model="anthropic/claude-sonnet-4", use_rag=True)

new_goldens = []

for i, golden in enumerate(dataset.goldens):
    print(f"Processing input: {i+1}/{len(dataset.goldens)}")
    output, context = chatbot.generate(golden.input)
    new_golden = Golden(
        input=golden.input,
        expected_output=golden.expected_output,
        actual_output=output,
        retrieval_context=context,
        context=context
    )
    new_goldens.append(new_golden)

new_dataset = EvaluationDataset(goldens=new_goldens)
filename = f"test_dataset_{chatbot.model_name.split('/')[-1]}"
new_dataset.save_as(file_type="csv", directory="./tests/erca-test-datasets", file_name=filename)