class RagGenerator:
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, prompt):
        response = self.llm.invoke(prompt)
        return response