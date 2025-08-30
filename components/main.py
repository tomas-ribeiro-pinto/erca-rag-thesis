import os
import asyncio
import warnings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from rag_generator import RagGenerator
from rag_retriever import RagRetriever

warnings.filterwarnings("ignore", category=UserWarning, module="milvus_lite")

async def main():
    llm = ChatOllama(
        model = "llama3.1",
        temperature = 0.8,
        num_predict = 4096,
        # other params ...
    )

    documents_path = "./documents/"
    db_path = "./rag_milvus.db"

    if not os.path.exists(db_path):
        if os.path.exists(documents_path) and os.path.isdir(documents_path):
            retriever = RagRetriever()
            await retriever.save_documents(documents_path)
        else:
            print(f"Error: File path {documents_path} is not a valid directory.")
            return
    else:
        retriever = RagRetriever(db_path)
        print(f"Using existing vector store at {db_path}")
    generator = RagGenerator(llm)
    
    try:
        system_template = (
            "You are an expert assistant helping a university student with questions about image processing. "
            "Use the provided context from lecture notes, slides, and other materials to answer clearly and concisely. "
            "If you can't answer or it is out of context, say 'I don't know'."
            "Ensure your response is suitable for a student, informative, and does not exceed 4096 tokens."
            "Context: {context}"
        )
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_template), ("user", "{user_prompt}")]
        )

        # response = retriever.vector_store.similarity_search_with_score(
        #     query="Binary image", k=15
        # )
        # print("Retrieved documents:", response)


        while True:
            user_prompt = input("\nAsk a question (or type 'exit' to quit): ")
            if user_prompt.lower() == 'exit':
                break
            
            docs = retriever.invoke(user_prompt, k=5)
            prompt = prompt_template.invoke({"user_prompt": user_prompt, "context": "\n\n".join([doc.page_content for doc in docs])})
            response = await asyncio.to_thread(generator.invoke, prompt)
            print(response.content)
            if not docs:
                print("No relevant documents found.")
            else:
                print(f"Found {len(docs)} relevant documents:")
            #print("Retrieved documents:", "\n\n".join([doc.page_content for doc in docs]))
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return  # Exit early on error

async def shutdown():
    """Cleanup tasks"""
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [t.cancel() for t in tasks]
    await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
    finally:
        try:
            loop.run_until_complete(shutdown())
        finally:
            loop.close()