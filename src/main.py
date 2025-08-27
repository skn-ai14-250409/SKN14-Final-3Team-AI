from src.orchestrator import Orchestrator

DOC_PATH = "./은행업감독규정 (금융위원회고시)(제2025-7호)(20250305).pdf"

def run(option):
    orchestrator = Orchestrator()
    path = DOC_PATH

    if option == 1:
        prompt = input("질문을 입력하세요: ")
        response = orchestrator.run_workflow(prompt)
        print("\n[Answer]\n", response)


    elif option == 2:
        category = "industry_policies_and_regulations"
        # category = "company_rules"
        response = orchestrator.upload_docs_to_rag(path, category)
        print("\n[Uploaded IDs]\n", response)

if __name__ == "__main__":
    option = 1
    run(option)