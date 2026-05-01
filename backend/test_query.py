import traceback
from services.rag import hybrid_retrieval_and_answer

try:
    print("Testing query...")
    res = hybrid_retrieval_and_answer("summarize the file")
    print(res)
except Exception as e:
    traceback.print_exc()
