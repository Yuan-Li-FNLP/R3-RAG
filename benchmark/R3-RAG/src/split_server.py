import argparse
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import uvicorn
import re, pdb
parser = argparse.ArgumentParser(description="Start the checker service.")
parser.add_argument('--host', type=str, required=True, help="Host address (e.g., 10.176.58.108)")
parser.add_argument('--port', type=int, required=True, help="Port number (e.g., 8002)")
parser.add_argument('--model_path', type=str, default="/opt/nas/p/models/Qwen_models/Qwen2.5-72B-Instruct", help="Path to the model")
parser.add_argument('--tp', type=int, default=8, help="GPU numbers")

class QueryRequest(BaseModel):
    question: str
    answer: str
    standard_answers: list
class SplitRequest(BaseModel):
    querys: list
def split_answer(response_text):
    result = {}
    lines = response_text.strip().split('\n')
    for line in lines:
        if line.startswith("Correctness analysis:"):
            result['Correctness analysis'] = line.replace("Correctness analysis:", "").strip()
        elif line.startswith("Final answer:"):
            result['Final answer'] = line.replace("Final answer:", "").strip()
    return result

def split_prompt(query):
    return f"""
You are an intelligent query decomposer. Based on the user's query, determine whether it needs to be broken down into smaller sub-questions. If it does, provide a list of sub-questions; otherwise, return a list containing only the original query.

Example 1:
Input: "Who is the director of 'Danger: Diabolik' and what is the release year of the film?"
Output: ["Who is the director of 'Danger: Diabolik'?", "What is the release year of the film 'Danger: Diabolik'?"]

Example 2:
Input: "What is the capital of France?"
Output: ["What is the capital of France?"]

Example 3:
Input: "Explain the theory of relativity."
Output: ["Explain the theory of relativity."]

Example 4:
Input: "List all the planets in the solar system and describe their main characteristics."
Output: ["List all the planets in the solar system.", "Describe the main characteristics of each planet."]

Now, given the following input query, provide the corresponding output list:

Input: "{query}"
Output:
"""

def split_query(querys : list):
    conversations = [
        [
            {"role": "system","content": "You are a helpful assistant"},
            {"role": "user","content": split_prompt(query)}
        ]  for query in querys
    ]
    outputs = llm.chat(conversations, sampling_params)
    outputs = [output.outputs[0].text.strip() for output in outputs]
    matches = [re.findall(r'"(.*?)"', output) for output in outputs]
    return matches

@app.post("/split_query")
async def split_query_endpoint(request: SplitRequest):
    response=split_query(request.querys)
    return {"response": response}

if __name__ == "__main__":
    app = FastAPI()
    args = parser.parse_args()

    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
    print(f"Loading model from {args.model_path}...")
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tp)

    uvicorn.run(app,host=args.host, port=args.port)