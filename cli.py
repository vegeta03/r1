import os
import typer
import json
import time
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables and configure OpenAI client
load_dotenv()
api_key = os.getenv("API_KEY")
provider = os.getenv("PROVIDER", "groq")
base_url = os.getenv("BASE_URL", "https://api.groq.com/openai/v1")
model_id = os.getenv("MODEL_ID", "llama-3.1-70b-versatile")
model_context_window = os.getenv("CONTEXT_WINDOW", 8000)

client = OpenAI(api_key=api_key, base_url=base_url)

console = Console()
app = typer.Typer()

# Constants
MAX_STEPS = 25
VERIFICATION_PROMPT = "Before giving the final answer, please verify your result using a different method and explain any discrepancies."
FINAL_ANSWER_PROMPT = "Please provide the final answer based on your reasoning above."

SYSTEM_MESSAGE = """You are an expert AI assistant that explains your reasoning step by step. 
    For each step, provide a title that describes what you're doing in that step, along with the content. 
    Decide if you need another step or if you're ready to give the final answer. 
    Respond in JSON format with 'title', 'content', and 'next_action' (either 'continue' or 'final_answer') keys. 
    USE AS MANY REASONING STEPS AS NECESSARY TO ENSURE ACCURACY. 
    ALWAYS DOUBLE-CHECK YOUR RESULTS AND CONSIDER EDGE CASES. 
    IF YOU FIND A DISCREPANCY IN YOUR REASONING, EXPLAIN IT AND CORRECT IT. 
    BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. 
    IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. 
    CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. 
    FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. 
    WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. 
    DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES.
    IF YOU ARE UNSURE OF SOMETHING, SAY SO. DO NOT MAKE UP FACTS.
    Example of a valid JSON response:
    ```json
    {
        "title": "Identifying Key Information",
        "content": "To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves...",
        "next_action": "continue"
    }
    ```
"""

def append_message(messages, role, content):
    messages.append({"role": role, "content": content})
    return messages

def make_api_call(messages, max_tokens, is_final_answer=False):
    common_params = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "response_format": {"type": "json_object"}
    }
    
    for attempt in range(3):
        try:
            response = client.chat.completions.create(**common_params)
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt == 2:
                error_message = f"Failed to generate {'final answer' if is_final_answer else 'step'} after 3 attempts. Error: {str(e)}"
                return {"title": "Error", "content": error_message, "next_action": "final_answer" if not is_final_answer else None}
            time.sleep(1)

def create_initial_messages(prompt):
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
    ]
    return messages

def process_step(messages, step_count):
    start_time = time.time()
    step_data = make_api_call(messages, 300)
    thinking_time = time.time() - start_time
    
    title = f"Step {step_count}: {step_data['title']}"
    messages = append_message(messages, "assistant", json.dumps(step_data))
    
    return step_data, title, thinking_time, messages

def generate_response(prompt):
    messages = create_initial_messages(prompt)
    step_count = 1
    
    while True:
        step_data, title, thinking_time, messages = process_step(messages, step_count)
        yield title, step_data['content'], thinking_time
        
        if step_data['next_action'] == 'final_answer' or step_count >= MAX_STEPS:
            messages = append_message(messages, "user", VERIFICATION_PROMPT)
            verification_data, verification_title, verification_time, messages = process_step(messages, step_count + 1)
            yield verification_title, verification_data['content'], verification_time
            break
        
        step_count += 1

    messages = append_message(messages, "user", FINAL_ANSWER_PROMPT)
    final_data, final_title, final_thinking_time, messages = process_step(messages, "Final Answer")
    yield final_title, final_data['content'], final_thinking_time

def process_and_print_response(query):
    console.print(f"\nQuery: {query}\n")
    console.print("Generating response...\n")

    total_thinking_time = 0
    step_count = 1

    for title, content, thinking_time in generate_response(query):
        total_thinking_time += thinking_time

        if title == "Final Answer":
            console.print(Panel(Markdown(f"### {title}\n\n{content}"), expand=False))
        else:
            console.print(Panel(Markdown(f"## {title}\n\n{content}"), expand=False))
            step_count += 1

    console.print(f"\nTotal thinking time: {total_thinking_time:.2f} seconds")

@app.command()
def main():
    console.print(Panel(f"g1: Using {model_id} on Groq to create o1-like reasoning chains", expand=False))
    console.print("Fork of Open source repository: https://github.com/bklieger-groq\n")
    console.print("This is an early prototype of using prompting to create o1-like reasoning chains to improve output accuracy. It is not perfect and accuracy has yet to be formally evaluated. It is powered by Groq so that the reasoning step is fast!")

    query = typer.prompt("\nEnter your query for the AI assistant")
    process_and_print_response(query)

if __name__ == "__main__":
    app()