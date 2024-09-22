import os
import typer
import json
import time
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
api_key = os.getenv("API_KEY")
provider = os.getenv("PROVIDER", "groq") # "groq"
base_url = os.getenv("BASE_URL", "https://api.groq.com/openai/v1") # "https://api.groq.com/openai/v1"
model_id = os.getenv("MODEL_ID", "llama-3.1-70b-versatile") # "llama-3.1-70b-versatile"
model_context_window = os.getenv("CONTEXT_WINDOW", 8000) # 8000 for llama-3.1-70b-versatile

# Configure OpenAI client
client = OpenAI(api_key=api_key, base_url=base_url)

console = Console()
app = typer.Typer()

def make_api_call(messages, max_tokens, is_final_answer=False):
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt == 2:
                if is_final_answer:
                    return {"title": "Error", "content": f"Failed to generate final answer after 3 attempts. Error: {str(e)}"}
                else:
                    return {"title": "Error", "content": f"Failed to generate step after 3 attempts. Error: {str(e)}", "next_action": "final_answer"}
            time.sleep(1)  # Wait for 1 second before retrying

def generate_response(prompt):
    messages = [
        {
            "role": "system", 
            "content": """You are an expert AI assistant that explains your reasoning step by step. 
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
        },
        {
            "role": "user", 
            "content": prompt
        },
        {
            "role": "assistant", 
            "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."
        }
    ]
    
    steps = []
    step_count = 1
    total_thinking_time = 0
    
    while True:
        start_time = time.time()
        step_data = make_api_call(messages, 300)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time
        
        steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'], thinking_time))
        
        messages.append({"role": "assistant", "content": json.dumps(step_data)})
        
        if step_data['next_action'] == 'final_answer' or step_count > 25: # Maximum of 25 steps to prevent infinite thinking time. Can be adjusted.:
            # Add verification step
            messages.append({"role": "user", "content": "Before giving the final answer, please verify your result using a different method and explain any discrepancies."})
            verification_data = make_api_call(messages, 300)
            steps.append((f"Step {step_count}: Verification", verification_data['content'], thinking_time))
            messages.append({"role": "assistant", "content": json.dumps(verification_data)})
            break
        
        step_count += 1

        # Yield after each step without Streamlit-specific formatting
        yield step_data['title'], step_data['content'], thinking_time

    # Generate final answer
    messages.append({"role": "user", "content": "Please provide the final answer based on your reasoning above."})
    
    start_time = time.time()
    final_data = make_api_call(messages, 200, is_final_answer=True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time
    
    steps.append(("Final Answer", final_data['content'], thinking_time))

    yield "Final Answer", final_data['content'], thinking_time

@app.command()
def main():
    console = Console()
    console.print(Panel("g1: Using " + model_id + " on Groq to create o1-like reasoning chains", expand=False))
    console.print("Fork of Open source repository: https://github.com/bklieger-groq\n")
    console.print("This is an early prototype of using prompting to create o1-like reasoning chains to improve output accuracy. It is not perfect and accuracy has yet to be formally evaluated. It is powered by Groq so that the reasoning step is fast!")

    # Prompt the user for input after the program starts
    query = typer.prompt("\nEnter your query for the AI assistant")

    console.print(f"\nQuery: {query}\n")
    console.print("Generating response...\n")

    total_thinking_time = 0
    step_count = 1

    for title, content, thinking_time in generate_response(query):
        total_thinking_time += thinking_time

        if title == "Final Answer":
            console.print(Panel(Markdown(f"### {title}\n\n{content}"), expand=False))
        else:
            console.print(Panel(Markdown(f"## Step {step_count}: {title}\n\n{content}"), expand=False))
            step_count += 1

    console.print(f"\nTotal thinking time: {total_thinking_time:.2f} seconds")

if __name__ == "__main__":
    app()