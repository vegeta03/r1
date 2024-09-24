# r1: Using models from Groq or SambaNova to Create o1-like or better Reasoning Chains

## Overview

r1 is an experimental project that leverages models on Groq or SambaNova to create o1-like or better reasoning chains. This prototype uses advanced prompting strategies to enhance the reasoning capabilities of the LLM (Large Language Model), enabling it to solve logical problems that typically challenge leading models.

The goal of r1 is to inspire the open-source community to develop new strategies for producing o1-like or better reasoning. This experiment demonstrates the power of prompting reasoning in visualized steps, rather than being a direct comparison or full replication of o1, which employs different techniques. OpenAI's o1 is trained with large-scale reinforcement learning to reason using Chain of Thought, achieving state-of-the-art performance on complex PhD-level problems.

r1 showcases the potential of prompting alone to address straightforward LLM logic issues, such as the Strawberry problem, allowing existing open-source models to benefit from dynamic reasoning chains and an improved interface for exploring them.

## How It Works

r1, powered by models from Groq Cloud, creates reasoning chains that function as a dynamic Chain of Thought, enabling the LLM to "think" and solve logical problems that typically stump leading models.

### Reasoning Process

1. **Step-by-Step Reasoning**: At each step, the LLM can choose to continue to another reasoning step or provide a final answer. Each step is titled and visible to the user.
2. **System Prompt**: The system prompt includes tips for the LLM, such as “include exploration of alternative answers” and “use at least 3 methods to derive the answer”.
3. **Combining Techniques**: The reasoning ability of the LLM is improved by combining Chain-of-Thought with the requirement to try multiple methods, explore alternative answers, question previous draft solutions, and consider the LLM’s limitations.

## Technical Details

### Environment Setup

1. **Clone the Repository**:

    ```sh
    git clone https://github.com/vegeta03/r1.git
    cd r1
    ```

2. **Create and Configure Environment Variables**:
    - Copy the sample environment file and update it with your API key and other configurations.

    ```sh
    cp SAMPLE.env .env
    ```

    - Edit the `.env` file to include your API key and other necessary details.

3. **Install Dependencies**:
    - Ensure you have Python 3.12 installed.
    - Install the required Python packages.

    ```sh
    pip install --no-cache-dir -r requirements.txt
    ```

### Running the Application

1. **Build and Run with Podman/Docker**:
    - Build the container image.

    ```sh
    podman build -t r1-cli .
    ```

    - Run the container.

    ```sh
    podman run -it --rm -e API_KEY=your_api_key r1-cli
    ```

2. **Direct Execution**:
    - Run the CLI application directly.

    ```sh
    python cli.py
    ```

### Code Structure

- **cli.py**: The main CLI application that interacts with the Groq API to generate reasoning chains.
  - **Environment Variables and Configuration**: Loads environment variables and configures the OpenAI client.
  - **Constants and System Messages**: Defines constants and system messages for the AI model.
  - **Utility Functions**: Implements utility functions for API calls and message handling.
    - `append_message`: Appends a new message to the existing list of messages.
    - `make_api_call`: Makes an API call to the Groq service using the OpenAI client.
    - `create_initial_messages`: Creates the initial set of messages for the conversation.
    - `process_step`: Processes a single step in the reasoning chain.
    - `generate_response`: Orchestrates the entire reasoning process.
    - `process_and_print_response`: Handles the entire process of generating a response to the user's query.
  - **Main Logic**: Contains the main logic for generating and processing AI responses.
  - **CLI Interface**: Sets up the CLI interface using Typer.

- **Containerfile**: Defines the container setup for the application.
  - Uses Python 3.12-slim as the base image.
  - Copies the application code into the container.
  - Installs the required Python packages.
  - Sets environment variables and exposes port 80.
  - Defines the command to run the CLI application.

- **requirements.txt**: Lists the Python dependencies required for the application.
  - `typer`
  - `openai`
  - `rich`
  - `python-dotenv`

- **.dockerignore**: Specifies files and directories to ignore when building the Podman/Docker image.

- **.gitignore**: Specifies files and directories to ignore in the Git repository.

### Example Usage

1. **Start the CLI Application**:

    ```sh
    python cli.py
    ```

2. **Enter a Query**:
    - When prompted, enter your query for the AI assistant.

    ```sh
    Enter your query for the AI assistant: How many r's are there in Strawberry?
    ```

3. **View the Response**:
    - The application will generate a step-by-step reasoning chain and provide the final answer.

### Credits

Started as a fork of Open source repository [g1](https://github.com/bklieger-groq) by [Benjamin Klieger](https://x.com/benjaminklieger). Modified by [Shyam Sreenivasan](https://x.com/vegeta_shyam).
