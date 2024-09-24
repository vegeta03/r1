# g1: Using models from Groq to Create better or o1-like Reasoning Chains

## Overview

r1 is an experimental project that leverages models on Groq to create better or o1-like reasoning chains. This prototype uses advanced prompting strategies to enhance the reasoning capabilities of the LLM (Large Language Model), enabling it to solve logical problems that typically challenge leading models. Unlike r1, all reasoning tokens are displayed, and the app utilizes an open-source model.

The goal of r1 is to inspire the open-source community to develop new strategies for producing better or o1-like reasoning. This experiment demonstrates the power of prompting reasoning in visualized steps, rather than being a direct comparison or full replication of o1, which employs different techniques. OpenAI's o1 is trained with large-scale reinforcement learning to reason using Chain of Thought, achieving state-of-the-art performance on complex PhD-level problems.

r1 showcases the potential of prompting alone to address straightforward LLM logic issues, such as the Strawberry problem, allowing existing open-source models to benefit from dynamic reasoning chains and an improved interface for exploring them.

## How It Works

r1, powered by models from Groq Cloud, creates reasoning chains that function as a dynamic Chain of Thought, enabling the LLM to "think" and solve logical problems that typically stump leading models.

### Reasoning Process

1. **Step-by-Step Reasoning**: At each step, the LLM can choose to continue to another reasoning step or provide a final answer. Each step is titled and visible to the user.
2. **System Prompt**: The system prompt includes tips for the LLM, such as “include exploration of alternative answers” and “use at least 3 methods to derive the answer”.
3. **Combining Techniques**: The reasoning ability of the LLM is improved by combining Chain-of-Thought with the requirement to try multiple methods, explore alternative answers, question previous draft solutions, and consider the LLM’s limitations.

### Credits

Started as a fork of Open source repository [g1](https://github.com/bklieger-groq) by [Benjamin Klieger](https://x.com/benjaminklieger). Modified by [Shyam Sreenivasan](https://x.com/vegeta_shyam).
