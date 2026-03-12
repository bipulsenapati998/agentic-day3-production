# agentic-day3-production
Prompt Versioning &amp; Prompt injection
## Objective

Build a **minimal customer support agent** from scratch and harden it using the **production foundations**:

- **Prompt templates** treated as code (YAML + Git)
- **Prompt injection defense** (3-layer model)
- **Error handling with retries** (rate limits, timeouts, context overflow)
- **Circuit breaker** to stop cascading failures
- **Structured logging + cost tracking**

## How to run the App 
1. Clone the Repository:
```
    git clone https://github.com/bipulsenapati998/agentic-day3-production.git
```
2. Create & Activate the virtual environment:
```
    Use conda	
    conda create -n llms python=3.11 && conda activate llms  
```
3. Update the .env file with your OpenAI API key:
```
    OPENAI_API_KEY=<your_actual_api_key_here>
```
4. Install Dependencies:
```
    pip install -r requirement.txt
```
5. Run the Application:
```
   python app.py 
```