# !pip install llama-cpp-python

from llama_cpp import Llama

system_prompt = """
ROLE: 
    You are an Business SME and Agile Product Owner in regulated enterprise environments who is championed in creating User Stories.
    Your responsibility is to create high-quality user stories that are:
    -   Business-outcome driven
    -   Technically feasible
    -   Testable with measurable acceptance criteria
    -   Aligned to enterprise governance standards
    You must not invent requirements, business rules, or assumptions that are not explicitly stated.
    If information is missing or ambiguous, you must clearly flag it.
"""

user_prompt = """
Business Context:
    The objective is to demonstrate how an LLM can bridge the gap between quantitative market data (Yahoo Finance) and qualitative strategic reasoning. 
    By analyzing price movements in ADNOC’s flagship Murban Crude ($BZ=F) against benchmarks like Brent ($LCO=F), the demo provides real-time "Spread Intelligence" and anomaly detection.
    -   Industry: Energy trading sector with primary business in Crude and processed oil products. 
    -   Business Process: 
    -   Primary Actor: Front office
    -   Downstream Systems:
    -   Regulatory Constraints (if any):
    -   Data Sensitivity Level: publicly available data.z

Input Requirement:
    - Market Data & Feature Engineering:
        As a Trading Analyst, I want to fetch 30 days of historical price data for Murban ($BZ=F) and Brent ($LCO=F) so that the AI can identify price spreads and volatility trends.
        Requirements:
            Use yfinance to download daily OHLC data.
            Calculate a simple Spread column ($BZ=F - $LCO=F).
            Compute 5-day and 20-day Moving Averages.
            Handle missing data gracefully to ensure the LLM prompt is clean.

    - Optimized Local Inference Setup
        As a Solution Architect, I want to initialize a Hugging Face model optimized for the M5 chip so that the demo runs without latency or dependency on external cloud APIs.
        Requirements:
            Auto-detect if torch.backends.mps.is_available() is true.
            Load a model (e.g., meta-llama/Llama-3.1-8B-Instruct) using bitsandbytes 4-bit quantization.
            Implement a local caching mechanism to prevent re-downloading the model weights (e.g., /Users/shared/huggingface/hub).

    - Context-Aware Market Reasoning
        As an Energy Trader, I want the LLM to analyze the calculated technical data and generate a "Market Signal Brief" so that I can identify potential hedging opportunities.
        Requirements:
            Construct a prompt template that injects the latest price, spread change, and moving averages.
            Prompt Constraint: The LLM must output in a professional, concise "Trader Talk" format (e.g., "Murban-Brent spread widening; watch for resistance at $X.XX").
            Implement a "Risk Guardrail" instruction to ensure the AI explicitly states it is not providing financial advice.

    - Executive Dashboard Interface
        As an ADNOC Executive, I want to see a visual dashboard with charts and AI-generated insights side-by-side so that I can quickly assess market conditions.
        Requirements:
            Use Streamlit for a dark-themed, professional UI.
            Plot the Murban vs. Brent spread using Plotly or Altair.
            Create a sidebar for ticker selection (e.g., $BZ=F, $LCO=F, $CL=F).
            Display the LLM's "Market Reasoning" in a dedicated text container with a "Thinking..." spinner during inference.
    
    - Technical Constraints:
        - Language - Python in Conda environment.
        - Data Source - yfinance.
        - LLM Engine - Hugging Face (transformers)
        - Hardware - Apple Macbook Pro with M5 chip and 16GB unified memory.
        - UI framework - Streamlit

User Story Structure (MANDATORY):
    -   Title
    -   User Story (As a / I want / So that)
    -   Acceptance Criteria (Given-When-Then)

Acceptance Criteria Rules:
    -   Be atomic (one condition per criterion) and unambiguous
    -   Be measurable and testable
    -   Avoid subjective words (fast, intuitive, easy)
    -   Use Given-When-Then format
    -   Cover:
        - happy path
        - Error/negative path
        - Edge cases (data, timing, integration)

Non-Functional Requirements (Minimum Coverage):
    Explicitly consider:
        - Performance
        - Security & access control
        - Auditability & logging
        - Data quality & validation
        - Availability & resilience
        - Regulatory compliance

Creativity Boundaries:
    Creativity is allowed only in:
        - Edge cases
        - Error handling
        - Alternative user flows
    Creativity is not allowed in:
        - Business rules
        - Regulatory logic
        - Data definitions
        - Control frameworks
        
Quality Gates (Self-Validation):
    Before finalising, validate that:
        - Stories meet INVEST principles
        - Acceptance criteria are fully testable
        - Scope is clearly bounded
        - No assumptions contradict business context
        - Ambiguities are explicitly flagged

Output Format:
    Present the output in:
        - Clear, professional language
        - Organise output under two headings: Functional User Stories and Technical User Stories.
        - Use Markdown format:
        #   Title
        ##  User Story - As a / I want / So that
        ##  Acceptance Criteria- [Given], [When], [Then]
    
"""

llm = Llama.from_pretrained(
        repo_id="MaziyarPanahi/gemma-3-12b-it-GGUF",
        filename="gemma-3-12b-it.Q6_K.gguf",
        #local_dir = "~/llm/models",
        #local_dir_use_symlinks=True,
        chat_format="llama-2",
        temperature=0.3,    # Keeps stories consistent, avoids hallucinated features
        n_gpu_layers=99,
        top_p=0.9,          # Allows structured creativity without losing control
        verbose=True,
        n_ctx=4096,
        max_tokens=-1,
        presence_penalty_field = 0.2,  # Prevents unnecessary new ideas mid-story
        frequency_penalty_field = 0.3, # Avoids repetitive acceptance criteria

)



response  = llm.create_chat_completion(
      messages = [
        {
            "role": "system",
            "content": "You are an Business SME who is championed in creating User Stories."
        },
        {
            "role": "user",
            "content": user_prompt,
          }
      ],

)

#print(json.dumps([{k: response[k]} for k in response], indent=4))
#print(json.dumps(response, indent=4))
content = response["choices"][0]["message"]["content"]
print(content)




