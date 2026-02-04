# WTI-Brent Intelligence Copilot

AI-powered trading analysis application for WTI-Brent crude oil spread analysis.

## Features

- **Real-time Market Data**: Fetches WTI and Brent crude oil prices from Yahoo Finance
- **Spread Analysis**: Calculates WTI-Brent spread with configurable moving averages
- **AI-Powered Insights**: Generates market signals using local LLM inference
- **Interactive Dashboard**: Streamlit-based UI with Plotly charts and dark theme
- **Config-Driven**: All parameters configurable via YAML configuration file
- **Docker Support**: Production-ready containerization with health checks
- **Dual-LLM Architecture**: Separate models for analysis and signal extraction
- **Flexible Model Backend**: Switch between llama-cpp-python (GGUF) and HuggingFace Transformers

## Quick Start

### Local Development

1. Clone and setup:
   ```bash
   git clone <repository-url>
   cd murban-intelligence-copilot
   conda env create -f environment.yml
   conda activate murban-copilot
   ```

2. Run the dashboard:
   ```bash
   PYTHONPATH=src streamlit run src/murban_copilot/interface/streamlit_app.py
   ```

### Using Docker

```bash
docker-compose up --build
```

Dashboard available at `http://localhost:8501`

## Configuration

The application is fully config-driven via `config/llm_config.yaml`:

```yaml
llm:
  defaults:
    n_ctx: 4096
    n_gpu_layers: -1
    verbose: false

  # Analysis model (generates comprehensive market analysis)
  analysis:
    model_type: "llama"
    model_repo: "MaziyarPanahi/gemma-3-12b-it-GGUF"
    model_file: "gemma-3-12b-it.Q6_K.gguf"
    inference:
      max_tokens: 2048
      temperature: 0.7
      top_p: 0.9
      top_k: 50
      frequency_penalty: 0.3
      presence_penalty: 0.1

  # Extraction model (extracts structured signals)
  extraction:
    model_type: "transformers"
    model_repo: "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    task: "sentiment-analysis"
    device: "cpu"
    inference:
      max_tokens: 150
      temperature: 0.1
      top_p: 0.65
      top_k: 25

market_data:
  wti_ticker: "CL=F"
  brent_ticker: "BZ=F"
  timeout: 60
  max_retries: 3

analysis:
  short_ma_window: 5
  long_ma_window: 20
  outlier_threshold: 3.0

signal:
  default_signal: "neutral"
  default_confidence: 0.5
  bullish_keywords: ["bullish", "upward", "positive", "buy"]
  bearish_keywords: ["bearish", "downward", "negative", "sell"]

ui:
  min_historical_days: 7
  max_historical_days: 90
  default_historical_days: 30
```

**Note:** `wti_ticker` and `brent_ticker` are required in the configuration file.

### Model Type Switching

The application supports two model backends:

| Model Type | Backend | Use Case | Example Models |
|------------|---------|----------|----------------|
| `llama` | llama-cpp-python | Text generation with GGUF models | Gemma, Llama, Mistral |
| `transformers` | HuggingFace Transformers | Sentiment classification | FinBERT, DistilRoBERTa |

#### Using Llama/GGUF Models (Default for Analysis)

```yaml
analysis:
  model_type: "llama"
  model_repo: "MaziyarPanahi/gemma-3-12b-it-GGUF"
  model_file: "gemma-3-12b-it.Q6_K.gguf"
  n_ctx: 4096
  n_gpu_layers: -1
```

#### Using HuggingFace Transformers (Recommended for Extraction)

```yaml
extraction:
  model_type: "transformers"
  model_repo: "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
  task: "sentiment-analysis"
  device: "cpu"  # Options: "cpu" | "cuda" | "mps"
```

#### Available Sentiment Models

| Model | Description | Size |
|-------|-------------|------|
| `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` | DistilRoBERTa fine-tuned on financial news | ~66M params |
| `ProsusAI/finbert` | FinBERT - Financial sentiment | ~110M params |
| `yiyanghkust/finbert-tone` | FinBERT for financial tone | ~110M params |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MURBAN_CONFIG` | Path to application config file | `config/llm_config.yaml` |
| `MURBAN_LLM_CONFIG` | Path to LLM config (legacy) | `config/llm_config.yaml` |
| `MURBAN_LOG_LEVEL` | Logging level | `INFO` |

## Architecture

```
src/murban_copilot/
├── domain/              # Pure business logic
│   ├── entities.py      # MarketData, SpreadData, MovingAverages, MarketSignal
│   ├── config.py        # Configuration dataclasses
│   ├── spread_calculator.py
│   ├── validators.py
│   └── exceptions.py
├── infrastructure/      # External integrations
│   ├── market_data/     # Yahoo Finance client
│   ├── llm/             # LLM inference (llama-cpp + transformers)
│   │   ├── llm_client.py        # Llama/GGUF models
│   │   ├── transformers_client.py  # HuggingFace models
│   │   └── factory.py           # Model factory
│   ├── config/          # YAML configuration loader
│   ├── health/          # Health check infrastructure
│   └── logging/         # Structured logging
├── application/         # Use cases
│   ├── fetch_market_data.py
│   ├── analyze_spread.py
│   └── generate_signal.py
└── interface/           # Streamlit dashboard
    └── streamlit_app.py
```

## Technical Details

### Spread Calculation

- **Spread**: WTI Close - Brent Close ($/barrel)
- **5-Day MA**: Short-term moving average for trend detection (configurable)
- **20-Day MA**: Long-term moving average for trend confirmation (configurable)
- **Outlier Handling**: Median Absolute Deviation (MAD) for robust statistics

### Dual-LLM Architecture

The application uses a two-step LLM approach with configurable model backends:

1. **Analysis Model**: Generates comprehensive market analysis
   - Recommended: Large generative LLM (Gemma, Llama)
   - Backend: `llama` (llama-cpp-python with GGUF models)
   - Temperature: 0.7 (creative, varied output)

2. **Extraction Model**: Extracts structured signal and confidence
   - Recommended: Domain-specific sentiment classifier
   - Backend: `transformers` (HuggingFace models like DistilRoBERTa)
   - Provides: bullish/bearish/neutral signal with confidence score

This separation allows each model to be optimized for its specific task.

### Data Handling

- **Missing Data**: Forward fill (≤2 days), linear interpolation (longer gaps)
- **Retry Logic**: Exponential backoff with configurable retries
- **Caching**: LLM responses cached locally for performance

## Docker Deployment

### Building and Running

```bash
# Build and run with Docker Compose (recommended)
docker-compose up --build

# Or build manually
docker build -t murban-copilot .
docker run -p 8501:8501 murban-copilot
```

### Custom Configuration

```bash
# Run with custom config file
docker run -p 8501:8501 \
  -v $(pwd)/config:/app/config:ro \
  -e MURBAN_CONFIG=/app/config/llm_config.yaml \
  murban-copilot
```

### Health Check

The application exposes a health endpoint at `/_stcore/health`. The docker-compose.yml includes:

- Health checks with 60s start period
- Automatic restart (`unless-stopped`)
- Memory limits (4GB limit, 2GB reserved)
- Persistent volumes for cache and logs

```yaml
services:
  murban-copilot:
    build: .
    ports:
      - "8501:8501"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    restart: unless-stopped
```

## Running Tests

### All Tests
```bash
python -m pytest tests/ -v
```

### Unit Tests Only
```bash
python -m pytest tests/unit/
```

### Integration Tests (requires network)
```bash
python -m pytest tests/integration/
```

### With Coverage Report
```bash
python -m pytest tests/ --cov=src/murban_copilot --cov-report=html
open htmlcov/index.html
```

## Dependencies

### Core
- Python 3.10+
- pandas >= 2.0
- numpy >= 1.24
- plotly >= 5.0
- streamlit >= 1.28
- pyyaml >= 6.0

### Data
- yfinance >= 0.2.30
- tenacity >= 8.0

### AI/ML
- pytorch >= 2.0
- llama-cpp-python >= 0.2.0 (for GGUF models)
- transformers >= 4.30 (for HuggingFace models)
- huggingface_hub >= 0.19

### Testing
- pytest >= 7.0
- pytest-cov >= 4.0
- pytest-mock >= 3.0

## Disclaimer

This application is for informational and educational purposes only. It does not constitute financial advice. Always conduct your own research and consult with qualified financial advisors before making trading decisions. Past performance is not indicative of future results.

## License

[Add your license here]
