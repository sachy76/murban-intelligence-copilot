# WTI-Brent Intelligence Copilot

AI-powered trading analysis application for WTI-Brent crude oil spread analysis.

## Features

- **Real-time Market Data**: Fetches WTI and Brent crude oil prices from Yahoo Finance
- **Spread Analysis**: Calculates WTI-Brent spread with configurable moving averages
- **AI-Powered Insights**: Generates market signals using local LLM inference (llama-cpp-python)
- **Interactive Dashboard**: Streamlit-based UI with Plotly charts and dark theme
- **Config-Driven**: All parameters configurable via YAML configuration file
- **Docker Support**: Production-ready containerization with health checks
- **Dual-LLM Architecture**: Separate models for analysis and signal extraction

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
│   ├── llm/             # LLM inference with llama-cpp-python
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

## Quick Start

### Using Docker (Recommended)

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t murban-copilot .
docker run -p 8501:8501 murban-copilot
```

The dashboard will be available at `http://localhost:8501`.

### Local Installation

#### Prerequisites

- Conda or Miniconda
- Python 3.11+
- (Optional) GGUF model for LLM inference

#### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd murban-intelligence-copilot
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate murban-copilot
   ```

3. Run the dashboard:
   ```bash
   PYTHONPATH=src streamlit run src/murban_copilot/interface/streamlit_app.py
   ```

## Configuration

The application is fully config-driven via `config/llm_config.yaml`:

```yaml
# LLM Configuration
llm:
  defaults:
    n_ctx: 4096
    n_gpu_layers: -1
    cache_enabled: true

  analysis:
    model_repo: "MaziyarPanahi/gemma-3-12b-it-GGUF"
    model_file: "gemma-3-12b-it.Q6_K.gguf"
    inference:
      max_tokens: 2048
      temperature: 0.7

  extraction:
    model_repo: "bartowski/gemma-2-9b-it-GGUF"
    model_file: "gemma-2-9b-it-Q4_K_M.gguf"
    inference:
      max_tokens: 1024
      temperature: 0.3

# Market Data Configuration
market_data:
  wti_ticker: "CL=F"
  brent_ticker: "BZ=F"
  timeout: 60
  max_retries: 3

# Analysis Configuration
analysis:
  short_ma_window: 5
  long_ma_window: 20
  outlier_threshold: 3.0

# Signal Generation Configuration
signal:
  default_signal: "neutral"
  default_confidence: 0.5
  bullish_keywords: ["bullish", "upward", "positive", "buy"]
  bearish_keywords: ["bearish", "downward", "negative", "sell"]

# UI Configuration
ui:
  min_historical_days: 7
  max_historical_days: 90
  default_historical_days: 30
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MURBAN_CONFIG` | Path to application config file | `config/llm_config.yaml` |
| `MURBAN_LLM_CONFIG` | Path to LLM config (legacy) | `config/llm_config.yaml` |
| `MURBAN_LOG_LEVEL` | Logging level | `INFO` |

### Configuration Sections

| Section | Description |
|---------|-------------|
| `llm` | LLM models, inference parameters, context settings |
| `cache` | LLM response caching settings |
| `market_data` | Ticker symbols, API timeouts, retry logic |
| `analysis` | Moving average windows, outlier detection |
| `signal` | Default values, classification keywords |
| `ui` | Dashboard slider bounds, sample data settings |

## Docker Deployment

### Docker Compose

```yaml
version: '3.8'
services:
  murban-copilot:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./config:/app/config:ro
      - ./.llm_cache:/app/.llm_cache
    environment:
      - MURBAN_CONFIG=/app/config/llm_config.yaml
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Health Checks

The application includes health check infrastructure:

```python
from murban_copilot.infrastructure.health import HealthChecker

checker = HealthChecker(market_client=client, llm_client=llm)
result = checker.check_all()
# Returns: HealthResult with status (HEALTHY, DEGRADED, UNHEALTHY)
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

## Technical Details

### Spread Calculation

- **Spread**: WTI Close - Brent Close ($/barrel)
- **5-Day MA**: Short-term moving average for trend detection (configurable)
- **20-Day MA**: Long-term moving average for trend confirmation (configurable)
- **Outlier Handling**: Median Absolute Deviation (MAD) for robust statistics

### Dual-LLM Architecture

The application uses a two-step LLM approach:
1. **Analysis LLM**: Generates comprehensive market analysis (higher temperature)
2. **Extraction LLM**: Extracts structured signal and confidence (lower temperature)

This separation allows for optimized models for each task.

### Data Handling

- **Missing Data**: Forward fill (≤2 days), linear interpolation (longer gaps)
- **Retry Logic**: Exponential backoff with configurable retries
- **Caching**: LLM responses cached locally for performance

## Project Structure

```
murban-intelligence-copilot/
├── config/
│   └── llm_config.yaml    # Application configuration
├── Dockerfile             # Multi-stage Docker build
├── docker-compose.yml     # Docker Compose configuration
├── environment.yml        # Conda environment definition
├── pytest.ini             # Pytest configuration
├── .gitignore             # Git ignore patterns
├── .dockerignore          # Docker ignore patterns
├── .env.example           # Environment variables template
├── README.md              # This file
├── src/
│   └── murban_copilot/    # Main package
└── tests/
    ├── unit/              # Unit tests
    ├── integration/       # Integration tests
    └── contract/          # Contract tests
```

## Dependencies

### Core
- Python 3.11
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
- llama-cpp-python >= 0.2.0
- huggingface_hub >= 0.19

### Testing
- pytest >= 7.0
- pytest-cov >= 4.0
- pytest-mock >= 3.0

## Disclaimer

This application is for informational and educational purposes only. It does not constitute financial advice. Always conduct your own research and consult with qualified financial advisors before making trading decisions. Past performance is not indicative of future results.

## License

[Add your license here]
