# Murban Crude Intelligence Copilot

AI-powered trading analysis application for Murban-Brent crude oil spread analysis.

## Features

- **Real-time Market Data**: Fetches Murban and Brent crude oil prices from Yahoo Finance
- **Spread Analysis**: Calculates Murban-Brent spread with 5-day and 20-day moving averages
- **AI-Powered Insights**: Generates market signals using local LLM inference (llama-cpp-python)
- **Interactive Dashboard**: Streamlit-based UI with Plotly charts and dark theme

## Architecture

```
src/murban_copilot/
├── domain/           # Pure business logic
│   ├── entities.py   # MarketData, SpreadData, MovingAverages, MarketSignal
│   ├── spread_calculator.py
│   ├── validators.py
│   └── exceptions.py
├── infrastructure/   # External integrations
│   ├── market_data/  # Yahoo Finance client
│   ├── llm/          # LLM inference with llama-cpp-python
│   └── logging/      # Structured logging
├── application/      # Use cases
│   ├── fetch_market_data.py
│   ├── analyze_spread.py
│   └── generate_signal.py
└── interface/        # Streamlit dashboard
    └── streamlit_app.py
```

## Installation

### Prerequisites

- Conda or Miniconda
- Python 3.11+
- (Optional) GGUF model for LLM inference

### Setup

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

3. (Optional) Download a GGUF model for local LLM inference:
   ```bash
   # The application will automatically download a model from HuggingFace
   # Or specify a local model path when configuring
   ```

## Usage

### Running the Dashboard

```bash
PYTHONPATH=src streamlit run src/murban_copilot/interface/streamlit_app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

### Dashboard Features

- **Ticker Selection**: Choose between Murban vs Brent analysis
- **Historical Days**: Adjust the analysis period (7-90 days)
- **AI Analysis**: Toggle AI-powered market signal generation
- **Interactive Charts**: Zoom, pan, and hover for detailed data

## Running Tests

### All Tests
```bash
PYTHONPATH=src pytest
```

### Unit Tests Only
```bash
PYTHONPATH=src pytest tests/unit/
```

### Integration Tests (requires network)
```bash
PYTHONPATH=src pytest -m integration
```

### Skip Slow Tests
```bash
PYTHONPATH=src pytest -m "not slow"
```

### With Coverage Report
```bash
PYTHONPATH=src pytest --cov=src/murban_copilot --cov-report=html
open htmlcov/index.html
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MURBAN_LOG_LEVEL` | Logging level | `INFO` |
| `MURBAN_MODEL_PATH` | Path to GGUF model | Auto-download |

### LLM Configuration

The application uses `llama-cpp-python` for local LLM inference with GPU acceleration on Apple Silicon (Metal).

```python
from murban_copilot.infrastructure.llm import LlamaClient

client = LlamaClient(
    model_path="/path/to/model.gguf",  # Optional: uses HuggingFace Hub by default
    n_ctx=4096,                         # Context window size
    n_gpu_layers=-1,                    # -1 = use all GPU layers
)
```

## Technical Details

### Spread Calculation

- **Spread**: Murban Close - Brent Close ($/barrel)
- **5-Day MA**: Short-term moving average for trend detection
- **20-Day MA**: Long-term moving average for trend confirmation
- **Outlier Handling**: Median Absolute Deviation (MAD) for robust statistics

### Data Handling

- **Missing Data**: Forward fill (≤2 days), linear interpolation (longer gaps)
- **Timeout**: 60s for Yahoo Finance API calls
- **Caching**: LLM responses are cached locally for performance

## Project Structure

```
murban-intelligence-copilot/
├── environment.yml        # Conda environment definition
├── pytest.ini             # Pytest configuration
├── .coveragerc            # Coverage configuration
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

### Data
- yfinance >= 0.2.30

### AI/ML
- pytorch >= 2.0
- llama-cpp-python >= 0.2.0
- huggingface_hub >= 0.19

### Testing
- pytest >= 7.0
- pytest-cov >= 4.0
- pytest-mock >= 3.0
- freezegun >= 1.2

## Disclaimer

This application is for informational and educational purposes only. It does not constitute financial advice. Always conduct your own research and consult with qualified financial advisors before making trading decisions. Past performance is not indicative of future results.

## License

[Add your license here]
