## Title: Fetch Historical Market Data
## User Story - As a Trading Analyst, I want to fetch 30 days of historical price data for Murban ($BZ=F) and Brent ($LCO=F) so that the AI can identify price spreads and volatility trends.
## Acceptance Criteria
- **Given** I have a valid internet connection, **When** I run the data fetching script, **Then** 30 days of daily OHLC (Open, High, Low, Close) data for both $BZ=F and $LCO=F are downloaded successfully from Yahoo Finance.
- **Given** the Yahoo Finance API returns an error, **When** I run the data fetching script, **Then** an informative error message is logged, and the script gracefully exits without crashing.
- **Given** there are missing data points in the downloaded data, **When** I run the data fetching script, **Then** the missing values are handled by forward filling or interpolation, ensuring a continuous dataset for analysis.
- **Given** the data fetching script is run during a period of high market volatility, **When** I run the data fetching script, **Then** the script completes within a reasonable timeframe (e.g., under 60 seconds) to avoid delays in analysis.

## Title: Calculate Spread and Moving Averages
## User Story - As a Trading Analyst, I want the script to calculate the Murban-Brent spread and moving averages so that I can identify trends and potential trading opportunities.
## Acceptance Criteria
- **Given** I have downloaded 30 days of historical price data for Murban and Brent, **When** I run the calculation script, **Then** a "Spread" column is calculated accurately as the difference between Murban and Brent closing prices.
- **Given** I have calculated the Spread, **When** I run the calculation script, **Then** a 5-day Moving Average and a 20-day Moving Average are calculated accurately for the Spread.
- **Given** the dataset contains fewer than 5 or 20 data points, **When** I run the calculation script, **Then** the moving averages are not calculated for those periods, and a warning message is logged.
- **Given** the data contains outliers, **When** I run the calculation script, **Then** the moving averages are calculated using robust statistical methods to minimize the impact of outliers.

## Title: Generate Market Signal Brief
## User Story - As an Energy Trader, I want the LLM to analyze the calculated technical data and generate a "Market Signal Brief" so that I can identify potential hedging opportunities.
## Acceptance Criteria
- **Given** the script has calculated the latest price, spread change, and moving averages, **When** I provide the data to the LLM, **Then** the LLM generates a concise "Market Signal Brief" in a "Trader Talk" format (e.g., "Murban-Brent spread widening; watch for resistance at $X.XX").
- **Given** the LLM generates a "Market Signal Brief", **When** I review the output, **Then** the brief includes a clear statement explicitly stating that it is not providing financial advice.
- **Given** the LLM encounters an error during inference, **When** I provide the data to the LLM, **Then** an informative error message is displayed to the user, and the "Thinking..." spinner stops.
- **Given** the LLM prompt is too complex or ambiguous, **When** I provide the data to the LLM, **Then** the LLM generates a brief that is relevant and actionable, but avoids speculative or overly complex language.

## Title: Display Dashboard Insights
## User Story - As an ADNOC Executive, I want to see a visual dashboard with charts and AI-generated insights side-by-side so that I can quickly assess market conditions.
## Acceptance Criteria
- **Given** the script has generated the Murban vs. Brent spread data, **When** I run the Streamlit application, **Then** a Plotly or Altair chart is displayed showing the spread over the 30-day period.
- **Given** I am using the Streamlit application, **When** I select a different ticker from the sidebar (e.g., $CL=F), **Then** the chart updates to display the selected ticker's data.
- **Given** the LLM is generating a "Market Signal Brief", **When** I run the Streamlit application, **Then** a "Thinking..." spinner is displayed in the designated text container while the LLM is processing.
- **Given** the LLM has generated a "Market Signal Brief", **When** I run the Streamlit application, **Then** the brief is displayed in the text container, clearly separated from the chart.
- **Given** the Streamlit application is running, **When** I resize the browser window, **Then** the dashboard elements (chart and text container) automatically adjust to fit the new window size.

# Technical User Stories

## Title: Developer instructions
## User Story - As a Developer, I want upto date system documentation so that new developer can get used to very quickly
## Acceptance Criteria
- **Given** project can be installed on new developer machine using YAML,  **When** new developer reads instructions from README.MD file and triggers installation, **Then** entire setup including environment build is executed.
- **Given** new non-techy/techy joins reads the code, **When** enough documentation is available embedded in the code, **Then** code logic is understood by non-techy as well as techy members.

## Title: Initialize Local LLM Inference
## User Story - As a Solution Architect, I want to initialize a Hugging Face model optimized for the M5 chip so that the demo runs without latency or dependency on external cloud APIs.
## Acceptance Criteria
- **Given** the script is running on an Apple Macbook Pro with an M5 chip, **When** the script initializes the LLM, **Then** `torch.backends.mps.is_available()` returns `True`, indicating MPS (Metal Performance Shaders) acceleration is enabled.
- **Given** the script is initializing the LLM, **When** I specify a model (e.g., `meta-llama/Llama-3.1-8B-Instruct`), **Then** the model is loaded using bitsandbytes 4-bit quantization to reduce memory footprint.
- **Given** the model weights have already been downloaded, **When** I run the script, **Then** the script utilizes the local caching mechanism (e.g., `/Users/shared/huggingface/hub`) to prevent re-downloading the model weights.
- **Given** the model weights are not found in the local cache, **When** I run the script, **Then** the script downloads the model weights from the Hugging Face Hub and stores them in the local cache for future use.

## Title: Implement Error Logging
## User Story - As a Developer, I want to implement robust error logging so that I can quickly identify and resolve issues in the application.
## Acceptance Criteria
- **Given** an error occurs during data fetching, **When** the script encounters the error, **Then** a detailed error message, including the timestamp and error type, is logged to a designated log file.
- **Given** an error occurs during LLM inference, **When** the script encounters the error, **Then** a detailed error message, including the input data and error type, is logged to a designated log file.
- **Given** an unexpected exception occurs, **When** the script encounters the exception, **Then** the exception is caught, logged with a traceback, and the script gracefully exits without crashing.
- **Given** the log file reaches a certain size, **When** new log entries are added, **Then** the log file is automatically rotated to prevent it from consuming excessive disk space.


## Title: Data Validation
## User Story - As a Data Engineer, I want to implement data validation checks so that I can ensure the quality and integrity of the data used for analysis.
## Acceptance Criteria
- **Given** the script is fetching data from Yahoo Finance, **When** the data is downloaded, **Then** the script validates that the data contains the expected columns (Open, High, Low, Close, Volume).
- **Given** the script is calculating the spread, **When** the spread is calculated, **Then** the script validates that the spread values are numeric and within a reasonable range.
- **Given** the script is providing data to the LLM, **When** the data is passed to the LLM, **Then** the script validates that the data is in the expected format and contains all the required fields.
- **Given** data validation fails, **When** a validation check fails, **Then** an error message is logged, and the script either attempts to correct the data or gracefully exits.

## Title: Performance Optimization
## User Story - As a Performance Engineer, I want to optimize the script's performance so that it runs efficiently on the M5 chip.
## Acceptance Criteria
- **Given** the script is running on the M5 chip, **When** the script is initialized, **Then** the script utilizes MPS acceleration for all tensor operations.
- **Given** the script is performing calculations, **When** the calculations are performed, **Then** the script utilizes vectorized operations whenever possible to improve performance.
- **Given** the script is running the Streamlit application, **When** the user interacts with the dashboard, **Then** the dashboard updates and renders within a reasonable timeframe (e.g., under 10 seconds).
- **Given** the script is running, **When** the script is monitored, **Then** CPU and memory usage remain within acceptable limits.
