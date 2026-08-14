Total runtime: 2 hr 28 minutes
GPU: NVIDIA 1660

Requirements: Docker Desktop, a Windows PC with an NVIDIA GPU, Python + IDE

Docker instructions: Run 
``` docker pull echuvb/neuralgcm-final ```
in your Terminal to download the Docker image for the pipeline.

Then run 
``` docker run --gpus all -v ${PWD}/output:/app echuvb/neuralgcm-final ```

You should then see backtest_summary.csv and forecast_summary.csv in your output folder as your forecast results.
