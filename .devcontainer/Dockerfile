# VS Code dev container for running patgen notebooks offline.
# Build this once with network, then the image contains Python, Jupyter,
# and the patgen package with all dev dependencies. No network is required
# while the notebook is running.

FROM python:3.10-slim

# Install only the tools needed to build/compile any wheels; most dependencies
# have pre-built wheels, but this keeps the build robust.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Set up the workspace exactly as VS Code devcontainers expect it.
WORKDIR /workspaces/patternExtract

# Copy the repo and install patgen plus all dev extras (jupyter, nbformat, etc.).
COPY . .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[dev]"

# Jupyter notebook port.
EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
