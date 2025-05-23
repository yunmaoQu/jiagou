FROM python:3.10-slim

# Install git and curl (git for cloning/PRs, curl for potential setup.sh needs)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    # GitHub CLI for PR creation
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y gh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent.py .
# AGENTS.md.example and setup.sh.example are not copied here
# They are expected to be in the user's code mounted at /app/code

# Ensure output directory exists for the swe
RUN mkdir -p /app/output

# Default command, can be overridden by Docker run command
# CMD ["python3", "swe.py"]
ENTRYPOINT ["python3", "agent.py"]