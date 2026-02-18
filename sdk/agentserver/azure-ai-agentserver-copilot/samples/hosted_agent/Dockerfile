FROM python:3.12-slim

WORKDIR /app

# Copy the package source and install it
COPY . package/
RUN pip install ./package/ azure-identity "github-copilot-sdk>=0.1.25"

# Copy the hosted agent entry point
COPY samples/hosted_agent/main.py main.py

EXPOSE 8088

CMD ["python", "main.py"]
