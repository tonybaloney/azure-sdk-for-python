FROM python:3.12-slim

# Ensure Python output is sent straight to the container logs (no buffering)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy the package source and install it
COPY . package/
RUN pip install ./package/ azure-identity "github-copilot-sdk>=0.1.25"

# Copy the hosted agent entry point and ACL file
COPY samples/hosted_agent/main.py main.py
COPY samples/hosted_agent/tools_acl.yaml tools_acl.yaml

EXPOSE 8088

CMD ["python", "main.py"]
