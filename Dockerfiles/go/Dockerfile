FROM golang:1.21 as builder

WORKDIR /app

# Copy go mod and sum files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -o swe ./swe

# Create final image
FROM ubuntu:24.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    diffutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the binary from builder
COPY --from=builder /app/agent .

# Create directories
RUN mkdir -p /app/code /app/output

ENV OPENAI_API_KEY=""
ENV GITHUB_TOKEN=""
ENV TASK_ID=""

CMD ["./agent"]
