FROM golang:1.21 as builder

WORKDIR /app

# Copy go mod and sum files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -o codex-api .

# Create final image
FROM alpine:3.19

WORKDIR /app

# Copy the binary from builder
COPY --from=builder /app/codex-api .

# Copy configuration
COPY configs/ ./configs/

EXPOSE 8080

CMD ["./codex-api"]
