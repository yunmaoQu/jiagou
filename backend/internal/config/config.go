package config

import (
	"os"
	"strings"

	"github.com/joho/godotenv"
)

// Config represents the application configuration
type Config struct {
	// Server configuration
	Server struct {
		Port string
		Env  string
	}

	// Database configuration
	Database struct {
		Host     string
		Port     string
		User     string
		Password string
		Name     string
		DSN      string
	}

	// Redis configuration
	Redis struct {
		Host     string
		Port     string
		Password string
		DB       int
	}

	// Kafka configuration
	Kafka struct {
		Brokers []string
		Topics  struct {
			Task   string
			Result string
		}
	}

	// COS configuration
	COS struct {
		SecretID  string
		SecretKey string
		Region    string
		Buckets   struct {
			Code string
			Logs string
		}
	}

	// Kubernetes configuration
	K8s struct {
		Namespace      string
		AgentImage     string
		ServiceAccount string
	}

	// OpenAI configuration
	OpenAI struct {
		APIKey string
	}

	// GitHub configuration
	GitHub struct {
		Token string
	}
}

// Load loads the configuration from environment variables or .env file
func Load() *Config {
	// Try to load .env file if it exists
	_ = godotenv.Load()

	// Initialize config
	var cfg Config

	// Server config
	cfg.Server.Port = getEnv("SERVER_PORT", "8080")
	cfg.Server.Env = getEnv("SERVER_ENV", "development")

	// Database config
	cfg.Database.Host = getEnv("DB_HOST", "localhost")
	cfg.Database.Port = getEnv("DB_PORT", "3306")
	cfg.Database.User = getEnv("DB_USER", "root")
	cfg.Database.Password = getEnv("DB_PASSWORD", "")
	cfg.Database.Name = getEnv("DB_NAME", "codex_sys")
	cfg.Database.DSN = buildDSN(cfg.Database.User, cfg.Database.Password, cfg.Database.Host, cfg.Database.Port, cfg.Database.Name)

	// Redis config
	cfg.Redis.Host = getEnv("REDIS_HOST", "localhost")
	cfg.Redis.Port = getEnv("REDIS_PORT", "6379")
	cfg.Redis.Password = getEnv("REDIS_PASSWORD", "")

	// Kafka config
	kafkaBrokers := getEnv("KAFKA_BROKERS", "localhost:9092")
	cfg.Kafka.Brokers = strings.Split(kafkaBrokers, ",")
	cfg.Kafka.Topics.Task = getEnv("KAFKA_TOPIC_TASK", "codex-tasks")
	cfg.Kafka.Topics.Result = getEnv("KAFKA_TOPIC_RESULT", "codex-results")

	// COS config
	cfg.COS.SecretID = getEnv("COS_SECRET_ID", "")
	cfg.COS.SecretKey = getEnv("COS_SECRET_KEY", "")
	cfg.COS.Region = getEnv("COS_REGION", "ap-guangzhou")
	cfg.COS.Buckets.Code = getEnv("COS_BUCKET_CODE", "codex-code")
	cfg.COS.Buckets.Logs = getEnv("COS_BUCKET_LOGS", "codex-logs")

	// K8s config
	cfg.K8s.Namespace = getEnv("K8S_NAMESPACE", "default")
	cfg.K8s.AgentImage = getEnv("K8S_AGENT_IMAGE", "codex-agent:latest")
	cfg.K8s.ServiceAccount = getEnv("K8S_SERVICE_ACCOUNT", "codex-agent")

	// OpenAI config
	cfg.OpenAI.APIKey = getEnv("OPENAI_API_KEY", "")

	// GitHub config
	cfg.GitHub.Token = getEnv("GITHUB_TOKEN", "")

	return &cfg
}

// LoadWorkerConfig loads the configuration specific to the worker service
func LoadWorkerConfig() *Config {
	return Load()
}

// LoadAPIConfig loads the configuration specific to the API service
func LoadAPIConfig() *Config {
	return Load()
}

// Helper functions
func getEnv(key, defaultValue string) string {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	return value
}

func buildDSN(user, password, host, port, dbName string) string {
	return user + ":" + password + "@tcp(" + host + ":" + port + ")/" + dbName + "?parseTime=true"
}
