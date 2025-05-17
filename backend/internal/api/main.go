package api

import (
	"codex-sys/backend/internal/api"
	"codex-sys/backend/internal/config"
	"codex-sys/backend/internal/platform/database"
	"codex-sys/backend/internal/platform/kafka"
	"codex-sys/backend/internal/platform/objectstorage"
	"log"

	"github.com/gin-gonic/gin"
	// For MySQL
	// "github.com/go-redis/redis/v8" // For Redis
	// "github.com/segmentio/kafka-go" // For Kafka
)

func main() {
	cfg := config.Load() // Load from .env or config file

	// --- Initialize Platforms ---
	db, err := database.NewMySQLConnection(cfg.MySQLDSN)
	if err != nil {
		log.Fatalf("Failed to connect to MySQL: %v", err)
	}
	defer db.Close()

	// redisClient := database.NewRedisClient(cfg.RedisAddr)
	// defer redisClient.Close()

	kafkaProducer, err := kafka.NewProducer(cfg.KafkaBrokers, cfg.KafkaTaskTopic)
	if err != nil {
		log.Fatalf("Failed to create Kafka producer: %v", err)
	}
	defer kafkaProducer.Close()

	cosClient, err := objectstorage.NewCOSClient(cfg.COSConfig) // COS/S3 client
	if err != nil {
		log.Fatalf("Failed to create COS client: %v", err)
	}

	// --- Setup Router & Handlers ---
	router := gin.Default()
	// Pass DB, Kafka Producer, COS Client to handlers
	api.RegisterRoutes(router, db, kafkaProducer, cosClient /*, redisClient */)

	log.Printf("API Server starting on port %s", cfg.APIPort)
	if err := router.Run(":" + cfg.APIPort); err != nil {
		log.Fatalf("Failed to run API server: %v", err)
	}
}
