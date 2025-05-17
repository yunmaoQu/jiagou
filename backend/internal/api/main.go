package api

import (
	"fmt"
	"github.com/yunmaoQu/codex-sys/internal/config"
	"github.com/yunmaoQu/codex-sys/internal/platform/database"
	"github.com/yunmaoQu/codex-sys/internal/platform/kafka"
	"github.com/yunmaoQu/codex-sys/internal/platform/objectstorage"
	"log"

	"github.com/gin-gonic/gin"
)

// SetupAndRun initializes all dependencies and starts the API server
func SetupAndRun() {
	cfg := config.LoadAPIConfig() // Load API-specific config from .env or config file

	// --- Initialize Platforms ---
	db, err := database.NewMySQLConnection(cfg.Database.DSN)
	if err != nil {
		log.Fatalf("Failed to connect to MySQL: %v", err)
	}
	defer db.Close()

	// redisClient := database.NewRedisClient(cfg.RedisAddr)
	// defer redisClient.Close()

	kafkaProducer, err := kafka.NewProducer(cfg.Kafka.Brokers, cfg.Kafka.Topics.Task)
	if err != nil {
		log.Fatalf("Failed to create Kafka producer: %v", err)
	}
	defer kafkaProducer.Close()

	// Create COS config from our configuration
	cosConfig := objectstorage.COSConfig{
		SecretID:  cfg.COS.SecretID,
		SecretKey: cfg.COS.SecretKey,
		Region:    cfg.COS.Region,
		BucketURL: fmt.Sprintf("https://%s.cos.%s.myqcloud.com", cfg.COS.Buckets.Code, cfg.COS.Region),
	}
	cosClient, err := objectstorage.NewCOSClient(cosConfig) // COS/S3 client
	if err != nil {
		log.Fatalf("Failed to create COS client: %v", err)
	}

	// --- Setup Router & Handlers ---
	router := gin.Default()
	// Pass DB, Kafka Producer, COS Client to handlers
	RegisterRoutes(router, db, kafkaProducer, cosClient /*, redisClient */)

	log.Printf("API Server starting on port %s", cfg.Server.Port)
	if err := router.Run(":" + cfg.Server.Port); err != nil {
		log.Fatalf("Failed to run API server: %v", err)
	}
}
