package api

import (
	"fmt"
	"log"

	"github.com/yunmaoQu/codex-sys/internal/config"
	"github.com/yunmaoQu/codex-sys/internal/platform/database"
	"github.com/yunmaoQu/codex-sys/internal/platform/kafka"
	"github.com/yunmaoQu/codex-sys/internal/platform/objectstorage"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

// SetupAndRun initializes all dependencies and starts the API server
func SetupAndRun() {
	cfg, err := config.LoadFromYAML("/backend/config") // Load API-specific config from .env or config file
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

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

	// CORS 配置
	corsConfig := cors.DefaultConfig()
	corsConfig.AllowOrigins = []string{"*"}
	corsConfig.AllowMethods = []string{"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}
	corsConfig.AllowHeaders = []string{"Origin", "Content-Type", "Accept", "Authorization"}
	router.Use(cors.New(corsConfig))

	// 注入 db 到 gin.Context
	router.Use(func(c *gin.Context) {
		c.Set("db", db)
		c.Next()
	})

	// Pass DB, Kafka Producer, COS Client to handlers
	RegisterRoutes(router, db, kafkaProducer, cosClient /*, redisClient */)

	// 前端静态文件服务
	router.Static("/ui", "../frontend")
	router.GET("/", func(c *gin.Context) {
		c.Redirect(302, "/ui/index.html")
	})

	log.Printf("API Server starting on port %s", cfg.Server.Port)
	if err := router.Run(":" + cfg.Server.Port); err != nil {
		log.Fatalf("Failed to run API server: %v", err)
	}
}
