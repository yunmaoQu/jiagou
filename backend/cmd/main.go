package main

import (
	"github.com/yunmaoQu/codex-sys/internal/api"
	"github.com/yunmaoQu/codex-sys/internal/task"
	"github.com/yunmaoQu/codex-sys/utils"
	"log"
	"os"
	"path/filepath"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
)

func main() {
	err := godotenv.Load("../.env") // Load .env from project root
	if err != nil {
		log.Println("No .env file found or error loading, relying on environment variables")
	}

	port := os.Getenv("BACKEND_PORT")
	if port == "" {
		port = "8080"
	}

	storagePath := os.Getenv("STORAGE_PATH")
	if storagePath == "" {
		storagePath = "../storage" // Relative to backend executable
	}
	// Ensure storagePath is absolute
	absStoragePath, err := filepath.Abs(storagePath)
	if err != nil {
		log.Fatalf("Error getting absolute path for storage: %v", err)
	}
	utils.GlobalStoragePath = absStoragePath // Set global storage path

	// Create necessary storage directories
	if err := os.MkdirAll(filepath.Join(utils.GlobalStoragePath, "repos"), 0755); err != nil {
		log.Fatalf("Failed to create repos directory: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(utils.GlobalStoragePath, "logs"), 0755); err != nil {
		log.Fatalf("Failed to create logs directory: %v", err)
	}

	// Initialize Task Manager
	task.InitManager()

	router := gin.Default()

	// CORS configuration
	config := cors.DefaultConfig()
	config.AllowOrigins = []string{"*"} // Allow all origins for simplicity. For production, restrict this.
	config.AllowMethods = []string{"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}
	config.AllowHeaders = []string{"Origin", "Content-Type", "Accept", "Authorization"}
	router.Use(cors.New(config))

	// API routes
	apiGroup := router.Group("/api")
	{
		apiGroup.POST("/task", api.HandleCreateTask)
		apiGroup.GET("/task/:task_id/status", api.HandleGetTaskStatus)
		apiGroup.GET("/logs/:task_id/:filename", api.HandleGetLogFile)
	}

	// Serve frontend static files (optional, for simple demo)
	// For production, use a dedicated web server or CDN for frontend
	// Note: This path is relative to where the 'backend' executable runs.
	// If you run 'go run main.go' from 'codex-sys/backend/', then '../frontend' is correct.
	router.Static("/ui", "../frontend")
	router.GET("/", func(c *gin.Context) {
		c.Redirect(302, "/ui/index.html")
	})

	log.Printf("Server starting on port %s", port)
	log.Printf("Storage path: %s", utils.GlobalStoragePath)
	log.Printf("Access UI at http://localhost:%s/ui/index.html or http://localhost:%s/", port, port)

	if err := router.Run(":" + port); err != nil {
		log.Fatalf("Failed to run server: %v", err)
	}
}
