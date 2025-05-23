package main

import (
	"fmt"
	"github.com/yunmaoQu/codex-sys/internal/platform/objectstorage"
	"github.com/yunmaoQu/codex-sys/worker/worker"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/yunmaoQu/codex-sys/internal/platform/database"
	"github.com/yunmaoQu/codex-sys/worker/config"
)

// RunWorker 启动 Worker 服务
func RunWorker() error {
	// 1. 加载配置
	appCfg, err := config.LoadFromYAML("/workspaces/Codex-like-SYS/backend/worker/config/config.yaml")
	if err != nil {
		return fmt.Errorf("加载配置失败: %w", err)
	}

	cosConfig := objectstorage.COSConfig{
		SecretID:  appCfg.COS.AccessKey,
		SecretKey: appCfg.COS.SecretKey,
		Region:    appCfg.COS.Region,
		BucketURL: fmt.Sprintf("https://%s.cos.%s.myqcloud.com", appCfg.COS.Buckets.Code, appCfg.COS.Region),
	}
	cosClient, _ := objectstorage.NewCOSClient(cosConfig) // COS/S3 client

	buildDSN := fmt.Sprintf("%s:%s@tcp(%s:%s)/%s?charset=utf8mb4&parseTime=True&loc=Local",
		appCfg.Database.User,
		appCfg.Database.Password,
		appCfg.Database.Host,
		appCfg.Database.Port,
		appCfg.Database.Name,
	)
	db, err := database.NewMySQLConnection(buildDSN)
	if err != nil {
		log.Fatalf("Failed to connect to MySQL: %v", err)
	}
	defer db.Close()
	var dbWrapper = database.NewDBClientWrapper(db)

	// 4. 创建 Worker 配置
	workerCfg := worker.Config{
		ExecutionMode:      worker.DockerMode,    // 或者 handler.K8sMode
		AgentImage:         "codex-agent:latest", // 可以从 appCfg 中获取
		TempDirBase:        "/tmp/codex-worker",
		TaskTopic:          appCfg.Kafka.Topics.Task,
		ReaslutTopic: 	    appCfg.Kafka.Topics.Result,
		K8sNamespace:       "default",     // 如果使用 K8s 模式
		K8sServiceAccount:  "codex-agent", // 如果使用 K8s 模式
		CPULimit:           "1",
		MemoryLimit:        "2Gi",
		CleanupTempDirs:    true,
		CodeBucket:         appCfg.COS.Buckets.Code,
		LogsBucket:         appCfg.COS.Buckets.Logs, 
		EnableGitHubAccess: true,
	}

	// 5. 创建 Worker 实例
	consumer, err := worker.NewWorker(workerCfg, appCfg.Kafka.Brokers, cosClient, dbWrapper)
	if err != nil {
		return fmt.Errorf("创建 Worker 失败: %w", err)
	}

	// 6. 启动 Worker
	if err := consumer.Start(); err != nil {
		return fmt.Errorf("启动 Worker 失败: %w", err)
	}

	log.Println("Worker 启动成功，等待任务...")

	// 7. 等待退出信号
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig

	// 8. 优雅关闭
	log.Println("正在关闭 Worker 服务...")
	if err := consumer.Stop(); err != nil {
		log.Printf("关闭 Worker 出错: %v", err)
	}

	return nil
}

func main() {
	log.Println("Worker 服务启动，等待任务...")

	if err := RunWorker(); err != nil {
		log.Fatalf("Worker 服务出错: %v", err)
	}
}
