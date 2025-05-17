package worker

import (
	"github.com/tencentyun/cos-go-sdk-v5"
	"log"
	"os"
	"os/signal"
	"syscall"
)

// Config 配置结构体
type Config struct {
	Kafka struct {
		Brokers []string
		Topics  struct {
			TaskTopic   string
			ResultTopic string
		}
	}
	COS struct {
		Bucket    string
		Region    string
		AccessKey string
		SecretKey string
	}
	Docker struct {
		AgentImage string
	}
}

// RunWorker 启动 Worker 服务
func RunWorker() error {
	// 1. 加载配置
	cfg := loadConfig()

	// 2. 初始化 COS 客户端
	cosClient, err := initCOS(cfg)
	if err != nil {
		return err
	}

	// 3. 创建任务消费者
	consumer, err := NewTaskConsumer(cfg.Kafka.Brokers, cosClient)
	if err != nil {
		return err
	}

	// 4. 启动消费者
	if err := consumer.Start(); err != nil {
		return err
	}

	// 5. 等待信号退出
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig

	// 6. 优雅关闭
	log.Println("正在关闭 Worker 服务...")
	if err := consumer.Stop(); err != nil {
		log.Printf("关闭消费者出错: %v", err)
	}

	return nil
}

// loadConfig 加载配置
func loadConfig() Config {
	// TODO: 从环境变量或配置文件加载配置
	return Config{
		Kafka: struct {
			Brokers []string
			Topics  struct {
				TaskTopic   string
				ResultTopic string
			}
		}{
			Brokers: []string{"localhost:9092"},
			Topics: struct {
				TaskTopic   string
				ResultTopic string
			}{
				TaskTopic:   "codex-tasks",
				ResultTopic: "codex-results",
			},
		},
		COS: struct {
			Bucket    string
			Region    string
			AccessKey string
			SecretKey string
		}{
			Bucket:    os.Getenv("COS_BUCKET"),
			Region:    os.Getenv("COS_REGION"),
			AccessKey: os.Getenv("COS_ACCESS_KEY"),
			SecretKey: os.Getenv("COS_SECRET_KEY"),
		},
		Docker: struct {
			AgentImage string
		}{
			AgentImage: "codex-agent:latest",
		},
	}

}
func initCOS(cfg Config) (*cos.Client, error) {
	return nil, nil
}
