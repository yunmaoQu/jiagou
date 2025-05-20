package main

import (
	"fmt"
	"github.com/tencentyun/cos-go-sdk-v5"
	"github.com/yunmaoQu/codex-sys/worker/config"
	"github.com/yunmaoQu/codex-sys/worker/consumer"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"syscall"
)

// RunWorker 启动 Worker 服务
func RunWorker() error {
	// 1. 加载配置
	cfg, _ := config.LoadFromYAML("config.yaml")

	bucketUrl := fmt.Sprintf("https://%s.cos.%s.myqcloud.com", cfg.COS.Bucket, cfg.COS.Region)
	u, _ := url.Parse(bucketUrl)
	b := &cos.BaseURL{BucketURL: u}
	client := cos.NewClient(b, &http.Client{
		Transport: &cos.AuthorizationTransport{
			SecretID:  cfg.COS.AccessKey,
			SecretKey: cfg.COS.SecretKey,
		},
	})
	// 3. 创建任务消费者
	taskConsumer, err := consumer.NewTaskConsumer(cfg.Kafka.Brokers, client, cfg.Kafka.Topics.TaskTopic)
	if err != nil {
		return err
	}

	// 4. 启动消费者
	if err := taskConsumer.Start(); err != nil {
		return err
	}

	// 5. 等待信号退出
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig

	// 6. 优雅关闭
	log.Println("正在关闭 Worker 服务...")
	if err := taskConsumer.Stop(); err != nil {
		log.Printf("关闭消费者出错: %v", err)
	}

	return nil
}

func main() {
	log.Println("Worker 服务启动，等待任务...")

	if err := RunWorker(); err != nil {
		log.Fatalf("Worker 服务出错: %v", err)
	}
}
