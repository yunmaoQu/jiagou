package worker

import (
	"log"

)

// 入口函数，在 cmd/worker/main.go 中调用
func main() {
	log.Println("Worker 服务启动，等待任务...")
	if err := RunWorker(); err != nil {
		log.Fatalf("Worker 服务出错: %v", err)
	}
}
