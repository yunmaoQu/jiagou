package worker

import (
	"log"
)

func main() {
	log.Println("Worker 服务启动，等待任务...")
	if err := RunWorker(); err != nil {
		log.Fatalf("Worker 服务出错: %v", err)
	}
}
