package agent

import (
	"flag"
	"log"
	"os"
	"path/filepath"
)

func main() {
	// 解析命令行参数
	taskDesc := flag.String("task", "", "任务描述")
	targetFile := flag.String("target", "", "目标文件路径（相对于代码目录）")
	flag.Parse()

	// 验证参数
	if *taskDesc == "" || *targetFile == "" {
		log.Fatal("必须提供任务描述和目标文件路径")
	}

	// 获取环境变量
	taskID := os.Getenv("TASK_ID")
	if taskID == "" {
		taskID = "unknown"
	}

	githubToken := os.Getenv("GITHUB_TOKEN")

	// 设置代码和输出目录
	codeDir := "/app/code"
	outputDir := "/app/output"

	// 确保输出目录存在
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		log.Fatalf("无法创建输出目录: %v", err)
	}

	// 创建并运行Agent
	agent := NewAgent(taskID, *taskDesc, *targetFile, codeDir, outputDir, githubToken)
	if err := agent.Run(); err != nil {
		log.Fatalf("Agent执行失败: %v", err)
	}

	log.Println("Agent执行完成，结果保存在输出目录中")

	// 记录输出文件列表
	outputFiles, err := filepath.Glob(filepath.Join(outputDir, "*"))
	if err == nil {
		log.Println("输出文件:")
		for _, file := range outputFiles {
			log.Printf("- %s\n", filepath.Base(file))
		}
	}
}
