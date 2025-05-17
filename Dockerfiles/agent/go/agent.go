package agent

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/sashabaranov/go-openai"
)

// Agent 代表一个代码分析和修改的Agent
type Agent struct {
	openaiClient *openai.Client
	taskID       string
	taskDesc     string
	targetFile   string
	codeDir      string
	outputDir    string
	githubToken  string
}

// NewAgent 创建一个新的Agent实例
func NewAgent(taskID, taskDesc, targetFile, codeDir, outputDir, githubToken string) *Agent {
	// 初始化OpenAI客户端
	openaiKey := os.Getenv("OPENAI_API_KEY")
	if openaiKey == "" {
		log.Fatal("OPENAI_API_KEY环境变量未设置")
	}

	client := openai.NewClient(openaiKey)

	return &Agent{
		openaiClient: client,
		taskID:       taskID,
		taskDesc:     taskDesc,
		targetFile:   targetFile,
		codeDir:      codeDir,
		outputDir:    outputDir,
		githubToken:  githubToken,
	}
}

// Run 执行代码分析和修改任务
func (a *Agent) Run() error {
	// 1. 读取目标文件
	code, err := a.readTargetFile()
	if err != nil {
		return fmt.Errorf("读取目标文件失败: %w", err)
	}

	// 2. 读取自定义指令（如果存在）
	agentInstructions := a.readAgentInstructions()

	// 3. 构建提示词
	prompt := a.buildPrompt(code, agentInstructions)

	// 4. 调用LLM
	response, err := a.callLLM(prompt)
	if err != nil {
		return fmt.Errorf("调用LLM失败: %w", err)
	}

	// 5. 保存响应
	if err := a.saveResponse(response); err != nil {
		return fmt.Errorf("保存响应失败: %w", err)
	}

	// 6. 生成差异文件
	if err := a.generateDiff(code, response); err != nil {
		return fmt.Errorf("生成差异文件失败: %w", err)
	}

	// 7. 如果有GitHub Token，尝试创建PR
	if a.githubToken != "" {
		if err := a.createPR(response); err != nil {
			return fmt.Errorf("创建PR失败: %w", err)
		}
	}

	return nil
}

// readTargetFile 读取目标文件内容
func (a *Agent) readTargetFile() (string, error) {
	filePath := filepath.Join(a.codeDir, a.targetFile)
	data, err := io.Read(filePath)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// readAgentInstructions 读取自定义Agent指令
func (a *Agent) readAgentInstructions() string {
	agentsFilePath := filepath.Join(a.codeDir, "AGENTS.md")
	data, err := io.Read(agentsFilePath)
	if err != nil {
		// 如果文件不存在，返回空字符串
		return ""
	}
	return string(data)
}

// buildPrompt 构建提示词
func (a *Agent) buildPrompt(code, instructions string) string {
	prompt := fmt.Sprintf("任务描述: %s\n\n", a.taskDesc)

	if instructions != "" {
		prompt += fmt.Sprintf("自定义指令:\n%s\n\n", instructions)
	}

	prompt += fmt.Sprintf("目标文件: %s\n\n代码:\n```\n%s\n```\n\n", a.targetFile, code)
	prompt += "请分析上述代码，并根据任务描述进行修改。提供详细的解释和修改后的完整代码。"

	return prompt
}

// callLLM 调用大语言模型
func (a *Agent) callLLM(prompt string) (string, error) {
	resp, err := a.openaiClient.CreateChatCompletion(
		context.Background(),
		openai.ChatCompletionRequest{
			Model: openai.GPT4,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: prompt,
				},
			},
		},
	)

	if err != nil {
		return "", err
	}

	return resp.Choices[0].Message.Content, nil
}

// saveResponse 保存LLM响应
func (a *Agent) saveResponse(response string) error {
	responsePath := filepath.Join(a.outputDir, "llm_response.txt")
	return ioutil.WriteFile(responsePath, []byte(response), 0644)
}

// generateDiff 生成差异文件
func (a *Agent) generateDiff(originalCode, response string) error {
	// 从响应中提取修改后的代码
	modifiedCode := a.extractCodeFromResponse(response)
	if modifiedCode == "" {
		return fmt.Errorf("无法从响应中提取代码")
	}

	// 保存修改后的代码
	modifiedPath := filepath.Join(a.outputDir, "modified_code.txt")
	if err := ioutil.WriteFile(modifiedPath, []byte(modifiedCode), 0644); err != nil {
		return err
	}

	// TODO: 使用diff工具生成差异文件
	// 这里简化实现，实际应该调用系统diff命令或使用专门的diff库

	return nil
}

// extractCodeFromResponse 从LLM响应中提取代码
func (a *Agent) extractCodeFromResponse(response string) string {
	// 查找代码块
	parts := strings.Split(response, "```")
	if len(parts) < 3 {
		return ""
	}

	// 代码块通常在第二个部分
	codeBlock := parts[1]
	// 如果代码块以语言标识开头，去掉第一行
	lines := strings.Split(codeBlock, "\n")
	if len(lines) > 0 && (strings.Contains(lines[0], "go") || strings.Contains(lines[0], "python") || strings.Contains(lines[0], "javascript")) {
		codeBlock = strings.Join(lines[1:], "\n")
	}

	return codeBlock
}

// createPR 创建GitHub PR
func (a *Agent) createPR(response string) error {
	// TODO: 实现GitHub PR创建逻辑
	// 需要使用GitHub API或命令行工具

	// 保存PR URL到文件
	prURL := "https://github.com/example/repo/pull/123" // 示例URL
	prURLPath := filepath.Join(a.outputDir, "pr_url.txt")
	return io.WriteFile(prURLPath, []byte(prURL), 0644)
}
