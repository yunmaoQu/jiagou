package main

import (
	"log"

	"github.com/yunmaoQu/codex-sys/internal/api"
)

func main() {
	log.Println("Starting backend...")
	api.SetupAndRun()
}
