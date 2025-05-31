package main

import (
	"log"

	"github.com/yunmaoQu/codex-sys/internal/api"
)

func main() {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("!!!!!!!! PANIC RECOVERED !!!!!!!!: %v", r)
		}
	}()

	log.Println("~~~~~ main function started ~~~~~")
	log.Println("Starting backend...") // Existing log line
	log.Println("~~~~~ Calling api.SetupAndRun()... ~~~~~")
	api.SetupAndRun()
	log.Println("~~~~~ api.SetupAndRun() completed (this might not be reached) ~~~~~")
}
