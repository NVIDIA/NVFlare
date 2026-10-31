package main

import (
	"os"
	"os/signal"
	"syscall"
)

func main() {
	if len(os.Args) != 1 {
		os.Exit(64)
	}
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop
}
