.PHONY: server client runc runs

CC=gcc
CFLAGS=-Wall -Werror -Wextra -g

SERVER_NAME=server
CLIENT_NAME=client
SRC_EXT=c
OUT_EXT=out

server:
	$(CC) $(SERVER_NAME).$(SRC_EXT) $(CFLAGS) -o $(SERVER_NAME).$(OUT_EXT)

client:
	$(CC) $(CLIENT_NAME).$(SRC_EXT) $(CFLAGS) -o $(CLIENT_NAME).$(OUT_EXT)

runs: server
	./$(SERVER_NAME).$(OUT_EXT)

runc: client
	./$(CLIENT_NAME).$(OUT_EXT) $(ARGS)