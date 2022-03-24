VERSION 	=1.00
CC			=gcc
DEBUG 		=-g
CFLAGS		=-Wall -Werror -Wextra
BINARIES	=bin
SOURCES	 	=$(wildcard ./src/*.c)
INCLUDES  	=-I./include
LIB_NAMES 	=
LIB_PATH 	=
OBJ			=$(patsubst %.c, %.o, $(SOURCES))
TARGET		=ftp_server

#links
$(TARGET):$(OBJ)
	@mkdir -p $(BINARIES)
	$(CC) $(OBJ) $(LIB_PATH) $(LIB_NAMES) -o $(BINARIES)/$(TARGET)$(VERSION)
	@rm -rf $(OBJ)
	
#compile
%.o: %.c
	$(CC) $(INCLUDES) $(DEBUG) -c $(CFLAGS) $< -o $@

.PHONY:clean
clean:
	# @echo "Remove linked and compiled files......"
	rm -rf $(OBJ) $(TARGET) $(BINARIES) 

run: $(TARGET)
	./$(BINARIES)/$(TARGET)$(VERSION)