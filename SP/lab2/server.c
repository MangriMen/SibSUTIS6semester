#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <errno.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdbool.h>
#include <signal.h>

#define BUFLEN 255
#define SOCKET_COUNT 256

void ProcessMessage(int sockClient)
{
    char buf[BUFLEN];
    int msgLength = 0;
    memset((char *)&buf, 0, sizeof(buf));

    if ((msgLength = recv(sockClient, buf, BUFLEN, 0)) < 0)
    {
        perror("Плохое получение дочерним процессом.");
        exit(1);
    }

    if (msgLength == 0)
    {
        printf("Сервер: клиент с номером %d разорвал соединение с сервером\n\n", sockClient);
        close(sockClient);
        exit(0);
    }

    printf("Сервер: Socket для клиента: %d\n", sockClient);
    printf("\tДлина сообщения: %d\n", msgLength);
    printf("\tСообщение: %s\n\n", buf);
}

int main()
{
    signal(SIGCHLD, SIG_IGN);

    int sockMain, length;
    struct sockaddr_in servAddr;

    if ((sockMain = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        perror("Сервер не может открыть главный socket.");
        exit(1);
    }

    memset((char *)&servAddr, 0, sizeof(servAddr));
    servAddr.sin_family = AF_INET;
    servAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servAddr.sin_port = 0;

    if (bind(sockMain, (struct sockaddr *)&servAddr, sizeof(servAddr)))
    {
        perror("Связывание сервера неудачно.");
        exit(1);
    }

    length = sizeof(servAddr);
    if (getsockname(sockMain, (struct sockaddr *)&servAddr, (socklen_t *restrict)&length))
    {
        perror("Вызов getsockname неудачен.");
        exit(1);
    }

    char string_buff[BUFLEN] = {'\0'};
    inet_ntop(AF_INET, &servAddr.sin_addr, string_buff, BUFLEN);
    printf("Сервер:\n");
    printf("\tIP:%s:%d\n\n", string_buff, ntohs(servAddr.sin_port));

    listen(sockMain, 5);

    while (true)
    {
        int socketClient = 0;
        if ((socketClient = accept(sockMain, 0, 0)) < 0)
        {
            perror("Неверный socket для клиента.");
            exit(1);
        }

        if (fork() == 0)
        {
            while (true)
            {
                ProcessMessage(socketClient);
            }
        }
    }

    return 0;
}