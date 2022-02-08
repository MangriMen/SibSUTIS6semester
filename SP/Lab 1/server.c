#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <errno.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

#define BUFFLEN 255

void processingMessage(char *in_message, char *out_message, int size)
{
    for (int i = 0; i < size; ++i)
    {
        out_message[i] = toupper(in_message[i]);
    }
}

int main()
{
    int socketMain = 0, msgLength = 0;
    struct sockaddr_in serverAddr = {}, clientAddr = {};
    char string_buff[BUFFLEN] = {'\0'};
    char in_buff[BUFFLEN] = {'\0'};
    char out_buff[BUFFLEN] = {'\0'};

    if ((socketMain = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
    {
        perror("Сервер не в состоянии открыть сокет.");
        exit(1);
    }

    memset((char *)&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = 0;

    if (bind(socketMain, (const struct sockaddr *)&serverAddr, sizeof(serverAddr)))
    {
        perror("Связывание сервера прошло неудачно.");
        exit(1);
    }

    socklen_t length = sizeof(serverAddr);
    if (getsockname(socketMain, (struct sockaddr *)&serverAddr, &length))
    {
        perror("Вызов getsockname неудачен.");
        exit(1);
    }

    inet_ntop(AF_INET, &serverAddr.sin_addr, string_buff, BUFFLEN);
    printf("Сервер:\n");
    printf("\tIP: %s:%d\n", string_buff, ntohs(serverAddr.sin_port));

    listen(socketMain, 2);

    while (1)
    {
        length = sizeof(clientAddr);
        memset(in_buff, 0, BUFFLEN);
        memset(out_buff, 0, BUFFLEN);
        memset(string_buff, 0, BUFFLEN);

        if ((msgLength = recvfrom(socketMain, in_buff, BUFFLEN, 0, (struct sockaddr *)&clientAddr, &length)) < 0)
        {
            perror("Неверный сокет клиента.");
            exit(1);
        }

        processingMessage(in_buff, out_buff, msgLength);

        if (sendto(socketMain, out_buff, msgLength, 0, (struct sockaddr *)&clientAddr, length) < 0)
        {
            perror("Проблемы с отправкой на клиент.");
            exit(1);
        }

        inet_ntop(AF_INET, &clientAddr.sin_addr, string_buff, BUFFLEN);
        printf("Сервер:\n");
        printf("\tIP клиента: %s:%d\n", string_buff, ntohs(clientAddr.sin_port));
        printf("\tДлина сообщения: %d\n", msgLength);
        printf("\tСообщение: %s\n\n", in_buff);
    }
}