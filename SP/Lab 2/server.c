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

#define BUFLEN 81
#define SOCKET_COUNT 256

void BuffWork(int sockClient)
{
    char buf[BUFLEN];
    int msgLength;
    bzero(buf, BUFLEN);

    if ((msgLength = recv(sockClient, buf, BUFLEN, 0)) < 0)
    {
        perror("Плохое получение дочерним процессом.");
        exit(1);
    }

    printf("SERVER: Socket дляклиента-%d\n", sockClient);
    printf("SERVER: Длинасообщения-%d\n", msgLength);
    printf("SERVER: Сообщение: %s\n\n", buf);
}

int main()
{
    int sockMain, length;
    struct sockaddr_in servAddr;

    if ((sockMain = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        perror("Сервер не может открыть главный socket.");
        exit(1);
    }

    bzero((char *)&servAddr, sizeof(servAddr));

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

    printf("СЕРВЕР: номер порта-% d\n", ntohs(servAddr.sin_port));
    listen(sockMain, 5);

    int *sockClients = (int *)malloc(sizeof(int) * SOCKET_COUNT);
    int *sockClientsPIDs = (int *)malloc(sizeof(int) * SOCKET_COUNT);
    int sockClientsCounter = 0;
    while (true)
    {
        if (sockClientsCounter > 255)
        {
            perror("Кончились socketы.");
            exit(1);
        }

        if ((sockClients[sockClientsCounter] = accept(sockMain, 0, 0)) < 0)
        {
            perror("Неверный socket для клиента.");
            exit(1);
        }

        int forkedPID = fork();
        if (forkedPID > 0)
        {
            sockClientsPIDs[sockClientsCounter] = forkedPID;
        }
        else if (forkedPID == 0)
        {
            BuffWork(sockClients[sockClientsCounter]);
        }

        // close(sockClient);
        sockClientsCounter++;
    }

    for (int i = 0; i < sockClientsCounter; ++i)
    {
        close(sockClients[i]);
        kill(sockClientsPIDs[i]);
    }

    return 0;
}
