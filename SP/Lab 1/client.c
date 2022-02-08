#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <unistd.h>
#include <stdbool.h>

#define BUFFLEN 255

int main(int argc, char *argv[])
{
    int sock = 0;
    char buffer[BUFFLEN] = {'\0'};
    struct sockaddr_in serverAddr = {}, clientAddr = {};
    struct hostent *hp, *gethostbyname();

    if (argc < 4)
    {
        printf("Ввести имя_хоста, порт и сообщение.\n");
        exit(1);
    }

    if ((sock = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
    {
        perror("Не получен socket");
        exit(1);
    }

    hp = gethostbyname(argv[1]);

    memset((char *)&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    memcpy(&serverAddr.sin_addr, hp->h_addr, hp->h_length);
    serverAddr.sin_port = htons(atoi(argv[2]));

    memset((char *)&clientAddr, 0, sizeof(clientAddr));
    clientAddr.sin_family = AF_INET;
    clientAddr.sin_addr.s_addr = INADDR_ANY;
    clientAddr.sin_port = 0;

    printf("Клиент: Готов к пересылке.\n");
    while (true)
    {
        if (sendto(sock, argv[3], strlen(argv[3]), 0, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) < 0)
        {
            perror("Проблемы с отправкой на сервер.");
            exit(1);
        }

        memset((char *)&buffer, 0, BUFFLEN);
        int slen = sizeof(serverAddr);

        if ((recvfrom(sock, (char *)buffer, BUFFLEN, 0,
                      (struct sockaddr *)&serverAddr, (socklen_t *)&slen)) == -1)
        {
            perror("Неверный сокет сервера.");
            exit(1);
        }

        printf("Возврат: %s\n", buffer);

        sleep(atoi(argv[3]));
    }

    printf("Клиент: Пересылка завершена.\n");
    close(sock);

    return 0;
}