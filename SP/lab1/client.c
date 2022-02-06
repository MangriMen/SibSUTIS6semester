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
    int sock;
    char buffer[BUFFLEN];
    struct sockaddr_in serverAddr, clientAddr;
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

    bzero((char *)&serverAddr, sizeof(serverAddr));

    serverAddr.sin_family = AF_INET;
    hp = gethostbyname(argv[1]);
    bcopy(hp->h_addr, &serverAddr.sin_addr, hp->h_length);
    serverAddr.sin_port = htons(atoi(argv[2]));

    bzero((char *)&clientAddr, sizeof(clientAddr));

    clientAddr.sin_family = AF_INET;
    clientAddr.sin_addr.s_addr = INADDR_ANY;
    clientAddr.sin_port = 0;

    // if (bind(sock, &clientAddr, sizeof(clientAddr))){
    //     perror("Клиент не получил порт.");
    //     exit(1);
    // }

    printf("Клиент: Готов к пересылке.\n");
    while (true)
    {
        if (sendto(sock, argv[3], strlen(argv[3]), 0, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) < 0)
        {
            perror("Проблемы с отправкой на сервер.");
            exit(1);
        }

        bzero(buffer, sizeof(BUFFLEN));
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
}