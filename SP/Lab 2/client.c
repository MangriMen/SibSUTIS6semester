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

int main(int argc, char *argv[])
{
    int sock;
    struct sockaddr_in servAddr;
    struct hostent *hp, *gethostbyname();

    if (argc < 4)
    {
        printf("ВВЕСТИ tcpclientимя_хоста порт сообщение\n");
        exit(1);
    }

    while (true)
    {
        if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
        {
            perror("He могу получить socket\n");
            exit(1);
        }

        bzero((char *)&servAddr, sizeof(servAddr));
        servAddr.sin_family = AF_INET;
        hp = gethostbyname(argv[1]);
        bcopy(hp->h_addr, &servAddr.sin_addr, hp->h_length);
        servAddr.sin_port = htons(atoi(argv[2]));

        if (connect(sock, (struct sockaddr *)&servAddr, sizeof(servAddr)) < 0)
        {
            perror("Клиент не может соединиться.\n");
            exit(1);
        }

        printf("CLIENT: Готов к пересылке\n");

        if (send(sock, argv[3], strlen(argv[3]), 0) < 0)
        {
            perror("Проблемы с пересылкой.\n");
            exit(1);
        }

        printf("CLIENT: Пересылка завершена. Счастливо оставаться.\n");
        close(sock);

        getchar();
    }

    return 0;
}