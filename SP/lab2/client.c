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

#define BUFLEN 255

int main(int argc, char *argv[])
{
    int sock = 0;
    struct sockaddr_in servAddr = {};
    struct hostent *hp, *gethostbyname();
    char *console_buffer = NULL;
    size_t console_len = 0;

    if (argc < 3)
    {
        printf("TCP Client. Введите: имя_хоста порт\n");
        exit(1);
    }

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        perror("He могу получить socket\n");
        exit(1);
    }

    hp = gethostbyname(argv[1]);

    memset((char *)&servAddr, 0, sizeof(servAddr));
    servAddr.sin_family = AF_INET;
    memcpy(&servAddr.sin_addr, hp->h_addr, hp->h_length);
    servAddr.sin_port = htons(atoi(argv[2]));

    if (connect(sock, (struct sockaddr *)&servAddr, sizeof(servAddr)) < 0)
    {
        perror("Клиент не может соединиться.\n");
        exit(1);
    }

    printf("Клиент: Готов к пересылке\n");

    while (true)
    {
        getline(&console_buffer, &console_len, stdin);
        console_buffer[strcspn(console_buffer, "\r\n")] = 0;
        if (strcmp(console_buffer, "/exit") == 0)
        {
            break;
        }

        if (send(sock, console_buffer, console_len, 0) < 0)
        {
            perror("Проблемы с пересылкой.\n");
            exit(1);
        }
    }

    printf("Клиент: Отключён от сервера.\n");

    close(sock);

    return 0;
}