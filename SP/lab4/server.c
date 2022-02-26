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
#include <sys/select.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdbool.h>
#include <signal.h>

#define BUFLEN 255
#define SOCKET_COUNT 256

int main()
{
    int sockMain = 0;
    int length = 0;
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

    char buf[BUFLEN] = {'\0'};
    fd_set master_rfds, rfds;
    int max_socket = sockMain;

    FD_ZERO(&master_rfds);
    FD_SET(sockMain, &master_rfds);

    while (true)
    {
        rfds = master_rfds;
        if (select(max_socket + 1, &rfds, NULL, NULL, NULL))
        {
            for (int i = 0; i <= max_socket; i++)
            {
                if (FD_ISSET(i, &rfds))
                {
                    if (i == sockMain)
                    {
                        int newfd = 0;
                        if ((newfd = accept(sockMain, NULL, NULL)) < 0)
                        {
                            perror("Failed to accept connection");
                        }
                        else
                        {
                            FD_SET(newfd, &master_rfds);
                            max_socket = newfd > max_socket ? newfd : max_socket;
                        }
                    }
                    else
                    {
                        memset((char *)&buf, 0, sizeof(buf));
                        int msgLength = recv(i, buf, BUFLEN, 0);

                        if (msgLength < 0)
                        {
                            perror("Плохое получение дочерним процессом.");
                        }
                        else if (msgLength == 0)
                        {
                            printf("Сервер: клиент с номером %d разорвал соединение с сервером\n\n", i);
                            close(i);
                            FD_CLR(i, &master_rfds);
                        }
                        else
                        {
                            printf("Сервер: Socket для клиента: %d\n", i);
                            printf("\tДлина сообщения: %d\n", msgLength);
                            printf("\tСообщение: %s\n\n", buf);
                        }
                    }
                }
            }
        }
    }

    return 0;
}