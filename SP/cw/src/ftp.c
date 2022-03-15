#include <ftp.h>

const int FTP_CONTROL_PORT = 21;
static int socket_main = 0;
static int socket_data = 0;
static const char welcome_message[] = "220 myFTP\n";

char *replace_char(char *str, char find, char replace)
{
    char *current_pos = strchr(str, find);
    while (current_pos)
    {
        *current_pos = replace;
        current_pos = strchr(current_pos, find);
    }
    return str;
}

char **ftp_parse_message(char *message)
{
    char **out = (char **)malloc(2 * sizeof(char *));
    char delim[] = " \r\n";

    out[0] = strtok(message, delim);
    out[1] = strtok(NULL, delim);

    return out;
}

void ftp_signal_handler(int sig)
{
    if (sig == SIGINT)
    {
        printf("Shutdown ftp server\n");
        close(socket_data);
        close(socket_main);
    }
}

int ftp_create_bind_socket(int domain, int type, int protocol, int ip, uint16_t port)
{
    int new_socket = socket(domain, type, protocol);

    if (new_socket < 0)
    {
        perror("Error opening socket\n");
        return -1;
    }

    struct sockaddr_in new_socket_address = {.sin_family = domain, .sin_addr.s_addr = htonl(ip), .sin_port = htons(port)};
    socklen_t new_socket_address_length = sizeof(new_socket_address);

    if (bind(new_socket, (struct sockaddr *)&new_socket_address, new_socket_address_length))
    {
        perror("Error binding socket\n");
        return -1;
    }

    if (getsockname(new_socket, (struct sockaddr *)&new_socket_address, (socklen_t *)&new_socket_address_length))
    {
        perror("Error getting socket address\n");
        return -1;
    }

    return new_socket;
}

char *ftp_get_passive_address(int domain, const void *src)
{
    size_t address_length = (domain == AF_INET) ? INET_ADDRSTRLEN : INET6_ADDRSTRLEN;
    char *address = (char *)(calloc(address_length, sizeof(char)));

    inet_ntop(domain, src, address, address_length);

    return replace_char(address, '.', ',');
}

void ftp_get_passive_port(uint16_t port, uint16_t *p1, uint16_t *p2)
{
    *p1 = port / 256;
    *p2 = port % 256;
}

int ftp_create_main_listener(int domain, int ip)
{
    signal(SIGINT, ftp_signal_handler);
    signal(SIGSTOP, ftp_signal_handler);

    socket_main = ftp_create_bind_socket(domain, SOCK_STREAM, 0, ip, FTP_CONTROL_PORT);
    listen(socket_main, 4);

    char buf[256] = {'\0'};

    int res;

    int result_recv = 0;
    int result_send = 0;
    while (true)
    {

        int socket_client = accept(socket_main, NULL, NULL);
        if (socket_client < 0)
        {
            perror("Error connecting client.\n");
            return -1;
        }

        printf("Successfull client connetion\n");
        result_send = send(socket_client, welcome_message, sizeof(welcome_message), 0);

        while (true)
        {
            memset(buf, 0, sizeof(buf));

            result_recv = recv(socket_client, buf, sizeof(buf), 0);

            if (result_recv < 0)
            {
                perror("Error recieve data from client.\n");
                return -1;
            }
            else if (result_recv == 0)
            {
                printf("Client disconnected\n");
                break;
            }

            printf("Client: %s\n", buf);
            char **msg = ftp_parse_message(buf);

            if (strcmp(msg[0], "USER") == 0)
            {
                result_send = send(socket_client, "331 Need password\n", sizeof("331 Need password\n"), 0);

                if (result_send)
                {
                }
            }
            else if (strcmp(msg[0], "PASS") == 0)
            {
                result_send = send(socket_client, "220 User ok\n", sizeof("220 User ok\n"), 0);
            }
            else if (strcmp(msg[0], "SYST") == 0)
            {
                result_send = send(socket_client, "215 UNIX Type: L8\n", sizeof("215 UNIX Type: L8\n"), 0);
            }
            else if (strcmp(msg[0], "PWD") == 0)
            {
                char pwd[1024] = {'\0'};
                getcwd(pwd, sizeof(pwd));

                char answer[sizeof(pwd) + 10] = {'\0'};
                sprintf(answer, "257 \"%s\"\n", /*"/mnt/d/"*/ pwd);
                result_send = send(socket_client, answer, sizeof(answer), 0);
            }
            else if (strcmp(msg[0], "CWD") == 0)
            {
                char answer[1024] = {'\0'};
                sprintf(answer, "%s", msg[1]);
                chdir(answer);
                result_send = send(socket_client, "250\n", sizeof("250\n"), 0);
            }
            else if (strcmp(msg[0], "TYPE") == 0)
            {
                char answer[1024] = "257 Type is i\n";
                result_send = send(socket_client, answer, sizeof(answer), 0);
            }
            else if (strcmp(msg[0], "PASV") == 0)
            {
                close(socket_data);
                socket_data = ftp_create_bind_socket(domain, SOCK_STREAM, 0, ip, 0);
                listen(socket_data, 1);

                struct sockaddr_in data_address = {0};
                socklen_t data_address_length = sizeof(data_address);
                if (getsockname(socket_data, (struct sockaddr *)&data_address, (socklen_t *)&data_address_length))
                {
                    perror("Error getting socket address\n");
                    return -1;
                }

                char *address = ftp_get_passive_address(domain, &data_address.sin_addr);

                uint16_t p1 = 0;
                uint16_t p2 = 0;
                ftp_get_passive_port(ntohs(data_address.sin_port), &p1, &p2);

                char answer[1024] = {'\0'};
                sprintf(answer, "227 Entering Passive Mode (%s,%d,%d)\n", address, p1, p2);
                result_send = send(socket_client, answer, sizeof(answer), 0);

                res = accept(socket_data, NULL, NULL);
            }
            else if (strcmp(msg[0], "LIST") == 0)
            {
                if (res)
                {
                }

                char answer[1024] = {'\0'};
                sprintf(answer, "150\n");
                result_send = send(socket_client, answer, sizeof(answer), 0);

                sprintf(answer, "drwxrwxrwx 1 ftp ftp               0 Jan 16 00:02 BeamNG.drive 0.23.1.0\n");

                result_send = send(res, answer, sizeof(answer), 0);

                result_send = send(socket_client, "226\n", sizeof("226\n"), 0);

                // char data[3072] = {'\0'};
                // FILE *pf = popen("ls -lt", "r");

                // char str[256] = {'\0'};
                // fgets(str, 256, pf);
                // while (!feof(pf))
                // {
                //     if (fgets(str, 256, pf))
                //     {
                //         strcat(data, str);
                //         data[strlen(data)] = '\r';
                //         strcat(data, "\n");
                //     }
                // }
                // if (pclose(pf) != 0)
                // {
                //     fprintf(stderr, "Error: Failed to close command stream\n");
                // }
                // result_send = send(res, "drwxrwxrwx 1 ftp ftp               0 Jan 16 00:02 BeamNG.drive 0.23.1.0\r\n", sizeof("drwxrwxrwx 1 ftp ftp               0 Jan 16 00:02 BeamNG.drive 0.23.1.0\r\n"), 0);
                // shutdown(res, SHUT_RDWR);
                // sleep(1);

                // result_send = send(socket_client, "451\n", sizeof("451\n"), 0);
            }
            else if (strcmp(msg[0], "QUIT") == 0)
            {
                close(socket_client);
                close(socket_data);
            }
            else
            {
                result_send = send(socket_client, "500 Command not implemented.\n", sizeof("502 Command not implemented.\n"), 0);
            }

            free(msg);
        }
    }

    printf("Shutdown ftp server\n");
    close(socket_main);

    return 0;
}