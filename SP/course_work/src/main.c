#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <ftp.h>

int main()
{
    ftp_create_main_listener(AF_INET, INADDR_ANY);
    return EXIT_SUCCESS;
}