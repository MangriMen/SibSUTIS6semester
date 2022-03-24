import os
import shutil
import socket
import threading


def caseless_equal(a, b):
    return a.casefold() == b.casefold()


class FTPServer:
    def __init__(self, domain) -> None:
        self.__is_running = False
        self.__socket_main = None
        self.FTP_PORT = 21

        self.__domain = domain
        self.__encoding = "ascii"
        self.__clients_threads = []

    def start(self):
        self.__is_running = True
        self.__listener()

    def stop(self):
        self.__is_running = False

    def send_msg(self, conn, msg):
        print(f"A: {msg}")
        return conn.send(msg.encode(self.__encoding, 'ignore')) if conn is not None else None

    def __create_binded_socket(self, ip, port):
        sock = socket.socket(self.__domain, socket.SOCK_STREAM)
        sock.bind((ip, port))
        return sock

    def __parse_message(self, msg):
        print(f"R: {msg}")
        newmsg = msg.decode(self.__encoding, 'ignore').strip().split(" ", 1)
        if len(newmsg) < 2:
            newmsg.append("")
        return newmsg

    @staticmethod
    def __get_passive_port(port):
        return (int(port / 256), int(port % 256))

    @staticmethod
    def __get_active_ip_port(address):
        parsed = address.split(",")
        ip = "%s.%s.%s.%s" % (parsed[0], parsed[1], parsed[2], parsed[3])
        port = int(parsed[4]) * 256 + int(parsed[5])
        return (ip, port)

    def __listener(self):
        self.__socket_main = self.__create_binded_socket(
            "127.0.0.1", self.FTP_PORT)
        self.__socket_main.listen()

        while self.__is_running:
            conn, addr = self.__socket_main.accept()

            print(f"Connected by {addr}")

            start_dir = "/" if os.name != "nt" else "C:/"

            self.__clients_threads.append(threading.Thread(
                target=self.__client_processing, args=(conn, start_dir)))

            self.__clients_threads[-1].start()

    def __client_processing(self, conn, start_dir):
        ctx = dict(
            cwd=start_dir,
            data_conn=None,
            last_cmd=""
        )
        os.chdir(ctx["cwd"])

        with conn:
            self.__send_hello_message(conn)
            while self.__is_running:
                message = conn.recv(1024)
                if not message:
                    return

                command, data = self.__parse_message(message)
                ctx["last_cmd"] = command

                if caseless_equal(command, "USER"):
                    self.__cmd_user(conn, ctx, data)
                elif caseless_equal(command, "PASS"):
                    self.__cmd_pass(conn, ctx, data)
                elif caseless_equal(command, "SYST"):
                    self.__cmd_syst(conn)
                elif caseless_equal(command, "FEAT"):
                    self.__cmd_feat(conn)
                elif caseless_equal(command, "OPTS"):
                    self.__cmd_opts(conn, ctx, data)
                elif caseless_equal(command, "DELE"):
                    self.__cmd_dele(conn, data)
                elif caseless_equal(command, "RMD"):
                    self.__cmd_rmd(conn, data)
                elif caseless_equal(command, "MKD"):
                    self.__cmd_mkd(conn, data)
                elif caseless_equal(command, "PWD") or caseless_equal(command, "XPWD"):
                    self.__cmd_pwd(conn)
                elif caseless_equal(command, "CWD") or caseless_equal(command, "XCWD"):
                    self.__cmd_cwd(conn, ctx, data)
                elif caseless_equal(command, "TYPE"):
                    self.__cmd_type(conn, ctx, data)
                elif caseless_equal(command, "RETR"):
                    self.__cmd_retr(conn, ctx, data)
                elif caseless_equal(command, "STOR"):
                    self.__cmd_stor(conn, ctx, data)
                elif caseless_equal(command, "APPE"):
                    self.__cmd_appe(conn, ctx, data)
                elif caseless_equal(command, "RNFR"):
                    self.__cmd_rnfr(conn, ctx, data)
                elif caseless_equal(command, "RNTO"):
                    self.__cmd_rnto(conn, ctx, data)
                elif caseless_equal(command, "STOU"):
                    self.__cmd_stou(conn, ctx, data)
                elif caseless_equal(command, "PORT"):
                    self.__cmd_port(conn, ctx, data)
                elif caseless_equal(command, "PASV"):
                    self.__cmd_pasv(conn, ctx)
                elif caseless_equal(command, "LIST"):
                    self.__cmd_list(conn, ctx, data)
                elif caseless_equal(command, "NLST"):
                    self.__cmd_nlst(conn, ctx, data)
                elif caseless_equal(command, "NOOP"):
                    self.__cmd_noop(conn)
                elif caseless_equal(command, "QUIT"):
                    self.__cmd_quit(conn)
                    break
                else:
                    self.__cmd_not_implemented(conn)

    def __send_hello_message(self, conn):
        self.send_msg(conn, "200 myFTP\r\n")

    def __cmd_user(self, conn, ctx, data):
        self.send_msg(conn, "331 Need password\r\n")

    def __cmd_pass(self, conn, ctx, data):
        self.send_msg(conn, "220 User ok\r\n")

    def __cmd_cwd(self, conn, ctx, data):
        starts_with_root = "/" if os.name != "nt" else ctx["cwd"].split(":")[
            0] + ":"

        if ctx["cwd"].startswith(starts_with_root) \
                or ctx["cwd"].startwith(".."):
            ctx["cwd"] = data
        else:
            ctx["cwd"] = os.path.normpath(
                starts_with_root + os.path.join(ctx["cwd"], data))

        print(f"CWD: {ctx['cwd']}")
        try:
            os.chdir(ctx["cwd"])
            self.send_msg(conn, "250 Okay\r\n")
        except FileNotFoundError:
            self.send_msg(
                conn, "550 Requested action not taken; file unavailable...\r\n")

    def __cmd_port(self, conn, ctx, data):
        ctx["data_conn"] = self.__create_binded_socket("127.0.0.1", 0)
        ctx["data_conn"].connect(self.__get_active_ip_port(data))

        self.send_msg(conn, "220 Connected to client\r\n")

    def __cmd_pasv(self, conn, ctx):
        data_socket = self.__create_binded_socket("127.0.0.1", 0)
        data_socket.listen()

        address, port = data_socket.getsockname()

        address = address.replace(".", ",")
        p1, p2 = self.__get_passive_port(port)

        self.send_msg(
            conn, f"227 Entering Passive Mode ({address},{p1},{p2})\r\n")

        ctx["data_conn"], _ = data_socket.accept()

    def __cmd_type(self, conn, ctx, data):
        if caseless_equal(data, "A"):
            self.send_msg(conn, "257 TYPE is A\r\n")
        elif caseless_equal(data, "I"):
            self.send_msg(conn, "257 TYPE is I\r\n")
        else:
            self.send_msg(
                conn, "500 Command not implemented\r\n")

    def __cmd_retr(self, conn, ctx, data):
        self.send_msg(conn, "150 About to start data transfer\r\n")

        with open(data, 'rb') as file:
            part = file.read(4096)
            while(part):
                ctx["data_conn"].send(part)
                part = file.read(4096)

        self.send_msg(conn, "226 Operation successful\r\n")
        ctx["data_conn"].close()

    def __cmd_stor(self, conn, ctx, data):
        self.send_msg(conn, "150 About to start data transfer\r\n")

        with open(data, 'wb') as file:
            part = ctx["data_conn"].recv(4096)
            while(part):
                file.write(part)
                part = ctx["data_conn"].recv(4096)

        self.send_msg(conn, "226 Operation successful\r\n")
        ctx["data_conn"].close()

    def __cmd_stou(self, conn, ctx, data):
        newpath = data
        if os.path.exists(newpath):
            noext_path, ext = os.path.splitext(data)
            number = "copy"
            while(os.path.exists(newpath := f"{noext_path}{number}{ext}")):
                number += "copy"
        self.__cmd_stor(conn, ctx, newpath)

    def __cmd_appe(self, conn, ctx, data):
        self.send_msg(conn, "150 About to start data transfer\r\n")

        with open(data, 'ab') as file:
            part = ctx["data_conn"].recv(4096)
            while(part):
                file.write(part)
                part = ctx["data_conn"].recv(4096)

        self.send_msg(conn, "226 Operation successful\r\n")
        ctx["data_conn"].close()

    def __cmd_rnfr(self, conn, ctx, data):
        ctx["rnfr"] = data
        self.send_msg(
            conn, "350 Requested file action pending further information.\r\n")

    def __cmd_rnto(self, conn, ctx, data):
        shutil.move(ctx["rnfr"], data)
        self.send_msg(conn, "250 Requested file action okay, completed.\r\n")

    def __cmd_abor(self, conn, data):
        pass

    def __cmd_dele(self, conn, data):
        if os.path.exists(data):
            os.remove(data)
            self.send_msg(
                conn, "250 Requested file action okay, completed\r\n")
        else:
            self.send_msg(
                conn, "550 Requested action not taken; file unavailable...\r\n")

    def __cmd_rmd(self, conn, data):
        if os.path.exists(data):
            shutil.rmtree(data)
            self.send_msg(
                conn, "250 Requested file action okay, completed\r\n")
        else:
            self.send_msg(
                conn, "550 Requested action not taken; file unavailable...\r\n")

    def __cmd_mkd(self, conn, data):
        os.mkdir(data)
        self.send_msg(conn, f"257 \"{data}\" created\r\n")

    def __cmd_pwd(self, conn):
        self.send_msg(conn, f"257 {os.getcwd()}\r\n")

    def __cmd_list(self, conn, ctx, data):
        self.send_msg(
            conn, "150 About to start data transfer\r\n")

        work_dir = data if os.path.join(
            ctx["cwd"], data) != ctx["cwd"] else ctx["cwd"]

        if os.name == "nt":
            ls_out = [f"{line.strip()}\r\n" for line in os.popen(f"dir")]
            ls_out = [nline for nline in [ls_out[i] for i in range(
                5, len(ls_out) - 2)] if (".." not in nline) or ("." not in nline)]
        else:
            ls_out = [f"{line.strip()}\r\n" for line in os.popen(
                f"/bin/ls -l {work_dir} | tail --lines=+2")]

        ls_message = ''.join(ls_out)

        self.send_msg(ctx["data_conn"], f"{ls_message}")

        self.send_msg(conn, "226 Operation successful\r\n")
        ctx["data_conn"].close()

    def __cmd_nlst(self, conn, ctx, data):
        self.send_msg(
            conn, "150 About to start data transfer\r\n")

        work_dir = data if os.path.join(
            ctx["cwd"], data) != ctx["cwd"] else ctx["cwd"]

        ls_out = [f"{line.strip()}\r\n" for line in os.listdir(work_dir)]
        ls_message = ''.join(ls_out)

        self.send_msg(ctx["data_conn"], f"{ls_message}")

        self.send_msg(conn, "226 Operation successful\r\n")
        ctx["data_conn"].close()

    def __cmd_quit(self, conn):
        self.send_msg(conn, "200 Goodbye\r\n")
        conn.close()

    def __cmd_syst(self, conn):
        self.send_msg(conn, "215 UNIX Type: L8\r\n")

    def __cmd_feat(self, conn):
        self.send_msg(
            conn, "211 feature-listing\r\n UTF-8\r\n211 END\r\n")

    def __cmd_opts(self, conn, ctx, data):
        command, args = data.split(" ", 1)
        if caseless_equal(command, "UTF8"):
            if caseless_equal(args, "ON"):
                self.__encoding = "utf-8"
            else:
                self.__encoding = "ascii"
        self.send_msg(conn, "200 Okay\r\n")

    def __cmd_noop(self, conn):
        self.send_msg(conn, "200 Ok\r\n")

    def __cmd_not_implemented(self, conn):
        self.send_msg(conn, "500 Command not implemented\r\n")


def main():
    ftp_server = FTPServer(socket.AF_INET)
    ftp_server.start()


if __name__ == "__main__":
    main()
