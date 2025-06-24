// === Simple TCP Proxy (Same Pod, localhost:80 target) ===
// Compile: g++ -O3 -std=c++17 proxy.cpp -o proxy

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/select.h>

constexpr int MAX_CLIENTS = 1024;
constexpr int BUF_SIZE = 16384;

int make_socket_non_blocking(int sockfd) {
    int flags = fcntl(sockfd, F_GETFL, 0);
    return fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);
}

int start_proxy(int proxy_port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(proxy_port);
    server_addr.sin_addr.s_addr = INADDR_ANY;
    bind(server_fd, (sockaddr*)&server_addr, sizeof(server_addr));
    listen(server_fd, 128);
    make_socket_non_blocking(server_fd);
    std::cout << "[+] Listening on port " << proxy_port << std::endl;

    int client_fds[MAX_CLIENTS];
    int target_fds[MAX_CLIENTS];
    memset(client_fds, -1, sizeof(client_fds));
    memset(target_fds, -1, sizeof(target_fds));

    while (true) {
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(server_fd, &read_fds);
        int max_fd = server_fd;

        for (int i = 0; i < MAX_CLIENTS; ++i) {
            if (client_fds[i] != -1) {
                FD_SET(client_fds[i], &read_fds);
                max_fd = std::max(max_fd, client_fds[i]);
            }
            if (target_fds[i] != -1) {
                FD_SET(target_fds[i], &read_fds);
                max_fd = std::max(max_fd, target_fds[i]);
            }
        }

        int ready = select(max_fd + 1, &read_fds, nullptr, nullptr, nullptr);
        if (ready < 0) continue;

        if (FD_ISSET(server_fd, &read_fds)) {
            sockaddr_in client_addr;
            socklen_t len = sizeof(client_addr);
            int client_fd = accept(server_fd, (sockaddr*)&client_addr, &len);
            if (client_fd != -1) {
                make_socket_non_blocking(client_fd);
                for (int i = 0; i < MAX_CLIENTS; ++i) {
                    if (client_fds[i] == -1) {
                        client_fds[i] = client_fd;
                        target_fds[i] = socket(AF_INET, SOCK_STREAM, 0);
                        sockaddr_in target_addr{};
                        target_addr.sin_family = AF_INET;
                        target_addr.sin_port = htons(80);
                        inet_pton(AF_INET, "127.0.0.1", &target_addr.sin_addr);
                        connect(target_fds[i], (sockaddr*)&target_addr, sizeof(target_addr));
                        make_socket_non_blocking(target_fds[i]);
                        std::cout << "[+] New client accepted (slot " << i << ")\n";
                        break;
                    }
                }
            }
        }

        for (int i = 0; i < MAX_CLIENTS; ++i) {
            if (client_fds[i] != -1 && FD_ISSET(client_fds[i], &read_fds)) {
                char buffer[BUF_SIZE];
                ssize_t len = recv(client_fds[i], buffer, BUF_SIZE, 0);
                if (len <= 0) {
                    close(client_fds[i]);
                    close(target_fds[i]);
                    client_fds[i] = target_fds[i] = -1;
                    continue;
                }
                send(target_fds[i], buffer, len, 0);
            }

            if (target_fds[i] != -1 && FD_ISSET(target_fds[i], &read_fds)) {
                char buffer[BUF_SIZE];
                ssize_t len = recv(target_fds[i], buffer, BUF_SIZE, 0);
                if (len <= 0) {
                    close(client_fds[i]);
                    close(target_fds[i]);
                    client_fds[i] = target_fds[i] = -1;
                    continue;
                }
                send(client_fds[i], buffer, len, 0);
            }
        }
    }
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./proxy <proxy_port>\n";
        return 1;
    }
    int proxy_port = std::stoi(argv[1]);
    return start_proxy(proxy_port);
}