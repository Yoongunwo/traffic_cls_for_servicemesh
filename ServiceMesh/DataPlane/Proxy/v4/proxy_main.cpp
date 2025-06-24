#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <netinet/tcp.h>
#include <torch/script.h>
#include <netdb.h>
#include "packet_parser_stack.h"

#define MAX_SESSIONS 65536
#define VEC_LEN 1479
#define WIN_SIZE 15
#define OUT_LEN 1496

float out_stack[WIN_SIZE * VEC_LEN];
torch::jit::script::Module model;

void handle_connection(int client_sock, const std::string& target_ip, int target_port) {
    int server_sock = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(target_port);
    inet_pton(AF_INET, target_ip.c_str(), &server_addr.sin_addr);

    if (connect(server_sock, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("Connect failed");
        close(client_sock);
        return;
    }

    fd_set readfds;
    uint8_t buf[16384];

    while (true) {
        FD_ZERO(&readfds);
        FD_SET(client_sock, &readfds);
        FD_SET(server_sock, &readfds);
        int maxfd = std::max(client_sock, server_sock) + 1;
        if (select(maxfd, nullptr, nullptr, nullptr, nullptr) <= 0) break;

        if (FD_ISSET(client_sock, &readfds)) {
            ssize_t len = recv(client_sock, buf, sizeof(buf), 0);
            if (len <= 0) break;

            uint32_t src_ip = ntohl(*reinterpret_cast<uint32_t*>(&buf[26]));
            uint32_t dst_ip = ntohl(*reinterpret_cast<uint32_t*>(&buf[30]));
            uint16_t src_port = ntohs(*reinterpret_cast<uint16_t*>(&buf[34]));
            uint16_t dst_port = ntohs(*reinterpret_cast<uint16_t*>(&buf[36]));
            uint8_t proto = buf[23];
            uint32_t session_id = (src_ip ^ dst_ip ^ src_port ^ dst_port ^ proto) % MAX_SESSIONS;

            int ret = parse_and_stack(buf, len, out_stack, session_id);
            std::cout << "[LOG] parse_and_stack() returned: " << ret << "\n";

            if (ret == 1) {
                std::vector<float> padded(34 * 44, 0.0f);
                for (int i = 0; i < std::min(OUT_LEN, (int)padded.size()); ++i) {
                    if (!std::isfinite(out_stack[i])) {
                        std::cerr << "[ERROR] NaN or INF detected in out_stack[" << i << "]: " << out_stack[i] << "\n";
                        continue;
                    }
                    padded[i] = out_stack[i] / 255.0f;
                }

                try {
                    auto input = torch::from_blob(padded.data(), {1, 1, 34, 44}, torch::kFloat32).clone();
                    auto output = model.forward({input}).toTensor();
                    std::cout << "[Model Output]: " << output << std::endl;
                } catch (const c10::Error& e) {
                    std::cerr << "[ERROR] Inference failed: " << e.what() << std::endl;
                }
            }

            send(server_sock, buf, len, 0);
        }

        if (FD_ISSET(server_sock, &readfds)) {
            ssize_t len = recv(server_sock, buf, sizeof(buf), 0);
            if (len <= 0) break;
            send(client_sock, buf, len, 0);
        }
    }

    close(server_sock);
    close(client_sock);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <model.pt> <proxy_port> <target_ip>\n";
        return 1;
    }

    init_buffers();
    try {
        model = torch::jit::load(argv[1]);
        model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Model load error: " << e.what() << std::endl;
        return 1;
    }

    int proxy_port = std::stoi(argv[2]);
    std::string target_ip = argv[3];

    // IP 변환
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    struct hostent* host_entry = gethostbyname(hostname);
    std::string local_ip = inet_ntoa(*((struct in_addr*)host_entry->h_addr_list[0]));
    if (inet_addr(argv[3]) == INADDR_LOOPBACK || argv[3] == local_ip) {
        target_ip = "127.0.0.1";
    }

    int listen_sock = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in proxy_addr{};
    proxy_addr.sin_family = AF_INET;
    proxy_addr.sin_port = htons(proxy_port);
    proxy_addr.sin_addr.s_addr = INADDR_ANY;
    bind(listen_sock, (sockaddr*)&proxy_addr, sizeof(proxy_addr));
    listen(listen_sock, 64);

    while (true) {
        sockaddr_in client_addr{};
        socklen_t addrlen = sizeof(client_addr);
        int client_sock = accept(listen_sock, (sockaddr*)&client_addr, &addrlen);
        std::cout << "[LOG] Accepted new client connection\n";
        if (client_sock >= 0)
            handle_connection(client_sock, target_ip, 80);
    }

    free_buffers();
    return 0;
}
