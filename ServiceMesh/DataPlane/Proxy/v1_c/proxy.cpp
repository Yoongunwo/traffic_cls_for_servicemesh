// === Optimized C++ INFERENCE PROXY (Single File, select()-based) ===
// Compile: g++ -O3 -std=c++17 proxy_select.cpp -o proxy

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

constexpr int VEC_LEN = 1479;
constexpr int WIN_SIZE = 5;
constexpr int H = 1479, W = 5;
constexpr float THRESHOLD = 0.5f;

std::unordered_map<uint32_t, std::vector<std::vector<float>>> session_windows;
std::mutex session_mutex;
torch::jit::script::Module model;

int parse_tcp_packet(const uint8_t* raw, size_t len, float* out_vec) {
    if (len < 54) return -1;
    uint8_t ttl = raw[22], proto = raw[23];
    uint16_t flags_frag = (raw[20] << 8) | raw[21];
    uint8_t ip_flags = (flags_frag >> 13) & 0x7;
    uint16_t frag_offset = flags_frag & 0x1FFF;
    int tcp_off = 14 + 20;
    if (len < tcp_off + 20) return -1;
    uint8_t data_offset = (raw[tcp_off + 12] >> 4) & 0xF;
    uint8_t flags = raw[tcp_off + 13];
    uint16_t window = (raw[tcp_off + 14] << 8) | raw[tcp_off + 15];
    uint16_t urgptr = (raw[tcp_off + 18] << 8) | raw[tcp_off + 19];
    const uint8_t* seq = &raw[tcp_off + 4];
    const uint8_t* ack = &raw[tcp_off + 8];
    int payload_start = tcp_off + data_offset * 4;
    const uint8_t* payload = (payload_start < len) ? &raw[payload_start] : nullptr;
    int payload_len = (payload != nullptr) ? (int)(len - payload_start) : 0;
    int idx = 0;
    out_vec[idx++] = ttl; out_vec[idx++] = proto; out_vec[idx++] = ip_flags;
    out_vec[idx++] = (frag_offset >> 8) & 0xFF; out_vec[idx++] = frag_offset & 0xFF;
    out_vec[idx++] = data_offset; out_vec[idx++] = flags;
    out_vec[idx++] = (window >> 8) & 0xFF; out_vec[idx++] = window & 0xFF;
    out_vec[idx++] = (urgptr >> 8) & 0xFF; out_vec[idx++] = urgptr & 0xFF;
    for (int i = 0; i < 4; ++i) out_vec[idx++] = seq[i];
    for (int i = 0; i < 4; ++i) out_vec[idx++] = ack[i];
    for (int i = 0; i < 1460; ++i) out_vec[idx++] = (i < payload_len) ? payload[i] : 0;
    return 0;
}

void detect_anomaly(const uint8_t* raw, size_t len, uint32_t session_id) {
    std::cout << "hi" << std::endl;
    float packet[VEC_LEN];
    if (parse_tcp_packet(raw, len, packet) != 0){
        std::cout << "Failed to parse packet" << std::endl;
        return;
    }
    std::cout << "Packet parsed successfully" << std::endl;
        
    std::lock_guard<std::mutex> lock(session_mutex);
    auto& window = session_windows[session_id];
    window.emplace_back(packet, packet + VEC_LEN);
    if (window.size() < WIN_SIZE) return;

    torch::Tensor input = torch::empty({1, 1, H, W}, torch::kFloat32);
    for (int w = 0; w < W; ++w)
        memcpy(input[0][0][0].data_ptr<float>() + w * H, window[w].data(), H * sizeof(float));
    window.erase(window.begin());

    auto feat = model.forward({input}).toTensor();
    float score = feat.mean().item<float>();
    std::cout << (score < THRESHOLD ? "Anomaly" : "Normal") << " (" << score << ")\n";
}

int make_socket_non_blocking(int sockfd) {
    int flags = fcntl(sockfd, F_GETFL, 0);
    return fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);
}

int start_proxy(int proxy_port, const std::string& target_ip, int target_port) {
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
        fd_set read_fds, write_fds;
        FD_ZERO(&read_fds);
        FD_ZERO(&write_fds);
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

        int ready = select(max_fd + 1, &read_fds, &write_fds, nullptr, nullptr);
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
                        target_addr.sin_port = htons(target_port);
                        inet_pton(AF_INET, target_ip.c_str(), &target_addr.sin_addr);
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
                // anomaly detection
                uint32_t session_id = i % MAX_CLIENTS;
                detect_anomaly((const uint8_t*)buffer, len, session_id);

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
    if (argc < 5) {
        std::cerr << "Usage: ./proxy <model.pt> <target_ip> <target_port> <proxy_port>\n";
        return 1;
    }
    model = torch::jit::load(argv[1]);
    model.eval();
    return start_proxy(std::stoi(argv[4]), argv[2], std::stoi(argv[3]));
}
