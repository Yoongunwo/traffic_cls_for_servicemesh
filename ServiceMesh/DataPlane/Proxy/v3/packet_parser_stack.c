#include <stdint.h>
#include <string.h>

#define MAX_SESSIONS 65536
#define VEC_LEN 1479   // 19 header + 1460 payload
#define WIN_SIZE 15

static float session_buffers[MAX_SESSIONS][WIN_SIZE][VEC_LEN];
static int session_counts[MAX_SESSIONS] = {0};

int parse_tcp_packet(const uint8_t* raw, size_t len, float* out_vec) {
    if (len < 54) return -1;

    uint8_t ttl = raw[22];
    uint8_t proto = raw[23];
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
    const uint8_t* payload = (payload_start < len) ? &raw[payload_start] : NULL;
    int payload_len = (payload != NULL) ? (int)(len - payload_start) : 0;

    int idx = 0;
    out_vec[idx++] = ttl;
    out_vec[idx++] = proto;
    out_vec[idx++] = ip_flags;
    out_vec[idx++] = (frag_offset >> 8) & 0xFF;
    out_vec[idx++] = frag_offset & 0xFF;
    out_vec[idx++] = data_offset;
    out_vec[idx++] = flags;
    out_vec[idx++] = (window >> 8) & 0xFF;
    out_vec[idx++] = window & 0xFF;
    out_vec[idx++] = (urgptr >> 8) & 0xFF;
    out_vec[idx++] = urgptr & 0xFF;

    for (int i = 0; i < 4; ++i) out_vec[idx++] = seq[i];
    for (int i = 0; i < 4; ++i) out_vec[idx++] = ack[i];

    for (int i = 0; i < 1460; ++i) {
        out_vec[idx++] = (i < payload_len) ? payload[i] : 0;
    }

    return 0;
}

int parse_and_stack(const uint8_t* raw, size_t len, float* out_stack, uint32_t session_id) {
    if (session_id >= MAX_SESSIONS) return -1;

    float temp[VEC_LEN];
    if (parse_tcp_packet(raw, len, temp) != 0) return -1;

    int idx = session_counts[session_id];
    if (idx >= WIN_SIZE) {
        memmove(session_buffers[session_id], session_buffers[session_id] + 1, sizeof(float) * (WIN_SIZE - 1) * VEC_LEN);
        idx = WIN_SIZE - 1;
    }
    memcpy(session_buffers[session_id][idx], temp, sizeof(float) * VEC_LEN);
    session_counts[session_id] = idx + 1;

    if (session_counts[session_id] < WIN_SIZE) return 0;

    // ✅ 개선된 복사: 앞에서부터 1496개만 복사
    int count = 0;
    for (int i = 0; i < WIN_SIZE && count < 1496; ++i) {
        int copy_len = VEC_LEN;
        if (count + copy_len > 1496) {
            copy_len = 1496 - count;
        }
        memcpy(out_stack + count, session_buffers[session_id][i], sizeof(float) * copy_len);
        count += copy_len;
    }

    return 1;
}