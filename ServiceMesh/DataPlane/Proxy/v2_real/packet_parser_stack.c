// gcc -shared -fPIC -o v1/packet_parser_stack.so v1/packet_parser_stack.c

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SESSIONS 65536
#define VEC_LEN 1479
#define WIN_SIZE 15

// 세션 구조체 정의
typedef struct {
    float buffer[WIN_SIZE][VEC_LEN];
    int count;
} Session;

// 세션 테이블 (지연 할당)
static Session** session_table = NULL;

// 세션 테이블 초기화
int init_session_storage() {
    session_table = (Session**)calloc(MAX_SESSIONS, sizeof(Session*));
    return (session_table != NULL) ? 0 : -1;
}

// 세션 할당 또는 반환
Session* get_or_create_session(uint32_t session_id) {
    if (session_id >= MAX_SESSIONS) return NULL;

    if (session_table[session_id] == NULL) {
        session_table[session_id] = (Session*)malloc(sizeof(Session));
        if (!session_table[session_id]) return NULL;
        memset(session_table[session_id], 0, sizeof(Session));
    }
    return session_table[session_id];
}

// TCP 패킷 파싱 후 벡터 추출
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

// 파싱 및 세션 스택에 저장
int parse_and_stack(const uint8_t* raw, size_t len, float* out_stack, uint32_t session_id) {
    if (session_id >= MAX_SESSIONS) return -1;

    Session* sess = get_or_create_session(session_id);
    if (!sess) return -1;

    float temp[VEC_LEN];
    if (parse_tcp_packet(raw, len, temp) != 0) return -1;

    int idx = sess->count;
    if (idx >= WIN_SIZE) {
        for (int i = 0; i < WIN_SIZE - 1; ++i) {
            memcpy(sess->buffer[i], sess->buffer[i + 1], sizeof(float) * VEC_LEN);
        }
        idx = WIN_SIZE - 1;
    }

    memcpy(sess->buffer[idx], temp, sizeof(float) * VEC_LEN);
    sess->count = idx + 1;

    if (sess->count < WIN_SIZE) return 0;

    for (int i = 0; i < WIN_SIZE; ++i) {
        memcpy(out_stack + i * VEC_LEN, sess->buffer[i], sizeof(float) * VEC_LEN);
    }

    return 1;
}
