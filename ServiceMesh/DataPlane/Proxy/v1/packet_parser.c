#include <stdint.h>
#include <string.h>

#define VECTOR_LEN 1484  // 24(header) + 1460(payload)

// C version of packet parser
int parse_packet(const uint8_t *raw_data, size_t len, uint8_t *out_vec) {
    if (len < 54) return -1;

    // IP header starts at offset 14 (Ethernet)
    const uint8_t *ip_hdr = raw_data + 14;
    const uint8_t *tcp_hdr = ip_hdr + 20;

    if (len < 14 + 20 + 20) return -1;

    // IP fields
    uint8_t ttl = ip_hdr[8];
    uint8_t proto = ip_hdr[9];
    uint16_t flags_frag = (ip_hdr[6] << 8) | ip_hdr[7];
    uint8_t ip_flags = (flags_frag >> 13) & 0x7;
    uint16_t frag_offset = flags_frag & 0x1FFF;

    // TCP fields
    uint8_t data_offset = (tcp_hdr[12] >> 4) & 0xF;
    uint8_t flags = tcp_hdr[13];
    uint16_t window = (tcp_hdr[14] << 8) | tcp_hdr[15];
    uint16_t urgptr = (tcp_hdr[18] << 8) | tcp_hdr[19];

    const uint8_t *seq = tcp_hdr + 4;
    const uint8_t *ack = tcp_hdr + 8;

    size_t payload_start = 14 + 20 + data_offset * 4;
    if (payload_start >= len) return -1;

    const uint8_t *payload = raw_data + payload_start;
    size_t payload_len = len - payload_start;
    if (payload_len > 1460) payload_len = 1460;

    // Assemble vector
    out_vec[0] = ttl;
    out_vec[1] = proto;
    out_vec[2] = ip_flags;
    out_vec[3] = (frag_offset >> 8) & 0xFF;
    out_vec[4] = frag_offset & 0xFF;
    out_vec[5] = data_offset;
    out_vec[6] = flags;
    out_vec[7] = (window >> 8) & 0xFF;
    out_vec[8] = window & 0xFF;
    out_vec[9] = (urgptr >> 8) & 0xFF;
    out_vec[10] = urgptr & 0xFF;

    memcpy(out_vec + 11, seq, 4);
    memcpy(out_vec + 15, ack, 4);
    memset(out_vec + 19, 0, 1460);  // zero-fill
    memcpy(out_vec + 19, payload, payload_len);

    return 0; // success
}
