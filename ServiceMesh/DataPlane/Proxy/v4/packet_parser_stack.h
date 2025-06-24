#ifndef PACKET_PARSER_STACK_H
#define PACKET_PARSER_STACK_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int parse_and_stack(const uint8_t* raw, size_t len, float* out_stack, uint32_t session_id);
void init_buffers();
void free_buffers();

#ifdef __cplusplus
}
#endif

#endif
