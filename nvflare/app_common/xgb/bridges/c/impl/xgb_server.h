#ifndef _XGB_SERVER_H_
#define _XGB_SERVER_H_

extern void xgbs_initialize(int world_size);

extern int xgbs_all_gather(
    int rank, int seq,
    unsigned char* send_buf, size_t send_size,
    unsigned char** rcv_buf, size_t* rcv_size);

extern int xgbs_all_gather_v(
    int rank, int seq,
    unsigned char* send_buf, size_t send_size,
    unsigned char** rcv_buf, size_t* rcv_size);

extern int xgbs_all_reduce(
    int rank, int seq,
    int data_type, int reduce_op,
    unsigned char* send_buf, size_t send_size,
    unsigned char** rcv_buf, size_t* rcv_size);

extern int xgbs_broadcast(
    int rank, int seq, int root,
    unsigned char* send_buf, size_t send_size,
    unsigned char** rcv_buf, size_t* rcv_size);

extern void xgbs_free_buf(unsigned char* buf);

extern void xgbs_abort();

#endif