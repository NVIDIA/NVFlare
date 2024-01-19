#ifndef _XGB_CLIENT_H_
#define _XGB_CLIENT_H_

#define OP_ALL_GATHER       1
#define OP_ALL_GATHER_V     2
#define OP_ALL_REDUCE       3
#define OP_BROADCAST        4
#define OP_DONE             99

#define ERR_OP_MISMATCH  -1
#define ERR_INVALID_RANK -2
#define ERR_NO_CLIENT_FOR_RANK -3
#define ERR_ABORTED -4

typedef struct {
    int rank;
    int op;
    int seq;
    int data_type;
    int reduce_op;
    int root;
    unsigned char* send_buf;
    size_t send_buf_size;
    unsigned char* rcv_buf;
    size_t rcv_buf_size;
    int waiting_op;
    int received;
    int aborted;
} XGBClient;

extern void xgbc_initialize(int max_num_clients);
extern int xgbc_new_client(int rank);

extern int xgbc_get_pending_op(
    int rank,
    int* seq,
    unsigned char** send_buf,
    size_t* send_size,
    int* data_type,     // for Allreduce
    int* reduce_op,     // for Allreduce
    int* root           // for Broadcast
);

extern int xgbc_reply(int op, int rank, unsigned char* rcv_buf, size_t rcv_size);

extern int xgbc_send_all_gather(
    int rank, int seq, unsigned char* send_buf, size_t send_size, unsigned char** rcv_buf, size_t* rcv_size
);

extern int xgbc_send_all_gather_v(
    int rank, int seq, unsigned char* send_buf, size_t send_size, unsigned char** rcv_buf, size_t* rcv_size
);

extern int xgbc_send_all_reduce(
    int rank, int seq, int data_type, int reduce_op,
    unsigned char* send_buf, size_t send_size, unsigned char** rcv_buf, size_t* rcv_size
);

extern int xgbc_send_broadcast(
    int rank, int seq, int root, unsigned char* send_buf, size_t send_size, unsigned char** rcv_buf, size_t* rcv_size
);

extern int xgbc_start(int rank, int num_rounds);

extern void xgbc_abort(int rank);

#endif