#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <pthread.h>
#include "../xgb_common.h"
#include "../xgb_server.h"

int XGBS_rcv_count = 0;
int XGBS_pending_count = 0;
int XGBS_pending_seq = -1;
int XGBS_pending_op = 0;
int XGBS_world_size = 0;
int XGBS_aborted = 0;
pthread_mutex_t XGBS_op_mutex;
pthread_mutex_t XGBS_count_mutex;

void xgbs_initialize(int world_size) {
    XGBS_world_size = world_size;
}

int _xgbs_process_request(
    int op, int rank, int seq,
    unsigned char* send_buf, size_t send_size,
    unsigned char** rcv_buf, size_t* rcv_size) {

    unsigned char* rcv_buf_ptr;
    int rc = 0;

    printf("start processing req: op=%d, rank=%d seq=%d\n", op, rank, seq);

    pthread_mutex_lock(&XGBS_op_mutex);
    if (XGBS_pending_seq < 0) {
        XGBS_pending_seq = seq;
        XGBS_pending_op = op;
    } else if (seq != XGBS_pending_seq) {
        printf("received seq %d from rank %d while working on seq %d\n", seq, rank, XGBS_pending_seq);
        rc = ERR_SEQ_MISMATCH;
    } else if (op != XGBS_pending_op) {
        printf("received op %d from rank %d while working on op %d\n", op, rank, XGBS_pending_op);
        rc = ERR_OP_MISMATCH;
    }
    pthread_mutex_unlock(&XGBS_op_mutex);

    if (rc != 0) {
        return rc;
    }

    // echo back
    rcv_buf_ptr = (unsigned char*)malloc(send_size);
    memcpy(rcv_buf_ptr, send_buf, send_size);
    *rcv_buf = rcv_buf_ptr;
    *rcv_size = send_size;

    // don't return until all ranks are received
    pthread_mutex_lock(&XGBS_count_mutex);
    XGBS_rcv_count ++;
    XGBS_pending_count ++;
    pthread_mutex_unlock(&XGBS_count_mutex);

    while (XGBS_rcv_count < XGBS_world_size) {
        if (XGBS_aborted) {
            printf("CCC: aborted while waiting for ranks for op=%d, rank=%d, seq=%d\n", op, rank, seq);
            return ERR_ABORTED;
        }
        usleep(1000);
    }

    pthread_mutex_lock(&XGBS_count_mutex);
    XGBS_pending_count --;
    pthread_mutex_unlock(&XGBS_count_mutex);
    if (XGBS_pending_count == 0) {
        // every rank has got its result - reset for next request
        XGBS_rcv_count = 0;
        XGBS_pending_seq = -1;
    }
    printf("finished req: op=%d rank=%d seq=%d\n", op, rank, seq);
    return 0;
}

int xgbs_all_gather(
    int rank, int seq,
    unsigned char* send_buf, size_t send_size,
    unsigned char** rcv_buf, size_t* rcv_size) {
    return _xgbs_process_request(OP_ALL_GATHER, rank, seq, send_buf, send_size, rcv_buf, rcv_size);
}

int xgbs_all_gather_v(
    int rank, int seq,
    unsigned char* send_buf, size_t send_size,
    unsigned char** rcv_buf, size_t* rcv_size) {
    return _xgbs_process_request(OP_ALL_GATHER_V, rank, seq, send_buf, send_size, rcv_buf, rcv_size);
}

int xgbs_all_reduce(
    int rank, int seq,
    int data_type, int reduce_op,
    unsigned char* send_buf, size_t send_size,
    unsigned char** rcv_buf, size_t* rcv_size) {
    return _xgbs_process_request(OP_ALL_REDUCE, rank, seq, send_buf, send_size, rcv_buf, rcv_size);
}

int xgbs_broadcast(
    int rank, int seq, int root,
    unsigned char* send_buf, size_t send_size,
    unsigned char** rcv_buf, size_t* rcv_size) {
    return _xgbs_process_request(OP_BROADCAST, rank, seq, send_buf, send_size, rcv_buf, rcv_size);
}

void xgbs_free_buf(unsigned char* buf) {
    free(buf);
}

void xgbs_abort() {
    printf("CCC: abort received\n");
    XGBS_aborted = 1;
}
