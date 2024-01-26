#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include "../xgb_common.h"
#include "../xgb_client.h"

int XGBC_max_num_clients = 0;
XGBClient** XGBC_clients = NULL;
pthread_mutex_t XGBC_op_mutex;

void xgbc_initialize(int max_num_clients) {
    XGBC_max_num_clients = max_num_clients;
    XGBC_clients = (XGBClient**)malloc(max_num_clients * sizeof(XGBClient*));

    for (int i=0; i < max_num_clients; i++) {
        XGBC_clients[i] = NULL;
    }
    printf("CCC: XGBC initialized with %d max clients\n", max_num_clients);
}

XGBClient* _get_client(int rank, int* rc) {
    if (rank < 0 || rank >= XGBC_max_num_clients) {
        printf("CCC: invalid rank %d\n", rank);
        *rc = ERR_INVALID_RANK;
        return NULL;
    }
    *rc = 0;
    XGBClient* c = XGBC_clients[rank];
    if (!c) {
        *rc = ERR_NO_CLIENT_FOR_RANK;
    }
    return c;
}

int xgbc_new_client(int rank) {
    int rc;
    XGBClient* c;

    printf("CCC: creating new client for rank %d\n", rank);

    c = _get_client(rank, &rc);
    if (rc != 0 && rc != ERR_NO_CLIENT_FOR_RANK) {
        return rc;
    }
    if (!c) {
        // create a new client
        c = (XGBClient*)malloc(sizeof(XGBClient));
        c->rank = rank;
        c->op = 0;
        c->received = 0;
        c->waiting_op = 0;
        c->aborted = 0;
        XGBC_clients[rank] = c;
    }
    printf("CCC: created new client for rank %d\n", rank);
    return 0;
}

int xgbc_get_pending_op(
    int rank,
    int* seq,
    unsigned char** send_buf,
    size_t* send_size,
    int* data_type,     // for Allreduce
    int* reduce_op,     // for Allreduce
    int* root           // for Broadcast
) {
    int rc;
    XGBClient* c;
    c = _get_client(rank, &rc);
    if (!c) {
        printf("CCC: no client for rank %d\n", rank);
        return rc;
    }

    pthread_mutex_lock(&XGBC_op_mutex);
    *seq = c->seq;
    *send_buf = c->send_buf;
    *send_size = c->send_buf_size;
    *data_type = c->data_type;
    *reduce_op = c->reduce_op;
    *root = c->root;
    c->waiting_op = c->op;
    c->op = 0;
    pthread_mutex_unlock(&XGBC_op_mutex);
    return c->waiting_op;
}

XGBClient* _check_waiting_op(int rank, int received_op, int* rc) {
    XGBClient* c;
    c = _get_client(rank, rc);
    if (!c) {
        printf("CCC: no client for rank %d\n", rank);
        return c;
    }
    if (c->aborted) {
        printf("CCC: rank %d is aborted, but %d is received\n", rank, received_op);
        *rc = ERR_ABORTED;
        return c;
    }
    if (c->waiting_op != received_op) {
        printf("CCC: rank %d is waiting for op %d, but %d is received\n", rank, c->waiting_op, received_op);
        *rc = ERR_OP_MISMATCH;
        return c;
    }
    c->waiting_op = 0;
    *rc = 0;
    return c;
}

int xgbc_reply(int op, int rank, unsigned char* rcv_buf, size_t rcv_size) {
    printf("CCC: xgbc_reply: rank=%d, op=%d rcv_size=%zu\n", rank, op, rcv_size);
    int rc;
    XGBClient* c;
    c = _check_waiting_op(rank, op, &rc);
    if (rc != 0) {
        return rc;
    }

    // Note: the rcv_buf is allocated by Python side.
    // We must make a copy here; otherwise it may be garbage-collected after this function returns!
    c->rcv_buf = (unsigned char*)malloc(rcv_size);
    memcpy(c->rcv_buf, rcv_buf, rcv_size);
    c->rcv_buf_size = rcv_size;
    c->received = 1;
    return 0;
}

int _send_op(
    int op, int rank, int seq,
    unsigned char* send_buf, size_t send_size,
    int data_type, int reduce_op,   // for Allreduce
    int root,                       // for Broadcast
    unsigned char** rcv_buf, size_t* rcv_size
) {
    int rc;
    XGBClient* c;
    c = _get_client(rank, &rc);
    if (!c) {
        printf("CCC: no client for rank %d\n", rank);
        return rc;
    }

    printf("CCC: _send_op op=%d send_size=%zu\n", op, send_size);

    pthread_mutex_lock(&XGBC_op_mutex);
    c->send_buf = send_buf;
    c->send_buf_size = send_size;
    c->seq = seq;
    c->data_type = data_type;
    c->reduce_op = reduce_op;
    c->root = root;
    c->received = 0;
    c->op = op;
    pthread_mutex_unlock(&XGBC_op_mutex);

    // wait for reply
    while (!c->received) {
        if (c->aborted) {
            return ERR_ABORTED;
        }
        usleep(1000);
    }

    printf("CCC: Received reply for op %d rank %d, seq %d\n", op, rank, seq);
    *rcv_buf = c->rcv_buf;
    *rcv_size = c->rcv_buf_size;
    return 0;
}

int xgbc_send_all_gather(
    int rank, int seq, unsigned char* send_buf, size_t send_size, unsigned char** rcv_buf, size_t* rcv_size
) {
    return _send_op(OP_ALL_GATHER, rank, seq, send_buf, send_size, 0, 0, 0, rcv_buf, rcv_size);
}

int xgbc_send_all_gather_v(
    int rank, int seq, unsigned char* send_buf, size_t send_size, unsigned char** rcv_buf, size_t* rcv_size
) {
    return _send_op(OP_ALL_GATHER_V, rank, seq, send_buf, send_size, 0, 0, 0, rcv_buf, rcv_size);
}

int xgbc_send_all_reduce(
    int rank, int seq,
    int data_type, int reduce_op,
    unsigned char* send_buf, size_t send_size,
    unsigned char** rcv_buf, size_t* rcv_size
) {
    return _send_op(OP_ALL_REDUCE, rank, seq, send_buf, send_size, data_type, reduce_op, 0, rcv_buf, rcv_size);
}

int xgbc_send_broadcast(
    int rank, int seq, int root, unsigned char* send_buf, size_t send_size, unsigned char** rcv_buf, size_t* rcv_size
) {
    return _send_op(OP_BROADCAST, rank, seq, send_buf, send_size, 0, 0, root, rcv_buf, rcv_size);
}

void _check_result(
    int rank, int seq,
    char* name, int rc,
    unsigned char* send_buf, size_t send_size,
    unsigned char* rcv_buf, size_t rcv_size)
{
    if (rc != 0) {
        printf("CCC: %s failed: rc=%d\n", name, rc);
        return;
    }

    if (send_size != rcv_size) {
        printf("CCC: %s failed: send_size %zu != rcv_size %zu\n", name, send_size, rcv_size);
    } else {
        int cmp_rc = memcmp(send_buf, rcv_buf, rcv_size);
        if (cmp_rc != 0) {
            printf("CCC: %s failed: send_buf != rcv_buf: %d\n", name, cmp_rc);
        } else {
            printf("CCC: %s OK: Rank=%d, Seq=%d, RcvSize=%zu, rc=%d\n", name, rank, seq, rcv_size, rc);
        }
    }

    // the rcv_buf is allocated on C side. Must free it after used.
    if (rcv_buf) {
        free(rcv_buf);
    }
}

int xgbc_start(int rank, int num_rounds) {
    int rc;
    XGBClient* c;
    c = _get_client(rank, &rc);
    if (!c) {
        printf("CCC: no client for rank %d\n", rank);
        return rc;
    }

    size_t buf_size = 100;
    unsigned char* send_buf = (unsigned char*)malloc(buf_size);
    unsigned char *rcv_buf;
    size_t rcv_size;
    int seq = 0;
    for (int i = 0; i < num_rounds; i++) {
        printf("CCC: STARTED Rank=%d, Round=%d\n", rank, i);
        if (c->aborted) {
            printf("CCC: rank %d is asked to abort!\n", rank);
            return 0;
        }

        rc = xgbc_send_all_gather(rank, seq++, send_buf, buf_size, &rcv_buf, &rcv_size);
        _check_result(rank, seq, "Allgather", rc, send_buf, buf_size, rcv_buf, rcv_size);

        rc = xgbc_send_all_gather_v(rank, seq++, send_buf, buf_size, &rcv_buf, &rcv_size);
        _check_result(rank, seq, "AllgatherV", rc, send_buf, buf_size, rcv_buf, rcv_size);

        rc = xgbc_send_all_reduce(rank, seq++, 2, 3, send_buf, buf_size, &rcv_buf, &rcv_size);
        _check_result(rank, seq, "Allreduce", rc, send_buf, buf_size, rcv_buf, rcv_size);

        rc = xgbc_send_broadcast(rank, seq++, 1, send_buf, buf_size, &rcv_buf, &rcv_size);
        _check_result(rank, seq, "Broadcast", rc, send_buf, buf_size, rcv_buf, rcv_size);

        sleep(1);
    }

    // notify end of training
    c->op = OP_DONE;
    return 0;
}

void xgbc_abort(int rank) {
    int rc;
    XGBClient* c;
    c = _get_client(rank, &rc);
    if (!c) {
        printf("CCC: no client for rank %d\n", rank);
        return;
    }
    printf("CCC: rank %d received abort\n", rank);
    c->aborted = 1;
}
