#ifndef _XGB_COMMON_H_
#define _XGB_COMMON_H_

#define OP_ALL_GATHER       1
#define OP_ALL_GATHER_V     2
#define OP_ALL_REDUCE       3
#define OP_BROADCAST        4
#define OP_DONE             99

#define ERR_OP_MISMATCH  -1
#define ERR_INVALID_RANK -2
#define ERR_NO_CLIENT_FOR_RANK -3
#define ERR_SEQ_MISMATCH -4
#define ERR_ABORTED -5


#endif