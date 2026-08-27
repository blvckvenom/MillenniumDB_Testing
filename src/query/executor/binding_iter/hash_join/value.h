#pragma once

#include <cstdint>

namespace HashJoin {
/*
Value is the payload stored in the in-memory hash-join hash tables: the join
key maps to a singly linked list of build-side rows that share that key.

Each row lives in a data chunk as `build_vars.size()` ObjectId payloads
followed by one pointer-sized slot holding the address of the next row in
the list (nullptr terminates it). `head` points to the first row, where
match enumeration starts; `tail` points to the last row so appending another
row with the same key is O(1).
*/
struct Value {
    uint64_t* head;
    uint64_t* tail;

    Value(uint64_t* head, uint64_t* tail) :
        head(head),
        tail(tail) {};
};

} // namespace HashJoin
