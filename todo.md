* Overview how search works
* In depth in each step
    * Merge Minimize
        * Why we want to merge
        * How merge works
        * How it used to be
        * Why we want to minize
        * DP algo
        * Performance bottleneck of hashing, how it was solved
        * Linked list
        * Bump allocator
        * It used to be a bottleneck for long queries, but for a long time the poor implementation was enough
    * Index format
        * LMDB + rkyv
        * Importance of aligned data (this will be talked again later)
        * Use mmap + offset to make data aligned (couldn't make lmdb be aligned)
        * madvise
    * Smart execution
        * Why we want to do it
        * How it works
    * Intersection
        * Overview (no binary search, no gallop)
            * Why do two phases of intersection
            * Naive
            * Simd
                * vp2intersect
                    * how it works
                    * deprecated
                    * performance on intel and amd
                    * emulated
                * compress/compressstore
                    * how it works
                * how it works, diagram maybe ?
                * compress + store vs compressstore
                    * performance on intel and amd
                * aligned data
                    * importance
                * show asm
                * uica uops the loop
                * elements/cycle
                * bad optimization from the compiler depeding where last is loaded (show asm in both cases)
                * loop alignment
                    * importance
                    * fix (align-all-functions + nops)
                    * show tool
            * Merge
        * Why binary search before intersection phases
        * When it's worth gallop intersect
    * Get token ids
        * Naive version
        * Simd version
        * In the future maybe let the user specificy if he wants the doc id or doc id + pos with a custom type