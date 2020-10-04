#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H
#include <forward_list>
#include <new>
#include <utility>

// Simple memory pool consisting of a singly-linked list
// of memory blocks. Adds new blocks as needed
class MemoryPool {
    std::forward_list<char*> m_pools;
    std::forward_list<char*>::iterator m_curr;
    const size_t m_pool_capacity;
    size_t m_curr_used = 0;
 public:
    explicit MemoryPool(size_t size)
        : m_pool_capacity(size)
    {
        m_pools.push_front((char*)malloc(size));
        m_curr = m_pools.begin();
    }
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    MemoryPool(MemoryPool&&) = default;

    ~MemoryPool()
    {
        for(char *pool : m_pools) {
            free(pool);
        }
    }

    template<typename T, typename ...Params>
    T* make(Params... args)
    {
        constexpr size_t object_size = sizeof(T);
        size_t offset = m_curr_used % alignof(T);
        if(m_curr_used + offset + object_size > m_pool_capacity) {
            add_new_pool();
            offset = 0;
        }

        m_curr_used += offset;
        T* object = new (*m_curr + m_curr_used) T{args...};
        m_curr_used += object_size;
        return object;
    }
 private:
    void add_new_pool()
    {
        m_pools.push_front((char*)malloc(m_pool_capacity));
        m_curr = m_pools.begin();
        m_curr_used = 0;
    }
};
#endif