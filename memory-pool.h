/* Bluebird compiler - ahead-of-time compiler for the Bluebird language using LLVM.
    Copyright (C) 2020  Cole Blakley

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/
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
    const size_t m_block_size;
    size_t m_curr_used = 0;
 public:
    explicit MemoryPool(size_t size) : m_block_size(size)
    {
        m_pools.push_front((char*)malloc(m_block_size));
        m_curr = m_pools.begin();
    }
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    ~MemoryPool()
    {
        for(char *pool : m_pools) {
            free(pool);
        }
    }

    template<typename T, typename ...Params>
    T* make(Params... args)
    {
        size_t offset = m_curr_used % alignof(T);
        if(m_curr_used + offset + sizeof(T) > m_block_size) {
            add_new_pool();
            offset = 0;
        }

        m_curr_used += offset;
        T* object = new (*m_curr + m_curr_used) T{std::forward<Params>(args)...};
        m_curr_used += sizeof(T);
        return object;
    }
 private:
    void add_new_pool()
    {
        m_pools.push_front((char*)malloc(m_block_size));
        m_curr = m_pools.begin();
        m_curr_used = 0;
    }
};
#endif
