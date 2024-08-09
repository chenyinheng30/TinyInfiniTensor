#include "core/allocator.h"
#include <cstddef>
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        for (auto free = free_blocks.begin();
                free != free_blocks.end(); free++){
            if (free->second >= size) {
                auto free_size = free->second;
                free->second = free_size - size;
                used += size;
                return free->first - free_size;
            }
        }

        auto offset = peak;
        used += size;
        peak += size;
        return offset;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        if(addr + size == peak){
            peak -= size;
        } else {
            auto free = free_blocks.begin();
            if(free == free_blocks.end()){
                free_blocks.insert({addr+size, size});
            }
            for(;free != free_blocks.end();free++){
                if(addr + size + free->second == free->first){
                    free->second += size;
                    break;
                }
            }
            if(free != free_blocks.end()){
                if(auto before = free_blocks.find(addr); before != free_blocks.end()){
                    free->second += before->second;
                    free_blocks.erase(before);
                }
            }
        }
        used -= size;
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
