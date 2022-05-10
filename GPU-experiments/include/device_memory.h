/*
 * @Author: Yangjie Zhou. https://zenodo.org/record/5535284
 * @Date: 2022-03-11 20:26:34
 * @LastEditTime: 2022-03-19 10:31:43
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /GPU-experiments/include/common.h
 */

#ifndef DEVICE_MEMORY_H
#define DEVICE_MEMORY_H

#include "common.h"

namespace device_memory {


template <typename T>
T* allocate(size_t count = 1) {

  T* ptr = 0;
  size_t bytes = 0;

  bytes = count * sizeof(T);

  cudaError_t cuda_error = cudaMalloc((void**)&ptr, bytes);

  if (cuda_error != cudaSuccess) {
    throw cuda_exception("Failed to allocate memory", cuda_error);
  }

  return ptr;
}

template <typename T>
void free(T* ptr) {
  if (ptr) {
    cudaError_t cuda_error = (cudaFree(ptr));
    if (cuda_error != cudaSuccess) {
      throw cuda_exception("Failed to free device memory", cuda_error);
    }
  }
}



template <typename T>
void copy(T* dst, T const* src, size_t count, cudaMemcpyKind kind) {
  // copy data from src to dst, data length is count*sizeof(T), kind
  size_t bytes = count*sizeof(T);
  if (bytes == 0 && count > 0)
    bytes = 1;
    
  cudaError_t cuda_error = (cudaMemcpy(dst, src, bytes, kind));
  if (cuda_error != cudaSuccess) {
    throw cuda_exception("cudaMemcpy() failed", cuda_error);
  }
}

template <typename T>
void copy_to_device(T* dst, T const* src, size_t count = 1) {
  copy(dst, src, count, cudaMemcpyHostToDevice);
}

template <typename T>
void copy_to_host(T* dst, T const* src, size_t count = 1) {
  copy(dst, src, count, cudaMemcpyDeviceToHost);
}

template <typename T>
void copy_device_to_device(T* dst, T const* src, size_t count = 1) {
  copy(dst, src, count, cudaMemcpyDeviceToDevice);
}

template <typename T>
void copy_host_to_host(T* dst, T const* src, size_t count = 1) {
  copy(dst, src, count, cudaMemcpyHostToHost);
}

template <typename OutputIterator, typename T>
void insert_to_host(OutputIterator begin, OutputIterator end, T const* device_begin) {
  size_t elements = end - begin;
  copy_to_host(&*begin, device_begin, elements);
}

template <typename T, typename InputIterator>
void insert_to_device(T* device_begin, InputIterator begin, InputIterator end) {
  size_t elements = end - begin;
  copy_to_device(device_begin, &*begin, elements);
}


}  



template <typename T>
class DeviceAllocation {
public:

  struct deleter {
    void operator()(T* ptr) {
      cudaError_t cuda_error = (cudaFree(ptr));
      if (cuda_error != cudaSuccess) {
        return;
      }
    }
  };

public:

  size_t capacity;

  std::unique_ptr<T, deleter> smart_ptr;

public:


  static size_t bytes(size_t elements) {

    return elements * sizeof(T);
  }

public:


  DeviceAllocation() : capacity(0) {}

  DeviceAllocation(size_t _capacity) : 
    smart_ptr(device_memory::allocate<T>(_capacity)), capacity(_capacity) {}

  DeviceAllocation(T *ptr, size_t _capacity) : smart_ptr(ptr), capacity(_capacity) {}

  DeviceAllocation(DeviceAllocation const &p): 
    smart_ptr(device_memory::allocate<T>(p.capacity)), capacity(p.capacity) {

    device_memory::copy_device_to_device(smart_ptr.get(), p.get(), capacity);
  }

  DeviceAllocation(DeviceAllocation &&p): capacity(0) {
    std::swap(smart_ptr, p.smart_ptr);
    std::swap(capacity, p.capacity);
  }

  ~DeviceAllocation() { reset(); }

  T* get() const { return smart_ptr.get(); }

  T* release() {
    capacity = 0;
    return smart_ptr.release();
  }

  void reset() {
    capacity = 0;
    smart_ptr.reset();
  }

  void reset(size_t _capacity) {
    reset(device_memory::allocate<T>(_capacity), _capacity);
  }

  void reset(T* _ptr, size_t _capacity) {
    smart_ptr.reset(_ptr);
    capacity = _capacity;
  }

  void reallocate(size_t new_capacity) {
    
    std::unique_ptr<T, deleter> new_allocation(device_memory::allocate<T>(new_capacity));

    device_memory::copy_device_to_device(
      new_allocation.get(), 
      smart_ptr.get(), 
      std::min(new_capacity, capacity));

    std::swap(smart_ptr, new_allocation);
    std::swap(new_capacity, capacity);
  }

  size_t size() const {
    return capacity;
  }

  size_t bytes() const {
    return bytes(capacity);
  }

  T* operator->() const { return smart_ptr.get(); }

  deleter& get_deleter() { return smart_ptr.get_deleter(); }

  const deleter& get_deleter() const { return smart_ptr.get_deleter(); }

  DeviceAllocation & operator=(DeviceAllocation const &p) {
    if (capacity != p.capacity) {
      smart_ptr.reset(device_memory::allocate<T>(p.capacity));
      capacity = p.capacity;
    }
    copy_device_to_device(smart_ptr.get(), p.get(), capacity);
    return *this;
  }

  DeviceAllocation & operator=(DeviceAllocation && p) {
    std::swap(smart_ptr, p.smart_ptr);
    std::swap(capacity, p.capacity);
    return *this;
  }

  void copy_from_device(T const *ptr) const {
    copy_from_device(ptr, capacity);
  }

  void copy_from_device(T const *ptr, size_t elements) const {
    device_memory::copy_device_to_device(get(), ptr, elements);
  }

  void copy_to_device(T *ptr) const {
    copy_to_device(ptr, capacity);
  }

  void copy_to_device(T *ptr, size_t elements) const {
    device_memory::copy_device_to_device(ptr, get(), elements);
  }

  void copy_from_host(T const *ptr) const {
    copy_from_host(ptr, capacity);
  }

  void copy_from_host(T const *ptr, size_t elements) const {
    device_memory::copy_to_device(get(), ptr, elements);
  }

  void copy_to_host(T *ptr) const {
    copy_to_host(ptr, capacity);
  }

  void copy_to_host(T *ptr, size_t elements) const {
    device_memory::copy_to_host(ptr, get(), elements); 
  }
};


namespace device_memory {

template <typename T>
using allocation = DeviceAllocation<T>;

}  





#endif 