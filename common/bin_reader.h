#ifndef _BIN_READER_H
#define _BIN_READER_H

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdint.h>
#include <fstream>
#include <map>
#include <typeinfo>

#define MAGIC 0x1b5e0dba00000000

typedef struct {
    uint64_t magic;
    uint64_t offset;
    uint64_t size;
    uint64_t type;
} BinHeader;

typedef enum {
    BinTypeInvalid = 0x00,
    BinTypeUint32,
    BinTypeUint64,
    BinTypeInt32,
    BinTypeInt64,
    BinTypeFloat, 
    BinTypeDouble,
    BinTypeFirst = BinTypeInvalid,
    BinTypeLast = BinTypeDouble
} BinType;

typedef struct {
    union {
    std::fstream * handle;
    int            fd;
    } ref;
    uint64_t       offset;
    uint64_t       size;
    BinType        type;
} BinInfo;

#define CLEAN_AND_RETURN(inst, ret) do {delete inst; return ret;} while(0)

//Since this is a header file, it is not a good practice to have below:
//There will be a map associated to each of the source file.
static std::map<void *, BinInfo> npReadMap;

int binOpenForRead(const char * filename, BinInfo * bi) {


    std::fstream * handle = new std::fstream();
    handle->open(filename, std::ios::in | std::ios::binary | std::ios::ate);

    if (!handle->is_open())
        CLEAN_AND_RETURN(handle, -1);

    uint64_t size = handle->tellg();
    if (size < sizeof(BinHeader))
        CLEAN_AND_RETURN(handle, -1);

    handle->seekg(std::ios::beg);

    BinHeader header;
    handle->read((char *) &header, sizeof(BinHeader));

    if (header.magic != MAGIC)
        CLEAN_AND_RETURN(handle, -1);

    if ((header.type < BinTypeFirst) || (header.type > BinTypeLast))
        CLEAN_AND_RETURN(handle, -1);

    bi->ref.handle = handle;
    bi->offset     = header.offset;
    bi->size       = size;
    bi->type       = static_cast<BinType>(header.type);

    return 0;
}

int binOpenForReadNP(const char * filename, BinInfo * bi) {

    int fd = open(filename, O_RDONLY);
    if (fd == -1)
        return -1;

    off_t size = lseek(fd, 0, SEEK_END);
    uint64_t size64 = static_cast<uint64_t>(size);
    if (size64 < sizeof(BinHeader))
        return -1;

    if (lseek(fd, 0, SEEK_SET) == -1)
        return -1;

    BinHeader header;
    read(fd, &header, sizeof(BinHeader));

    
    if (header.magic != MAGIC)
        return -1;

    if ((header.type < BinTypeFirst) || (header.type > BinTypeLast))
        return -1;

    bi->ref.fd     = fd;
    bi->offset     = header.offset;
    bi->size       = size64;
    bi->type       = static_cast<BinType>(header.type);

    return 0;
}

int binOpenForWrite(const char * filename, BinInfo * bi) {

    std::fstream * handle = new std::fstream();
    handle->open(filename, std::ios::out | std::ios::binary);

    if (!handle->is_open())
        CLEAN_AND_RETURN(handle, -1);

    bi->ref.handle = handle;
    bi->offset     = 0;
    bi->size       = 0;
    bi->type       = BinTypeInvalid;

    return 0;
}

int binClose(BinInfo & bi) {

    if (bi.ref.handle == NULL)
        return -1;

    if (!bi.ref.handle->is_open())
        return 0;

    bi.ref.handle->close();

    return 0;
}

int CheckBinTypeCompatibility(const std::type_info & t, BinType * type)
{
#define CHECK_AND_RETURN(typ, binType, ret) \
    if (t==typeid(typ)) {*type=binType; return ret;} while(0)

    if ((sizeof(unsigned int) != sizeof(uint32_t)) ||
        (sizeof(unsigned long) != sizeof(uint64_t)) ||
        (sizeof(int) != sizeof(int32_t)) ||
        (sizeof(long) != sizeof(int64_t)))
        return -1;

    CHECK_AND_RETURN(uint32_t, BinTypeUint32, 0);
    CHECK_AND_RETURN(unsigned int, BinTypeUint32, 0);
    CHECK_AND_RETURN(uint64_t, BinTypeUint64, 0);
    CHECK_AND_RETURN(unsigned long, BinTypeUint64, 0);
    CHECK_AND_RETURN(int32_t, BinTypeInt32, 0);
    CHECK_AND_RETURN(int, BinTypeInt32, 0);
    CHECK_AND_RETURN(int64_t, BinTypeInt64, 0);
    CHECK_AND_RETURN(long, BinTypeInt64, 0);
    CHECK_AND_RETURN(float, BinTypeFloat, 0);
    CHECK_AND_RETURN(double, BinTypeDouble, 0);

    return -1;
        
#undef CHECK_AND_RETURN    
}

template <typename T> int binReadAsArray(
    const char * filename, BinInfo * bi, T ** arr, size_t * count) {

    if (arr == NULL)
        return -1;

    BinInfo ibi;

    int ret;
    ret = binOpenForRead(filename, &ibi);
    if (ret != 0)
        return -1;

    BinType type;
    if (CheckBinTypeCompatibility(typeid(T), &type))
        return -1;

    if (ibi.type != type)
        return -1;

    uint64_t cnt = (ibi.size - ibi.offset) / sizeof(T);
    T * iarr = new T [cnt];
    ibi.ref.handle->seekg(ibi.offset, std::ios::beg);
    ibi.ref.handle->read((char *) iarr, ibi.size);

    *arr = iarr;
    *count = cnt;
    if (bi != NULL)
        *bi = ibi;

    ret = binClose(ibi);
    if (ret != 0)
        return -1;

    return 0;
}

template <typename T> int binReadAsArrayNP(
    const char * filename, BinInfo * bi, T ** arr, size_t * count) {

    if (arr == NULL)
        return -1;

    BinInfo ibi;
    int ret;
    ret = binOpenForReadNP(filename, &ibi);
    if (ret != 0)
        return -1;
    
    BinType type;
    if (CheckBinTypeCompatibility(typeid(T), &type))
        return -1;

    if (ibi.type != type)
        return -1;

    void * ptr = mmap(NULL, ibi.size, PROT_READ, MAP_PRIVATE, ibi.ref.fd, 0);
    if (ptr == MAP_FAILED)
        return -1;

    T * data = reinterpret_cast<T *>(
        reinterpret_cast<char *>(ptr) + ibi.offset);
    *arr = data;
    *count = (ibi.size - ibi.offset) / sizeof(T);
    if (bi != NULL)
        *bi = ibi;

    npReadMap.insert(std::make_pair(reinterpret_cast<void *>(data), ibi));

    return 0;
}

template <typename T> int binDiscardArray(T * arr) {

    delete arr;

    return 0;
}

template <typename T> int binDiscardArrayNP(T * array) {

    std::map<void *, BinInfo>::iterator it = 
        npReadMap.find(reinterpret_cast<void *>(array));

    if (it == npReadMap.end())
        return -1;

    BinInfo bi = it->second;
    char * array_c = reinterpret_cast<char *>(array) - bi.offset;
    if (munmap(array_c, bi.size) == -1)
        return -1;

    return 0;
}

template <typename T> int binWriteArray(
    const char * filename, BinInfo * bi, T * arr, size_t size) {

    if (arr == NULL)
        return -1;

    BinInfo ibi;

    int ret;
    ret = binOpenForWrite(filename, &ibi);
    if (ret != 0)
        return -1;

    BinType type;
    if (CheckBinTypeCompatibility(typeid(T), &type))
        return -1;

    ibi.offset = sizeof(BinHeader);
    ibi.size   = size * sizeof(T);
    ibi.type   = type;

    BinHeader header;
    header.magic  = MAGIC;
    header.offset = ibi.offset;
    header.type   = ibi.type;

    ibi.ref.handle->write((char *) &header, sizeof(BinHeader));
    ibi.ref.handle->write((char *) arr, ibi.size);

    if (bi != NULL)
        *bi = ibi;

    ret = binClose(ibi);
    if (ret != 0)
        return -1;

    return 0;
}

#endif //BIN_READER_H

