/+
    Module contains memory handing routines used throughout the library.
+/
module mir.cv.core.memory;

import core.stdc.stdlib : malloc, free;

version (Posix)
@nogc nothrow
private extern(C) int posix_memalign(void**, size_t, size_t);

// Copy pasta of aligned memory allocation used in std.experimental.allocation.mallocator.AlignedMallocator ////////////

version (Windows)
{
    // DMD Win 32 bit, DigitalMars C standard library misses the _aligned_xxx
    // functions family (snn.lib)
    version(CRuntime_DigitalMars)
    {
        // Helper to cast the infos written before the aligned pointer
        // this header keeps track of the size (required to realloc) and of
        // the base ptr (required to free).
        private struct AlignInfo
        {
            void* basePtr;
            size_t size;

            @nogc nothrow
            static AlignInfo* opCall(void* ptr)
            {
                return cast(AlignInfo*) (ptr - AlignInfo.sizeof);
            }
        }

        @nogc nothrow
        private void* _aligned_malloc(size_t size, size_t alignment)
        {
            import std.c.stdlib: malloc;
            size_t offset = alignment + size_t.sizeof * 2 - 1;

            // unaligned chunk
            void* basePtr = malloc(size + offset);
            if (!basePtr) return null;

            // get aligned location within the chunk
            void* alignedPtr = cast(void**)((cast(size_t)(basePtr) + offset)
                & ~(alignment - 1));

            // write the header before the aligned pointer
            AlignInfo* head = AlignInfo(alignedPtr);
            head.basePtr = basePtr;
            head.size = size;

            return alignedPtr;
        }

        @nogc nothrow
        private void* _aligned_realloc(void* ptr, size_t size, size_t alignment)
        {
            import std.c.stdlib: free;
            import std.c.string: memcpy;

            if(!ptr) return _aligned_malloc(size, alignment);

            // gets the header from the exising pointer
            AlignInfo* head = AlignInfo(ptr);

            // gets a new aligned pointer
            void* alignedPtr = _aligned_malloc(size, alignment);
            if (!alignedPtr)
            {
                //to https://msdn.microsoft.com/en-us/library/ms235462.aspx
                //see Return value: in this case the original block is unchanged
                return null;
            }

            // copy exising data
            memcpy(alignedPtr, ptr, head.size);
            free(head.basePtr);

            return alignedPtr;
        }

        @nogc nothrow
        private void _aligned_free(void *ptr)
        {
            import std.c.stdlib: free;
            if (!ptr) return;
            AlignInfo* head = AlignInfo(ptr);
            free(head.basePtr);
        }

    }
    // DMD Win 64 bit, uses microsoft standard C library which implements them
    else
    {
        @nogc nothrow private extern(C) void* _aligned_malloc(size_t, size_t);
        @nogc nothrow private extern(C) void _aligned_free(void *memblock);
        @nogc nothrow private extern(C) void* _aligned_realloc(void *, size_t, size_t);
    }
}

version(Posix)
@trusted @nogc nothrow
void[] alignedAllocate(size_t bytes, uint a)
{
    import core.stdc.errno : ENOMEM, EINVAL;
    void* result;
    auto code = posix_memalign(&result, a, bytes);
    if (code == ENOMEM)
        return null;

    else if (code == EINVAL)
        assert (0, "AlignedMallocator.alignment is not a power of two multiple of (void*).sizeof, according to posix_memalign!");

    else if (code != 0)
        assert (0, "posix_memalign returned an unknown code!");

    else
        return result[0 .. bytes];
}
else version(Windows)
@trusted @nogc nothrow
void[] alignedAllocate(size_t bytes, uint a)
{
    auto result = _aligned_malloc(bytes, a);
    return result ? result[0 .. bytes] : null;
}
else static assert(0);

/**
Calls $(D free(b.ptr)) on Posix and
$(WEB msdn.microsoft.com/en-US/library/17b5h8td(v=vs.80).aspx,
$(D __aligned_free(b.ptr))) on Windows.
*/
version (Posix)
@system @nogc nothrow
bool deallocate(void[] b)
{
    import core.stdc.stdlib : free;
    free(b.ptr);
    return true;
}
else version (Windows)
@system @nogc nothrow
bool deallocate(void[] b)
{
    _aligned_free(b.ptr);
    return true;
}
else static assert(0);

