/++
    Image filtering routines.
+/

module mir.cv.imgproc.imfilter;

import core.stdc.stdlib;

import mir.utility : max;

import mir.ndslice.slice : Slice, Contiguous, sliced;
import mir.ndslice.topology : zip, unzip, blocks, universal, map, windows;
import mir.ndslice.dynamic : strided;
import mir.ndslice.algorithm : each;

import mir.cv.core.memory;
import mir.cv.core.simd;

import cpuid.x86_any;

import core.simd;
import ldc.simd;


extern (C) @system nothrow @nogc:


// Border handing methods used in image filteirng.
enum BORDER_REPLICATE = 0;
enum BORDER_SAME = 1;
enum BORDER_SYMMETRIC = 2;

// Error codes that can occur in image filtering calls.
enum ERROR_INVALID_BORDER_METHOD = 1;

/++
    Separable image filtering.

    Params:
        inputbuf        = Input image. Size of 'rows' * 'cols'.
        hmask           = Horizontal mask. Size of 'hsize'.
        vmask           = Vertical mask. Size of 'vsize'.
        outputbuf       = Output image. Size of 'rows' * 'cols'.
        rows            = Row size (height) of input/output images.
        cols            = Column size (width) of input/output images.
        hsize           = Size of horizontal kernel.
        vsize           = Size of vertical kernel.
        border_handling = Border handling scheme used to process borders of the image.

    Returns:
        0 if successful, ERROR_INVALID_BORDER_METHOD if invalid value is passed for border_handling parameter.
+/
pragma(inline, false)
int separable_imfilter
(
    float* inputbuf,
    float* hmask,
    float* vmask,
    float* outputbuf,
    size_t rows,
    size_t cols,
    size_t hsize,
    size_t vsize,
    int border_handling = BORDER_REPLICATE
)
{
    return separable_imfilter_impl
        (inputbuf, hmask, vmask, outputbuf, rows, cols, hsize, vsize, border_handling);
}

package(mir.cv):

template Horizontal_kernel_func(alias InstructionSet)
{
    alias Horizontal_kernel_func = void function(InstructionSet.Scalar*, // input pointer.
        InstructionSet.Vector*, // kernel mask pointer.
        InstructionSet.Scalar*, // output pointer.
        size_t); // size of the kernel mask.
}

template Vertical_kernel_func(alias InstructionSet)
{
    alias Vertical_kernel_func = void function(InstructionSet.Scalar*, // input pointer.
        InstructionSet.Vector*, // kernel mask pointer.
        InstructionSet.Scalar*, // output pointer.
        size_t, // size of the kernel mask.
        size_t); // rowstride to read another row element.
}

int separable_imfilter_impl(T)
(
    T* inputbuf,
    T* hmask,
    T* vmask,
    T* outputbuf,
    size_t rows,
    size_t cols,
    size_t hsize,
    size_t vsize,
    int border_handling = BORDER_REPLICATE
)
{
    auto tempbuf = cast(T[])alignedAllocate(rows * cols * T.sizeof, 16); // temporary buffer, used to store horizontal filtering

    scope (exit)
    {
        deallocate(tempbuf);
    }

    auto input = inputbuf.sliced(rows, cols);
    auto temp = tempbuf.sliced(rows, cols);
    auto output = outputbuf.sliced(rows, cols);
    auto hm = hmask.sliced(hsize);
    auto vm = vmask.sliced(vsize);

    if (avx) // add version (compile time flag if avx should be supported)
    {
        inner_filtering_impl!(AVX!T)(input, temp, hm, vm, output,
            &apply_horizontal_kernel_simd!(AVX!T), &apply_vertical_kernel_simd!(AVX!T));
    }
    else version (sse)
    {
        inner_filtering_impl!(SSE!T)(input, temp, hm, vm, output,
            &apply_horizontal_kernel_simd!(SSE!T), &apply_vertical_kernel_simd!(SSE!T));
    }
    else
    {
        inner_filtering_impl!(Non_SIMD!T)(input, temp, hm, vm, output,
            get_horizontal_kernel_for_mask!T(hsize), get_vertical_kernel_for_mask!T(vsize));
    }

    if (border_handling != BORDER_REPLICATE)
    {
        return ERROR_INVALID_BORDER_METHOD;
    }

    borders_replicate_impl(output, hsize, vsize);

    return 0;
}

pragma(inline, true)
void inner_filtering_impl(alias InstructionSet)
(
    Slice!(Contiguous, [2], InstructionSet.Scalar*) input,
    Slice!(Contiguous, [2], InstructionSet.Scalar*) temp,
    Slice!(Contiguous, [1], InstructionSet.Scalar*) hmask,
    Slice!(Contiguous, [1], InstructionSet.Scalar*) vmask,
    Slice!(Contiguous, [2], InstructionSet.Scalar*) output,
    Horizontal_kernel_func!InstructionSet hkernel,
    Vertical_kernel_func!InstructionSet vkernel,
)
in
{
    assert(input.shape == temp.shape, "Incompatible input and temporary buffer slice");
    assert(temp.shape == output.shape, "Incompatible input and output buffer slice");
}
body
{
    alias T = InstructionSet.Scalar;
    alias V = InstructionSet.Vector;
    immutable velems = InstructionSet.elementCount;
    immutable tbytes = T.sizeof;
    immutable vbytes = V.sizeof;
    immutable hsize = hmask.length;
    immutable vsize = vmask.length;
    immutable rows = input.length!0;
    immutable cols = input.length!1;

    static if (Is_SIMD!InstructionSet)
    {
        // It this is SIMD instructions set, allocate vectors for kernels

        auto hk = cast(V[])alignedAllocate(hsize * vbytes, 16); // horizontal simd kernel
        auto vk = cast(V[])alignedAllocate(vsize * vbytes, 16); // vertical simd kernel

        scope (exit)
        {
            deallocate(hk);
            deallocate(vk);
        }

        foreach (i; 0 .. hsize)
        {
            hk[i].array[] = hmask[i];
        }

        foreach (i; 0 .. vsize)
        {
            vk[i].array[] = vmask[i];
        }
    }
    else
    {
        // ... otherwise just use input buffers.
        auto hk = hmask._iterator[0 .. hsize];
        auto vk = vmask._iterator[0 .. vsize];
    }

    // Get pointers to where vectors are loaded (has to be public)
    static auto toPtr(ref T e)
    {
        return &e;
    }

    auto _in = input.universal.map!toPtr;
    auto _tmp = temp.universal.map!toPtr;
    auto _out = output.universal.map!toPtr;

    // Stride buffers to match vector indexing.
    auto a = _in[0 .. $ - vsize, 0 .. $ - hsize].strided!1(velems);
    auto t = _tmp[0 .. $ - vsize, 0 .. $ - hsize].strided!1(velems);
    auto b = _out[0 .. $ - vsize, 0 .. $ - hsize].strided!1(velems);

    // L1 cache block tiling (under 8192 bytes)
    immutable rtiling = 32;
    immutable ctiling = max(size_t(1), 256 / (tbytes * hsize));

    // Process blocks
    zip!true(a, t, b)
    .blocks(rtiling, ctiling)
  //.parallel
    .each!((b) {
        b.each!((w) { hkernel(w.a, hk.ptr, w.b, hsize); }); // apply horizontal kernel
            b.each!((w) { vkernel(w.b, vk.ptr, w.c, vsize, cols); }); // apply vertical kernel
    });

    // Fill-in block horizontal borders
    zip!true(t, b)[rtiling - vsize .. $ - vsize, 0 .. $]
    .windows(vsize, b.length!1)
    .strided!0(rtiling)
    .each!((b) {
        b.each!((w) { vkernel(w.a, vk.ptr, w.b, vsize, cols); });
    });

    // perform scalar processing for the remaining pixels (from vector block selection)
    immutable rrb = a.length!0 - (a.length!0 % rtiling) - vsize;
    immutable crb = (a.length!1 - (a.length!1 % ctiling)) * velems - hsize;
    immutable rowstride = input.shape[1];

    // get scalar kernels
    auto hskernel = get_horizontal_kernel_for_mask!T(hsize);
    auto vskernel = get_vertical_kernel_for_mask!T(vsize);

    // bottom rows
    if (rrb < rows - vsize)
    {
        zip!true(_in, _tmp)[rrb .. $, 0 .. $ - hsize].each!((w) { hskernel(w.a, hmask._iterator, w.b, hsize); });
        zip!true(_tmp, _out)[rrb .. $ - vsize, 0 .. $ - hsize / 2].each!((w) { vskernel(w.a, vmask._iterator, w.b, vsize, rowstride); });
    }

    // right columns
    if (crb < cols - vsize)
    {
        zip!true(_in, _tmp)[0 .. $ - vsize, crb .. $ - hsize].each!((w) { hskernel(w.a, hmask._iterator, w.b, hsize); });
        zip!true(_tmp, _out)[0 .. $ - vsize, crb .. $ - hsize / 2].each!((w) { vskernel(w.a, vmask._iterator, w.b, vsize, rowstride); });
    }
}

pragma(inline, true)
void borders_replicate_impl(T)
(
    Slice!(Contiguous, [2], T*) output,
    size_t hsize,
    size_t vsize
)
{
    auto top = output[0 .. vsize / 2 + 1, hsize / 2 .. $ - hsize / 2 + 1];
    auto bottom = output[$ - vsize / 2 - 2 .. $, hsize / 2 .. $ - hsize / 2 + 1];

    foreach (c; 0 .. top.length!1)
        foreach (r; 0 .. top.length!0 - 1)
        {
            top[r, c] = top[hsize / 2 + 1, c];
            bottom[r + 1, c] = bottom[0, c];
        }

    auto left = output[0 .. $, 0 .. hsize / 2 + 1];
    auto right = output[0 .. $, $ - hsize / 2 - 2 .. $];

    foreach (r; 0 .. left.length!0)
        foreach (c; 0 .. left.length!1 - 1)
        {
            left[r, c] = left[r, hsize / 2 + 1];
            right[r, c + 1] = right[r, 0];
        }
}
// Horizontal kernels //////////////////////////////////////////////////////////

pragma(inline, true)
void apply_horizontal_kernel_3(T)(T* i, T* k, T* o, size_t)
{
    o[1] = cast(T)(i[0] * k[0] +
                   i[1] * k[1] +
                   i[2] * k[2]);
}

pragma(inline, true)
void apply_horizontal_kernel_5(T)(T* i, T* k, T* o, size_t)
{
    o[2] =
        cast(T)(i[0] * k[0] +
                i[1] * k[1] +
                i[2] * k[2] +
                i[3] * k[3] +
                i[4] * k[4]);
}

pragma(inline, true)
void apply_horizontal_kernel_7(T)(T* i, T* k, T* o, size_t)
{
    o[3] =
        cast(T)(i[0] * k[0] +
                i[1] * k[1] +
                i[2] * k[2] +
                i[3] * k[3] +
                i[4] * k[4] +
                i[5] * k[5] +
                i[6] * k[6]);
}

pragma(inline, true)
void apply_horizontal_kernel(T)(T* i, T* k, T* o, size_t msize)
{
    T r = T(0);
    foreach (_; 0 .. msize)
    {
        r += (*(i++)) * (*(k++));
    }
    o[msize / 2] = r;
}

// Vertical kernels ////////////////////////////////////////////////////////////

pragma(inline, true)
void apply_vertical_kernel_3(T)(T* i, T* k, T* o, size_t, size_t rowstride)
{
    o[rowstride] =
        cast(T)(i[0] * k[0] +
                i[rowstride] * k[1] +
                i[rowstride * 2] * k[2]);
}

pragma(inline, true)
void apply_vertical_kernel_5(T)(T* i, T* k, T* o, size_t, size_t rowstride)
{
    o[rowstride * 2] =
        cast(T)(i[0] * k[0] +
                i[rowstride] * k[1] +
                i[rowstride * 2] * k[2] +
                i[rowstride * 3] * k[3] +
                i[rowstride * 4] * k[4]);
}

pragma(inline, true)
void apply_vertical_kernel_7(T)(T* i, T* k, T* o, size_t, size_t rowstride)
{
    o[rowstride * 3] =
        cast(T)(i[0] * k[0] +
                i[rowstride] * k[1] +
                i[rowstride * 2] * k[2] +
                i[rowstride * 3] * k[3] +
                i[rowstride * 4] * k[4] +
                i[rowstride * 5] * k[5] +
                i[rowstride * 6] * k[6]);
}

pragma(inline, true)
void apply_vertical_kernel(T)(T* i, T* k, T* o, size_t msize,
    size_t rowstride)
{
    T r = T(0);
    foreach (_; 0 .. msize)
    {
        r += (*i) * (*(k++));
        i += rowstride;
    }
    o[rowstride * (msize / 2)] = r;
}

// SIMD kernels ////////////////////////////////////////////////////////////////

pragma(inline, true)
void apply_horizontal_kernel_simd(alias InstructionSet)
(
    InstructionSet.Scalar* i,
    InstructionSet.Vector* k,
    InstructionSet.Scalar* o,
    size_t ksize
)
{
    alias V = InstructionSet.Vector;
    V e = InstructionSet.Scalar(0);
    foreach (_; 0 .. ksize)
    {
        e += loadUnaligned!V(i++) * (*(k++));
    }
    storeUnaligned!V(e, o + ksize / 2);
}

pragma(inline, true)
void apply_vertical_kernel_simd(alias InstructionSet)
(
    InstructionSet.Scalar* i,
    InstructionSet.Vector* k,
    InstructionSet.Scalar* o,
    size_t ksize,
    size_t rowstride
)
{
    alias V = InstructionSet.Vector;
    V e = InstructionSet.Scalar(0);
    foreach (_; 0 .. ksize)
    {
        e += loadUnaligned!V(i) * (*(k++));
        i += rowstride;
    }
    storeUnaligned!V(e, o + (ksize / 2) * rowstride);
}

// Scalar kernel getters ///////////////////////////////////////////////////////

pragma(inline, true)
auto get_horizontal_kernel_for_mask(T)(size_t mask_size)
{
    switch (mask_size)
    {
    case 3:
        return &apply_horizontal_kernel_3!T;
    case 5:
        return &apply_horizontal_kernel_5!T;
    case 7:
        return &apply_horizontal_kernel_7!T;
    default:
        break;
    }
    return &apply_horizontal_kernel!T;
}

pragma(inline, true)
auto get_vertical_kernel_for_mask(T)(size_t mask_size)
{
    switch (mask_size)
    {
    case 3:
        return &apply_vertical_kernel_3!T;
    case 5:
        return &apply_vertical_kernel_5!T;
    case 7:
        return &apply_vertical_kernel_7!T;
    default:
        break;
    }
    return &apply_vertical_kernel!T;
}
