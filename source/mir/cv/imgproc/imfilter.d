/++
    Image filtering routines.
+/

module mir.cv.imgproc.imfilter;

import core.stdc.stdlib;

import mir.utility : max;
import mir.internal.utility : fastmath;

import mir.ndslice.slice : Slice, Contiguous, sliced;
import mir.ndslice.topology : zip, unzip, blocks, universal, map, windows;
import mir.ndslice.dynamic : strided;
import mir.ndslice.algorithm : each;

import mir.cv.core.memory;
import mir.cv.core.simd;

import cpuid.x86_any;

import core.simd;
import ldc.simd;


extern (C) @system nothrow @nogc @fastmath:


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
    size_t ksize,
    int border_handling = BORDER_REPLICATE
)
{
    return separable_imfilter_impl
        (inputbuf, hmask, vmask, outputbuf, rows, cols, ksize, border_handling);
}

package(mir.cv):

template Horizontal_kernel_func(alias InstructionSet)
{
    alias Horizontal_kernel_func = void function(InstructionSet.Scalar*, // input pointer.
                                                 InstructionSet.Vector*, // kernel mask pointer.
                                                 InstructionSet.Scalar*, // output pointer.
                                                 size_t, // size of the kernel mask.
                                                 void*); // additional data
}

template Vertical_kernel_func(alias InstructionSet)
{
    alias Vertical_kernel_func = void function(InstructionSet.Scalar*, // input pointer.
                                               InstructionSet.Vector*, // kernel mask pointer.
                                               InstructionSet.Scalar*, // output pointer.
                                               size_t, // size of the kernel mask.
                                               size_t, // rowstride to read another row element.
                                               void*); // additional data.
}

int separable_imfilter_impl(T)
(
    T* inputbuf,
    T* hmask,
    T* vmask,
    T* outputbuf,
    size_t rows,
    size_t cols,
    size_t ksize,
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
    auto hm = hmask.sliced(ksize);
    auto vm = vmask.sliced(ksize);

    if (avx) // add version (compile time flag if avx should be supported)
    {
        inner_filtering_impl!(AVX!T, apply_horizontal_kernel_simd!(AVX!T), apply_vertical_kernel_simd!(AVX!T))
            (input, temp, hm, vm, output);
    }
    else if (sse)
    {
        inner_filtering_impl!(SSE!T, apply_horizontal_kernel_simd!(SSE!T), apply_vertical_kernel_simd!(SSE!T))
            (input, temp, hm, vm, output);
    }
    else
    {
        switch (ksize) {
        case 3:
            inner_filtering_impl!(Non_SIMD!T, apply_horizontal_kernel_3!T, apply_vertical_kernel_3!T)
                (input, temp, hm, vm, output);
            break;
        case 5:
            inner_filtering_impl!(Non_SIMD!T, apply_horizontal_kernel_5!T, apply_vertical_kernel_5!T)
                (input, temp, hm, vm, output);
            break;
        case 7:
            inner_filtering_impl!(Non_SIMD!T, apply_horizontal_kernel_7!T, apply_vertical_kernel_7!T)
                (input, temp, hm, vm, output);
            break;
        default:
            inner_filtering_impl!(Non_SIMD!T, apply_horizontal_kernel!T, apply_vertical_kernel!T)
                (input, temp, hm, vm, output);
        }
    }

    if (border_handling != BORDER_REPLICATE)
    {
        return ERROR_INVALID_BORDER_METHOD;
    }

    borders_replicate_impl(output, ksize);

    return 0;
}

pragma(inline, true)
void inner_filtering_impl
(
    alias InstructionSet,
    alias hkernel,
    alias vkernel
)
(
    Slice!(Contiguous, [2], InstructionSet.Scalar*) input,
    Slice!(Contiguous, [2], InstructionSet.Scalar*) temp,
    Slice!(Contiguous, [1], InstructionSet.Scalar*) hmask,
    Slice!(Contiguous, [1], InstructionSet.Scalar*) vmask,
    Slice!(Contiguous, [2], InstructionSet.Scalar*) output
)
in
{
    assert(input.shape == temp.shape, "Incompatible input and temporary buffer slice");
    assert(temp.shape == output.shape, "Incompatible input and output buffer slice");
}
body
{
    import mir.ndslice.topology : flattened, iota;

    alias T = InstructionSet.Scalar;
    alias V = InstructionSet.Vector;
    immutable velems = InstructionSet.elementCount;
    immutable tbytes = T.sizeof;
    immutable vbytes = V.sizeof;
    immutable ksize = hmask.length;
    immutable shape = input.shape;
    immutable rows = input.length!0;
    immutable cols = input.length!1;

    static if (Is_SIMD!InstructionSet)
    {
        // It this is SIMD instructions set, allocate vectors for kernels

        auto hk = cast(V[])alignedAllocate(ksize * vbytes, 16); // horizontal simd kernel
        auto vk = cast(V[])alignedAllocate(ksize * vbytes, 16); // vertical simd kernel

        scope (exit)
        {
            deallocate(hk);
            deallocate(vk);
        }

        foreach (i; 0 .. ksize)
        {
            hk[i].array[] = hmask[i];
        }

        foreach (i; 0 .. ksize)
        {
            vk[i].array[] = vmask[i];
        }
    }
    else
    {
        // ... otherwise just use input buffers.
        auto hk = hmask._iterator[0 .. ksize];
        auto vk = vmask._iterator[0 .. ksize];
    }

    auto _in = iota(shape, input._iterator).universal;
    auto _tmp = iota(shape, temp._iterator).universal;
    auto _out = iota(shape, output._iterator).universal;

    // Stride buffers to match vector indexing.
    auto a = _in[0 .. $ - ksize, 0 .. $ - ksize].strided!1(velems);
    auto t = _tmp[0 .. $ - ksize, 0 .. $ - ksize].strided!1(velems);
    auto b = _out[0 .. $ - ksize, 0 .. $ - ksize].strided!1(velems);

    // L1 cache block tiling (under 8192 bytes)
    immutable rtiling = 32;
    immutable ctiling = max(size_t(1), 256 / (tbytes * ksize));

    // Process blocks
    auto tiles = zip!true(a, t, b)
        .blocks(rtiling, ctiling)
        .flattened;

    foreach(tile; tiles) {
        auto flat_windows = tile.flattened;
        foreach(window; flat_windows) {
            hkernel(window.a, hk.ptr, window.b, ksize);
        }
        foreach(window; flat_windows) {
            vkernel(window.b, vk.ptr, window.c, ksize, cols);
        }
    }

    // Fill-in block horizontal borders
    auto middles = zip!true(t, b)[rtiling - ksize .. $ - ksize, 0 .. $]
        .windows(ksize, b.length!1)
        .strided!0(rtiling)
        .flattened;

    foreach(tile; middles)
        foreach(window; tile.flattened) {
            vkernel(window.a, vk.ptr, window.b, ksize, cols);
        }

    // perform scalar processing for the remaining pixels (from vector block selection)
    immutable rrb = a.length!0 - (a.length!0 % rtiling) - ksize;
    immutable crb = (a.length!1 - (a.length!1 % ctiling)) * velems - ksize;
    immutable rowstride = input.shape[1];

    // get scalar kernels
    auto hskernel = get_horizontal_kernel_for_mask!T(ksize);
    auto vskernel = get_vertical_kernel_for_mask!T(ksize);

    // bottom rows
    if (rrb < rows - ksize)
    {
        zip!true(_in, _tmp)[rrb .. $, 0 .. $ - ksize].each!((w) { hskernel(w.a, hmask._iterator, w.b, ksize); });
        zip!true(_tmp, _out)[rrb .. $ - ksize, 0 .. $ - ksize / 2].each!((w) { vskernel(w.a, vmask._iterator, w.b, ksize, rowstride); });
    }

    // right columns
    if (crb < cols - ksize)
    {
        zip!true(_in, _tmp)[0 .. $ - ksize, crb .. $ - ksize].each!((w) { hskernel(w.a, hmask._iterator, w.b, ksize); });
        zip!true(_tmp, _out)[0 .. $ - ksize, crb .. $ - ksize / 2].each!((w) { vskernel(w.a, vmask._iterator, w.b, ksize, rowstride); });
    }
}

pragma(inline, true)
void borders_replicate_impl(T)
(
    Slice!(Contiguous, [2], T*) output,
    size_t ksize,
)
{
    auto top = output[0 .. ksize / 2 + 1, ksize / 2 .. $ - ksize / 2 + 1];
    auto bottom = output[$ - ksize / 2 - 2 .. $, ksize / 2 .. $ - ksize / 2 + 1];

    foreach (c; 0 .. top.length!1)
        foreach (r; 0 .. top.length!0 - 1)
        {
            top[r, c] = top[ksize / 2 + 1, c];
            bottom[r + 1, c] = bottom[0, c];
        }

    auto left = output[0 .. $, 0 .. ksize / 2 + 1];
    auto right = output[0 .. $, $ - ksize / 2 - 2 .. $];

    foreach (r; 0 .. left.length!0)
        foreach (c; 0 .. left.length!1 - 1)
        {
            left[r, c] = left[r, ksize / 2 + 1];
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
