/+
    Utilities to help use SIMD vectors.
+/

module mir.cv.core.simd;

import cpuid.x86_any;
import std.traits : isNumeric, TemplateOf;

extern(C) @system nothrow @nogc:

private
{
    // TODO: Do we need this?
    struct Init_cpu_id
    {
        static void init()
        {
            cpuid_x86_any_init();
        }
    }
    static Init_cpu_id __cpuid_init;
}


/++
    SIMD instructionset traits.
+/
mixin template Instruction_set_trait(size_t _bitsize, T)
{
    enum bitsize = _bitsize;
    enum elementCount = (bitsize / 8) / T.sizeof;

    alias Vector = .Vector!(bitsize, T);
    alias Scalar = T;
}

/// SSE (128bit) instruction set descriptor.
template SSE(T)
if (isNumeric!T)
{
    mixin Instruction_set_trait!(128, T);
}

/// AVX (256bit) instruction set descriptor.
template AVX(T)
if (isNumeric!T)
{
    mixin Instruction_set_trait!(256, T);
}

/// Non-simd, instruction set mock-up.
template Non_SIMD(T)
if (isNumeric!T)
{
    enum bitsize = 8;
    enum elementCount = 1;
    alias Vector = T;
    alias Scalar = T;
}

template Is_SIMD(alias InstructionSet){
    static if (__traits(isSame, TemplateOf!InstructionSet, SSE)) {
        enum Is_SIMD = true;
    } else static if (__traits(isSame, TemplateOf!InstructionSet, AVX) ) {
        enum Is_SIMD = true;
    } else {
        enum Is_SIMD = false;
    }
}

/// SIMD vector trait - build vector type using bitsize and scalar type.
template Vector(size_t bitsize, T)
{
    alias Vector = __vector(T[(bitsize / 8) / T.sizeof]);
}

