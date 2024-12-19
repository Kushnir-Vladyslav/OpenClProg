__kernel void vector_add(__global const float *a,
                         __global const float *b,
                         __global float *result)
{

    result[get_global_id(0)] = a[get_global_id(0)] + b[get_global_id(0)];

}