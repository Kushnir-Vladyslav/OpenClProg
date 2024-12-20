__kernel void vector_add(__global const float *a,
                         __global const float *b,
                         __global float *result)
{
    int GID = get_global_id(0);

    for (int i = 0; i < 1000000; i++) {
        result[GID] += (a[GID] - b[GID]) * (i % 10);
    }

}