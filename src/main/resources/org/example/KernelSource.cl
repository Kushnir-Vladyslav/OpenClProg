__kernel void vector_add(__global const float *a,
                         __global const float *b,
                         __global float *result,
                         __local float *local_a,
                         __local float *local_b)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);

    // "Завантаження даних в локальну пам'ять"
    local_a[lid] = a[gid];
    local_b[lid] = b[gid];

    // "Синхронізація локальної групи"
    barrier(CLK_LOCAL_MEM_FENCE);
    // "Додавання з використанням локальної пам'яті"
    result[gid] = local_a[lid] + local_b[lid];
}