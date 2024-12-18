package org.example;

import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.*;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

public class OpenClCode {

    public void start () {
        org.lwjgl.system.Configuration.OPENCL_EXPLICIT_INIT.set(true);
        CL.create();

        final int VECTOR_SIZE = 16_000_000; // Збільшено розмір
        final int LOCAL_WORK_SIZE = 256; // Оптимальний розмір локальної групи

        FloatBuffer aBuffer = MemoryUtil.memAllocFloat(VECTOR_SIZE);
        FloatBuffer bBuffer = MemoryUtil.memAllocFloat(VECTOR_SIZE);
        FloatBuffer resultBuffer = MemoryUtil.memAllocFloat(VECTOR_SIZE);

        long startTime = System.nanoTime();

        try (MemoryStack stack = MemoryStack.stackPush()) {
            // Заповнення вхідних векторів
            for (int i = 0; i < VECTOR_SIZE; i++) {
                aBuffer.put(i, (float) Math.random());
                bBuffer.put(i, (float) Math.random());
            }
            aBuffer.rewind();
            bBuffer.rewind();

            // Розширений OpenCL kernel з локальною пам'яттю
            String kernelSource =
                    "__kernel void vector_add(__global const float *a, " +
                            "                         __global const float *b, " +
                            "                         __global float *result, " +
                            "                         __local float *local_a, " +
                            "                         __local float *local_b) {" +
                            "    int gid = get_global_id(0);" +
                            "    int lid = get_local_id(0);" +
                            "    int group_size = get_local_size(0);" +

                            "    // Завантаження даних в локальну пам'ять" +
                            "    local_a[lid] = a[gid];" +
                            "    local_b[lid] = b[gid];" +

                            "    // Синхронізація локальної групи" +
                            "    barrier(CLK_LOCAL_MEM_FENCE);" +

                            "    // Додавання з використанням локальної пам'яті" +
                            "    result[gid] = local_a[lid] + local_b[lid];" +
                            "}";

            // Отримання платформ та пристроїв (без змін)
            IntBuffer platformCount = stack.mallocInt(1);
            CL10.clGetPlatformIDs(null, platformCount);

            PointerBuffer platforms = stack.mallocPointer(platformCount.get(0));
            CL10.clGetPlatformIDs(platforms, (IntBuffer) null);

            long platform = platforms.get(0);

            IntBuffer deviceCount = stack.mallocInt(1);
            CL10.clGetDeviceIDs(platform, CL10.CL_DEVICE_TYPE_GPU, null, deviceCount);

            PointerBuffer devices = stack.mallocPointer(deviceCount.get(0));
            CL10.clGetDeviceIDs(platform, CL10.CL_DEVICE_TYPE_GPU, devices, (IntBuffer) null);

            long device = devices.get(0);

            // Додаткова діагностика пристрою
            try (MemoryStack infoStack = MemoryStack.stackPush()) {
                PointerBuffer paramSize = infoStack.mallocPointer(1);
                CL10.clGetDeviceInfo(device, CL10.CL_DEVICE_MAX_COMPUTE_UNITS, (IntBuffer)null, paramSize);

                IntBuffer paramValue = infoStack.mallocInt((int) paramSize.get(0));
                CL10.clGetDeviceInfo(device, CL10.CL_DEVICE_MAX_COMPUTE_UNITS, paramValue, null);

                System.out.printf("Максимальна кількість обчислювальних блоків: %d%n", paramValue.get(0));
            }

            // Створення контексту та черги
            PointerBuffer contextProperties = stack.mallocPointer(3)
                    .put(CL10.CL_CONTEXT_PLATFORM)
                    .put(platform)
                    .put(0)
                    .rewind();

            long context = CL10.clCreateContext(contextProperties, device, null, 0, null);
            long commandQueue = CL10.clCreateCommandQueue(context, device, CL10.CL_QUEUE_PROFILING_ENABLE,(IntBuffer) null);

            // Створення буферів
            long clABuffer = CL10.clCreateBuffer(context, CL10.CL_MEM_READ_ONLY | CL10.CL_MEM_COPY_HOST_PTR,
                    aBuffer, null);
            long clBBuffer = CL10.clCreateBuffer(context, CL10.CL_MEM_READ_ONLY | CL10.CL_MEM_COPY_HOST_PTR,
                    bBuffer, null);
            long clResultBuffer = CL10.clCreateBuffer(context, CL10.CL_MEM_WRITE_ONLY,
                    VECTOR_SIZE * Float.BYTES, null);

            // Компіляція та створення kernel
            long program = CL10.clCreateProgramWithSource(context, kernelSource, null);
            CL10.clBuildProgram(program, device, "", null, 0);

            long kernel = CL10.clCreateKernel(program, "vector_add", (IntBuffer) null);

            // Локальна пам'ять
            long localMemSize = LOCAL_WORK_SIZE * Float.BYTES * 2;

            // Встановлення аргументів
            LongBuffer clMemBuffer = stack.mallocLong(1);

            clMemBuffer.put(0, clABuffer);
            CL10.clSetKernelArg(kernel, 0, clMemBuffer);

            clMemBuffer.put(0, clBBuffer);
            CL10.clSetKernelArg(kernel, 1, clMemBuffer);

            clMemBuffer.put(0, clResultBuffer);
            CL10.clSetKernelArg(kernel, 2, clMemBuffer);

            // Локальна пам'ять як аргументи
            CL10.clSetKernelArg(kernel, 3, localMemSize);
            CL10.clSetKernelArg(kernel, 4, localMemSize);

            // Виконання kernel з явним розміром локальної групи
            long globalWorkSize = (long) Math.ceil(VECTOR_SIZE / (float) LOCAL_WORK_SIZE) * LOCAL_WORK_SIZE;

            PointerBuffer global = stack.mallocPointer(1).put(globalWorkSize).rewind();
            PointerBuffer local = stack.mallocPointer(1).put(LOCAL_WORK_SIZE).rewind();

            CL10.clEnqueueNDRangeKernel(
                    commandQueue, kernel, 1, null,
                    global, local,
                    null, null
            );

            // Читання результату
            CL10.clEnqueueReadBuffer(commandQueue, clResultBuffer, true, 0,
                    resultBuffer, null, null);

            long endTime = System.nanoTime();
            System.out.printf("Час виконання: %.3f мс%n", (endTime - startTime) / 1_000_000.0);

            // Перевірка результату
            for (int i = 0; i < 10; i++) {
                System.out.printf("%.2f + %.2f = %.2f%n",
                        aBuffer.get(i), bBuffer.get(i), resultBuffer.get(i));
            }

            // Очищення ресурсів (без змін)
            CL10.clReleaseKernel(kernel);
            CL10.clReleaseProgram(program);
            CL10.clReleaseMemObject(clABuffer);
            CL10.clReleaseMemObject(clBBuffer);
            CL10.clReleaseMemObject(clResultBuffer);
            CL10.clReleaseCommandQueue(commandQueue);
            CL10.clReleaseContext(context);

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            MemoryUtil.memFree(aBuffer);
            MemoryUtil.memFree(bBuffer);
            MemoryUtil.memFree(resultBuffer);
            CL.destroy();
        }
    }
}
