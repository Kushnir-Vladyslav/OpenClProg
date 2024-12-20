package org.example;

import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.*;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Instant;
import static org.example.Main.VECTOR_SIZE;

public class OpenClCode {

    public void start () {
        org.lwjgl.system.Configuration.OPENCL_EXPLICIT_INIT.set(true);
        CL.create();


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

            // Отримання платформ та пристроїв (без змін)
            IntBuffer platformCount = stack.mallocInt(1);
            CL10.clGetPlatformIDs(null, platformCount);

            PointerBuffer platforms = stack.mallocPointer(platformCount.get(0));
            CL10.clGetPlatformIDs(platforms, (IntBuffer) null);


            // Отримання інформації про платформи
            for (int i = 0; i < platformCount.get(0); i++) {
                long platformId = platforms.get(i);

                // Запитуємо розмір буфера для зберігання назви платформи
                PointerBuffer sizeBuffer = stack.mallocPointer(1);
                CL10.clGetPlatformInfo(platformId, CL10.CL_PLATFORM_NAME, (ByteBuffer) null, sizeBuffer);

                // Створюємо буфер для зчитування назви
                ByteBuffer nameBuffer = stack.malloc((int) sizeBuffer.get(0));
                CL10.clGetPlatformInfo(platformId, CL10.CL_PLATFORM_NAME, nameBuffer, null);

                // Перетворюємо буфер у строку
                String platformName = MemoryUtil.memUTF8(nameBuffer);
                System.out.println("Platform " + i + ": " + platformName);
            }

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

            // Завантаження OpenCL kernel
            long localMemSizeInfo = CL10.clGetDeviceInfo(device, CL10.CL_DEVICE_LOCAL_MEM_SIZE, (IntBuffer) null, (PointerBuffer) null );

            URL URLKernelSource;
            if (localMemSizeInfo > 0) {
                URLKernelSource = getClass().getResource("KernelSource.cl");
            } else {
                URLKernelSource = getClass().getResource("KernelSourceWithoutLocalMemory.cl");
            }

            assert URLKernelSource != null;
            String kernelSource = Files.readString(
                    Paths.get(URLKernelSource.toURI()));

            // Компіляція та створення kernel
            long program = CL10.clCreateProgramWithSource(context, kernelSource, null);
            CL10.clBuildProgram(program, device, "", null, 0);

            long kernel = CL10.clCreateKernel(program, "vector_add", (IntBuffer) null);

            // Перевірка чи правельно пройшла уомпіляція
//            int buildStatus = CL10.clBuildProgram(program, device, "", null, 0);
//            if (buildStatus != CL10.CL_SUCCESS) {
//                // Отримання журналу компіляції
//                PointerBuffer sizeBuffer = MemoryStack.stackMallocPointer(1);
//                CL10.clGetProgramBuildInfo(program, device, CL10.CL_PROGRAM_BUILD_LOG, (ByteBuffer) null, sizeBuffer);
//
//                ByteBuffer buildLogBuffer = MemoryStack.stackMalloc((int) sizeBuffer.get(0));
//                CL10.clGetProgramBuildInfo(program, device, CL10.CL_PROGRAM_BUILD_LOG, buildLogBuffer, null);
//
//                String buildLog = MemoryUtil.memUTF8(buildLogBuffer);
//                System.err.println("Build log:\n" + buildLog);
//                throw new RuntimeException("Failed to build OpenCL program.");
//            }

            // Локальна пам'ять
            long localMemSize = LOCAL_WORK_SIZE * Float.BYTES * 2;

            //валідація обєктів OpenCl
            if (clABuffer == 0 || clBBuffer == 0 || clResultBuffer == 0) {
                throw new IllegalStateException("Failed to create OpenCL memory buffers.");
            }
            if (kernel == 0) {
                throw new IllegalStateException("Failed to create OpenCL kernel.");
            }
            if (context == 0 || commandQueue == 0) {
                throw new IllegalStateException("Failed to create OpenCL context or command queue.");
            }

            // Встановлення аргументів
            CL10.clSetKernelArg(kernel, 0, PointerBuffer.allocateDirect(1).put(0, clABuffer));
            CL10.clSetKernelArg(kernel, 1, PointerBuffer.allocateDirect(1).put(0, clBBuffer));
            CL10.clSetKernelArg(kernel, 2, PointerBuffer.allocateDirect(1).put(0, clResultBuffer));

            // Локальна пам'ять як аргументи
            if (localMemSizeInfo > 0) {
                CL10.clSetKernelArg(kernel, 3, localMemSize);
                CL10.clSetKernelArg(kernel, 4, localMemSize);
            }

            // Виконання kernel з явним розміром локальної групи
            long globalWorkSize = (long) Math.ceil(VECTOR_SIZE / (float) LOCAL_WORK_SIZE) * LOCAL_WORK_SIZE;

            PointerBuffer global = stack.mallocPointer(1).put(globalWorkSize).rewind();
            PointerBuffer local = stack.mallocPointer(1).put(LOCAL_WORK_SIZE).rewind();

            Main.start = Instant.now();

            CL10.clEnqueueNDRangeKernel(
                    commandQueue, kernel, 1, null,
                    global, local,
                    null, null
            );

            Main.end = Instant.now();

            // Читання результату
            CL10.clEnqueueReadBuffer(commandQueue, clResultBuffer, true, 0,
                    resultBuffer, null, null);

//            long endTime = System.nanoTime();
//            System.out.printf("Час виконання: %.3f мс%n", (endTime - startTime) / 1_000_000.0);
//
            // Перевірка результату
//            for (int i = 0; i < 10; i++) {
//                System.out.printf("%.2f + %.2f = %.2f%n",
//                        aBuffer.get(i), bBuffer.get(i), resultBuffer.get(i));
//            }

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
