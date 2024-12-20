package org.example;

import org.lwjgl.system.MemoryUtil;

import java.nio.FloatBuffer;
import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.ForkJoinPool;

public class Main {
    public static Instant start;
    public static Instant end;
    public static int VECTOR_SIZE = 1_000_000_000;

    public static float[] aBuffer = new float[VECTOR_SIZE];
    public static float[] bBuffer = new float[VECTOR_SIZE];
    public static float[] resultBuffer = new float[VECTOR_SIZE];

    public static void main(String[] args) {
//        start = Instant.now();
        OpenClCode OCC = new OpenClCode();
        OCC.start();
//        end = Instant.now();
        printExecutionTime(start, end);



        for (int i = 0; i < VECTOR_SIZE; i++) {
            aBuffer[i] = (float) Math.random();
            bBuffer[i] = (float) Math.random();
        }
        start = Instant.now();
        ForkJoinPool pool = new ForkJoinPool();
        pool.invoke(new ForkAdd(0));

        end = Instant.now();
//        for(int i = 0; i < 10; i++) {
//            System.out.printf("%.2f + %.2f = %.2f%n",
//                    aBuffer.get(i), bBuffer.get(i), resultBuffer.get(i));
//        }
        printExecutionTime(start, end);
    }
    private static void printExecutionTime(Instant start, Instant end) {
        Duration duration = Duration.between(start, end);
        System.out.println("Час виконання: " + duration.toSeconds() + " sec");
        System.out.println("Час виконання: " + duration.toMillis() + " millisec");
    }
}