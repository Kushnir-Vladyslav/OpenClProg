package org.example;

import org.lwjgl.system.MemoryUtil;

import java.nio.FloatBuffer;
import java.time.Duration;
import java.time.Instant;

public class Main {
    public static Instant start;
    public static Instant end;

    public static void main(String[] args) {
//        start = Instant.now();
        OpenClCode OCC = new OpenClCode();
        OCC.start();
//        end = Instant.now();
        printExecutionTime(start, end);

        int VECTOR_SIZE = 1_000_000;
        FloatBuffer aBuffer = MemoryUtil.memAllocFloat(VECTOR_SIZE);
        FloatBuffer bBuffer = MemoryUtil.memAllocFloat(VECTOR_SIZE);
        FloatBuffer resultBuffer = MemoryUtil.memAllocFloat(VECTOR_SIZE);
        for (int i = 0; i < VECTOR_SIZE; i++) {
            aBuffer.put(i, (float) Math.random());
            bBuffer.put(i, (float) Math.random());
        }
        start = Instant.now();
        for(int i = 0; i < VECTOR_SIZE; i++) {
            resultBuffer.put(aBuffer.get(i) + bBuffer.get(i));

        }
        end = Instant.now();
        for(int i = 0; i < 10; i++) {
            System.out.printf("%.2f + %.2f = %.2f%n",
                    aBuffer.get(i), bBuffer.get(i), resultBuffer.get(i));
        }
        printExecutionTime(start, end);
    }
    private static void printExecutionTime(Instant start, Instant end) {
        Duration duration = Duration.between(start, end);
        System.out.println("Час виконання: " + duration.toSeconds() + " sec");
        System.out.println("Час виконання: " + duration.toMillis() + " millisec");
    }
}