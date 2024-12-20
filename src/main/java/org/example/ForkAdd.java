package org.example;

import static org.example.Main.*;
import java.util.concurrent.RecursiveAction;

public class ForkAdd extends RecursiveAction{
    int i;

    ForkAdd(int i) {
        this.i = i;
    }

    @Override
    protected void compute() {
        if (i == VECTOR_SIZE) {
            return;
        }
        new ForkAdd(i + 1).fork();

        for(int i = 0; i < VECTOR_SIZE; i++) {
            for (int k = 0; k < 1_000_000; k++) {
                resultBuffer[i] =  (aBuffer[i] + bBuffer[i]) * (i % 10);
            }
        }
    }
}
