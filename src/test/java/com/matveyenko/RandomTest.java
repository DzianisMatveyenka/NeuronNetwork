package com.matveyenko;

import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.util.stream.DoubleStream;

/**
 * Created by 1 on 06.11.2017.
 */
public class RandomTest {

    @Test
    public void test() {
        Random random = new Random();
        DoubleStream doubles = random.doubles(100, -1 * Math.pow(100, -0.5), Math.pow(100, -0.5));
        doubles.forEach(System.out::println);
    }
}
