/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rocks.propeller.karpathy;

import java.util.Date;

/**
 *
 * @author gola
 */
public class Benchmark {
    
    public static void main(String[] args) {
        // L1 Conv Layer definition
        
        // { in_sx:128, in_sy:128, in_depth:3, sx:11, filters:96, stride: 1, pad: 0 }
        Definition opt = new Definition("conv");
        opt.in_sx = 128; opt.in_sy = 128; opt.in_depth = 3; opt.sx = 11; opt.filters = 96; opt.stride = 1; opt.pad = 0;
        
        ILayer layer = new ConvLayer(opt);

        // create a random input volume
        Vol x = new Vol(128, 128, 3, null);

        // run it through batch_size number of times
        int batch_size = 128;
        int dtall = 0;
        for(int i=0;i<batch_size;i++) { // batch of 128
          long t0 = new Date().getTime();
          layer.forward(x, false); // forward
          long t1 = new Date().getTime();
          long dt = t1 - t0;
          dtall += dt;
          
          System.out.println(i + " took " + dt + "ms. Estimating full batch to take " + (dtall/(i+1))*batch_size + "ms");
        }
        
        System.out.println("total: " + dtall + "ms"); 
    }
}
