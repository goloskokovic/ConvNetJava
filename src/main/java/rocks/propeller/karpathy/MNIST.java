/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rocks.propeller.karpathy;

import java.util.Date;
import java.util.List;
import java.util.Map;

/**
 *
 * @author gola
 */
public class MNIST {
    
    public static void main(String[] args) throws Exception {
        
        int num_layers = 6;
        String[] layer_defs = new String[num_layers];  
        
        layer_defs[0] = "{type:'input', out_sx:24, out_sy:24, out_depth:1}";       
        layer_defs[1] = "{type:'conv', sx:5, filters:8, stride:1, pad:2, activation:'relu'}";
        layer_defs[2] = "{type:'pool', sx:2, stride:2}";
        layer_defs[3] = "{type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'}";
        layer_defs[4] = "{type:'pool', sx:3, stride:3}";
        layer_defs[5] = "{type:'softmax', num_classes:10}";
        
        String folder = "/home/gola/mnist/train/";
        DataSet.loadImagePaths(folder);
        
        List<String> paths = DataSet.getImagePaths();
        Map<Integer, String> classes = DataSet.getClassLabels();
        
        Net net = new Net();
        net.makeLayers(layer_defs);
        Trainer trainer = new Trainer(net, "{method:'adadelta', batch_size:20, l2_decay:0.001}");
             
        VolUtil util = new VolUtil();
        LossUtil xLoss = new LossUtil();
        LossUtil wLoss = new LossUtil();
        LossUtil trainAcc = new LossUtil();
        LossUtil valAcc = new LossUtil();
        
        int iteration = 0; 
        long start = new Date().getTime();
        
//        do {
        
        for(String path : paths) {
            
            String[] image_class = path.split(" ");
            String image = image_class[0];
            String iclass = image_class[1];
            int output = Integer.parseInt(iclass);
            String label = classes.get(output);
            
            Vol x = util.imagePath_to_vol(image, true);
            
            iteration++;
            if(iteration %100 == 0) {
                // use x to build our estimate of validation error
                net.forward(x, false);
                double yhat = net.getPrediction();
                double val_acc = yhat == output ? 1.0 : 0.0;
                valAcc.add(val_acc);
                continue; // get out
            }
            
            // train on it with network
            Loss stats = trainer.train(x, output);

            double lossx = stats.cost_loss;
            double lossw = stats.l2_decay_loss;

            // keep track of stats such as the average training error and loss
            double yhat = net.getPrediction();
            double train_acc = yhat == output ? 1.0 : 0.0;
            xLoss.add(lossx);
            wLoss.add(lossw);
            trainAcc.add(train_acc);
            
            System.out.println("====================================");
            System.out.println("iteration: " + iteration);
            System.out.println("label: " + label);
            System.out.println("Forward time per example: " + stats.fwd_time + " ms");
            System.out.println("Backprop time per example: " + stats.bwd_time + " ms");
            System.out.println("Classification loss: " + xLoss.f2t());
            System.out.println("L2 Weight decay loss: " + wLoss.f2t());
            System.out.println("Training accuracy: " + trainAcc.f2t());
            System.out.println("Validation accuracy: " + valAcc.f2t());
            
        }
        
//            if (valAcc.f2t() > 0.6)
//                net.toJSON("/home/gola/Documents/mnist_test.json");
        
//        }while (trainAcc.f2t() < 0.66);
        
        long end = new Date().getTime();
        System.out.println("All training time: " + (end-start) + " ms");
               
    }
    
}
