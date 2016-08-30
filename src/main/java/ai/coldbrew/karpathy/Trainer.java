/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ai.coldbrew.karpathy;

import java.util.Date;
import java.util.ArrayList;
import org.json.JSONObject;

/**
 *
 * @author gola
 */
public class Trainer {
    
    Net net;
    JSONObject options;
    
    double learning_rate;
    double l1_decay;
    double l2_decay;
    int batch_size;
    String method;

    double momentum;
    double ro;
    double eps;
    double beta1;
    double beta2;

    int k; // iteration counter
    ArrayList<double[]> gsum = new ArrayList<double[]>(); // last iteration gradients (used for momentum calculations)
    ArrayList<double[]> xsum = new ArrayList<double[]>(); // used in adam or adadelta
    
    boolean regression;
    
    Util global = new Util();
    
    public Trainer(Net net, String opt) {
        
        this.net = net;

        JSONObject options = new JSONObject(opt);
        this.learning_rate = options.has("learning_rate") ? options.getDouble("learning_rate") : 0.01;
        this.l1_decay = options.has("l1_decay") ? options.getDouble("l1_decay") : 0.0;
        this.l2_decay = options.has("l2_decay") ? options.getDouble("l2_decay") : 0.0;
        this.batch_size = options.has("batch_size") ? options.getInt("batch_size") : 1;
        this.method = options.has("method") ? options.getString("method") : "sgd"; // sgd/adam/adagrad/adadelta/windowgrad/netsterov

        this.momentum = options.has("momentum") ? options.getDouble("momentum") : 0.9;
        this.ro = options.has("ro") ? options.getDouble("ro") : 0.95; // used in adadelta
        this.eps = options.has("eps") ? options.getDouble("eps") : 1e-8; // used in adam or adadelta
        this.beta1 = options.has("beta1") ? options.getDouble("beta1") : 0.9; // used in adam
        this.beta2 = options.has("beta2") ? options.getDouble("beta2") : 0.999; // used in adam

        this.k = 0; // iteration counter
        //this.gsum = []; // last iteration gradients (used for momentum calculations)
        //this.xsum = []; // used in adam or adadelta

        // check if regression is expected 
        if(this.net.layers[this.net.layers.length - 1].get_layer_type().equals("regression"))
          this.regression = true;
        else
          this.regression = false;
        
    }
    
    Loss train(Vol x, int y) {

      long start = new Date().getTime();
      this.net.forward(x, true); // also set the flag that lets the net know we're just training
      long end = new Date().getTime();
      long fwd_time = end - start;

      start = new Date().getTime();
      double cost_loss = this.net.backward(y);
      double l2_decay_loss = 0.0;
      double l1_decay_loss = 0.0;
      end = new Date().getTime();
      double bwd_time = end - start;

      if(this.regression /*&& y.constructor !== Array*/)
        System.out.println("Warning: a regression net requires an array as training output vector.");
      
      this.k++;
      if(this.k % this.batch_size == 0) {

        ParamsAndGrads[] pglist = this.net.getParamsAndGrads();

        // initialize lists for accumulators. Will only be done once on first iteration
        if(this.gsum.isEmpty() && (!this.method.equals("sgd") || this.momentum > 0.0)) {
          // only vanilla sgd doesnt need either lists
          // momentum needs gsum
          // adagrad needs gsum
          // adam and adadelta needs gsum and xsum
          for(int i=0;i<pglist.length;i++) {
            this.gsum.add(global.zeros(pglist[i].params.length));
            if(this.method.equals("adam") || this.method.equals("adadelta")) {
              this.xsum.add(global.zeros(pglist[i].params.length));
            } else {
              this.xsum.add(new double[0]); // conserve memory 
            }
          }
        }

        // perform an update for all sets of weights
        for(int i=0;i<pglist.length;i++) {
          ParamsAndGrads pg = pglist[i]; // param, gradient, other options in future (custom learning rate etc)
          double[] p = pg.params;
          double[] g = pg.grads;

          // learning rate for some parameters.
          double l2_decay_mul = pg.l2_decay_mul != 0.0 ? pg.l2_decay_mul : 1.0;
          double l1_decay_mul = pg.l1_decay_mul != 0.0 ? pg.l1_decay_mul : 1.0;
          double l2_decay = this.l2_decay * l2_decay_mul;
          double l1_decay = this.l1_decay * l1_decay_mul;

          int plen = p.length;
          for(int j=0;j<plen;j++) {
            l2_decay_loss += l2_decay*p[j]*p[j]/2; // accumulate weight decay loss
            l1_decay_loss += l1_decay*Math.abs(p[j]);
            double l1grad = l1_decay * (p[j] > 0 ? 1 : -1);
            double l2grad = l2_decay * (p[j]);

            double gij = (l2grad + l1grad + g[j]) / this.batch_size; // raw batch gradient

            double[] gsumi = this.gsum.get(i);
            double[] xsumi = this.xsum.get(i);
            if(this.method.equals("adam")) {
              // adam update
              gsumi[j] = gsumi[j] * this.beta1 + (1- this.beta1) * gij; // update biased first moment estimate
              xsumi[j] = xsumi[j] * this.beta2 + (1-this.beta2) * gij * gij; // update biased second moment estimate
              double biasCorr1 = gsumi[j] * (1 - Math.pow(this.beta1, this.k)); // correct bias first moment estimate
              double biasCorr2 = xsumi[j] * (1 - Math.pow(this.beta2, this.k)); // correct bias second moment estimate
              double dx =  - this.learning_rate * biasCorr1 / (Math.sqrt(biasCorr2) + this.eps);
              p[j] += dx;
            } else if(this.method.equals("adagrad")) {
              // adagrad update
              gsumi[j] = gsumi[j] + gij * gij;
              double dx = - this.learning_rate / Math.sqrt(gsumi[j] + this.eps) * gij;
              p[j] += dx;
            } else if(this.method.equals("windowgrad")) {
              // this is adagrad but with a moving window weighted average
              // so the gradient is not accumulated over the entire history of the run. 
              // it's also referred to as Idea #1 in Zeiler paper on Adadelta. Seems reasonable to me!
              gsumi[j] = this.ro * gsumi[j] + (1-this.ro) * gij * gij;
              double dx = - this.learning_rate / Math.sqrt(gsumi[j] + this.eps) * gij; // eps added for better conditioning
              p[j] += dx;
            } else if(this.method.equals("adadelta")) {
              gsumi[j] = this.ro * gsumi[j] + (1-this.ro) * gij * gij;
              double dx = - Math.sqrt((xsumi[j] + this.eps)/(gsumi[j] + this.eps)) * gij;
              xsumi[j] = this.ro * xsumi[j] + (1-this.ro) * dx * dx; // yes, xsum lags behind gsum by 1.
              p[j] += dx;
            } else if(this.method.equals("nesterov")) {
            	double dx = gsumi[j];
            	gsumi[j] = gsumi[j] * this.momentum + this.learning_rate * gij;
                dx = this.momentum * dx - (1.0 + this.momentum) * gsumi[j];
                p[j] += dx;
            } else {
              // assume SGD
              if(this.momentum > 0.0) {
                // momentum update
                double dx = this.momentum * gsumi[j] - this.learning_rate * gij; // step
                gsumi[j] = dx; // back this up for next iteration of momentum
                p[j] += dx; // apply corrected gradient
              } else {
                // vanilla sgd
                p[j] +=  - this.learning_rate * gij;
              }
            }
            g[j] = 0.0; // zero out gradient so that we can begin accumulating anew
          }
        }
      }

      // appending softmax_loss for backwards compatibility, but from now on we will always use cost_loss
      // in future, TODO: have to completely redo the way loss is done around the network as currently 
      // loss is a bit of a hack. Ideally, user should specify arbitrary number of loss functions on any layer
      // and it should all be computed correctly and automatically. 
      return new Loss(fwd_time, bwd_time, l2_decay_loss, l1_decay_loss, cost_loss, cost_loss, cost_loss + l1_decay_loss + l2_decay_loss);
    }
}
