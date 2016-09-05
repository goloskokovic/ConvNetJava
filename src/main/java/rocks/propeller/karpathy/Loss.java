/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rocks.propeller.karpathy;

/**
 *
 * @author gola
 */
public class Loss {

    double fwd_time;
    double bwd_time; 
    double l2_decay_loss;
    double l1_decay_loss; 
    double cost_loss;
    double softmax_loss;
    double loss;
    
    public Loss(double fwd_time, double bwd_time, double l2_decay_loss, double l1_decay_loss, double cost_loss, double softmax_loss, double loss) {
        this.fwd_time = fwd_time;
        this.bwd_time = bwd_time;
        this.l2_decay_loss = l2_decay_loss;
        this.l1_decay_loss = l1_decay_loss;
        this.cost_loss = cost_loss;
        this.softmax_loss = softmax_loss;
        this.loss = loss;
    }
    
}
