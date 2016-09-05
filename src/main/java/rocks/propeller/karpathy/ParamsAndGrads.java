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
public class ParamsAndGrads {

    double[] params;
    double[] grads;
    double l1_decay_mul;
    double l2_decay_mul;
    
    public ParamsAndGrads(double[] params, double[] grads, double l1_decay_mul, double l2_decay_mul) {
        this.params = params;
        this.grads = grads;
        this.l1_decay_mul = l1_decay_mul;
        this.l2_decay_mul = l2_decay_mul;
    }    
}
