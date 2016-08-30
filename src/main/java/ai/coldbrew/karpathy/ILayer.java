/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ai.coldbrew.karpathy;

/**
 *
 * @author gola
 */
public interface ILayer {
    
    int get_out_sx();
    int get_out_sy();
    int get_out_depth();
    Vol get_out_act();
    String get_layer_type();
    
    Vol forward(Vol V, boolean is_training);
    void backward();
    ParamsAndGrads[] getParamsAndGrads();    
}
