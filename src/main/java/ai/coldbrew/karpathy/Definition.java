/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ai.coldbrew.karpathy;

import java.util.Map;

/**
 *
 * @author gola
 */
public class Definition {

    String type;
    int num_neurons;
    String activation;
    int group_size;
    
    int sx; // filter size in x, y dims
    int sy;
    
    int filters;
    int stride;
    int pad;
    
    int in_depth; // depth of input volume
    int in_sx;
    int in_sy;
        
    int out_depth;
    int depth;
    int out_sx;
    int out_sy;
    
    int width;
    int height;
    
    String layer_type;
    
    double l1_decay_mul;
    double l2_decay_mul;
    
    double bias_pref;
    String drop_prob;
    
    public Definition(String type, int num_neurons) {
        this.type = type;
        this.num_neurons = num_neurons;
    }
    
    public Definition(int group_size, String type) {
        this.type = type;
        this.group_size = group_size;
    }
    
    public Definition(String type) {
        this.type = type;
    }
}
