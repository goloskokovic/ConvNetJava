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
public class InputLayer implements ILayer {
    
    // required: depth
    int out_depth;

    // optional: default these dimensions to 1
    int out_sx;
    int out_sy;
    
    Vol in_act;
    Vol out_act;
    
    // computed
    String layer_type = "input";
    
    Util global = new Util();
    
    public int get_out_sx() { return out_sx; }
    public int get_out_sy() { return out_sy; }
    public int get_out_depth() { return out_depth; }
    public String get_layer_type() { return layer_type; }
    public Vol get_out_act() { return out_act; }
    
    public InputLayer() {} 
    
    public InputLayer(Definition opt) {
    
        // required: depth
        //this.out_depth = getopt(opt, ['out_depth', 'depth'], 0);
        this.out_depth = global.getopt(0, opt.out_depth, opt.depth);

        // optional: default these dimensions to 1
        //this.out_sx = getopt(opt, ['out_sx', 'sx', 'width'], 1);
        this.out_sx = global.getopt(1, opt.out_sx, opt.sx, opt.width);
        //this.out_sy = getopt(opt, ['out_sy', 'sy', 'height'], 1);
        this.out_sy = global.getopt(1, opt.out_sy, opt.sy, opt.height);
    
        // computed
        this.layer_type = "input";
    }
    
    public Vol forward(Vol V, boolean is_training) {
        this.in_act = V;
        this.out_act = V;
        return this.out_act; // simply identity function for now
    }
    
    public void backward() { }
    
    public ParamsAndGrads[] getParamsAndGrads() { 
        return new ParamsAndGrads[0];
    }
    
}
