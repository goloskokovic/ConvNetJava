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
public class ReluLayer implements ILayer {
    
    int out_sx;
    int out_sy;
    int out_depth;
    String layer_type = "relu";
    
    Vol in_act;
    Vol out_act;
    
    Util global = new Util();
    
    public int get_out_sx() { return out_sx; }
    public int get_out_sy() { return out_sy; }
    public int get_out_depth() { return out_depth; }
    public String get_layer_type() { return layer_type; }
    public Vol get_out_act() { return out_act; }
    
    public ReluLayer(Definition opt) {
        
        // computed
        this.out_sx = opt.in_sx;
        this.out_sy = opt.in_sy;
        this.out_depth = opt.in_depth;
        this.layer_type = "relu";
        
    }
    
    public Vol forward(Vol V, boolean is_training) {
      this.in_act = V;
      Vol V2 = V.cloneVol();
      int N = V.w.length;
      double[] V2w = V2.w;
      for(int i=0;i<N;i++) { 
        if(V2w[i] < 0) V2w[i] = 0; // threshold at 0
      }
      this.out_act = V2;
      return this.out_act;
    }
    
    public void backward() {
      Vol V = this.in_act; // we need to set dw of this
      Vol V2 = this.out_act;
      int N = V.w.length;
      V.dw = global.zeros(N); // zero out gradient wrt data
      for(int i=0;i<N;i++) {
        if(V2.w[i] <= 0) V.dw[i] = 0; // threshold
        else V.dw[i] = V2.dw[i];
      }
    }
    
    public ParamsAndGrads[] getParamsAndGrads() { 
      return new ParamsAndGrads[0];
    }
    
}
