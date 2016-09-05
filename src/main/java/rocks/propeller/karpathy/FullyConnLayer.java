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
public class FullyConnLayer implements ILayer {
    
    int out_depth;

    // optional 
    double l1_decay_mul;
    double l2_decay_mul;

    // computed
    int num_inputs;
    int out_sx;
    int out_sy;
    String layer_type = "fc";

    Vol[] filters;
    Vol biases;
    
    Vol in_act;
    Vol out_act;
    
    Util global = new Util();
    
    public int get_out_sx() { return out_sx; }
    public int get_out_sy() { return out_sy; }
    public int get_out_depth() { return out_depth; }
    public String get_layer_type() { return layer_type; }
    public Vol get_out_act() { return out_act; }
    
    public FullyConnLayer(Definition opt) {
    
        // required
        // ok fine we will allow 'filters' as the word as well
        this.out_depth = opt.num_neurons != 0 ? opt.num_neurons : opt.filters;

        // optional 
        this.l1_decay_mul = opt.l1_decay_mul;
        this.l2_decay_mul = opt.l2_decay_mul != 0.0 ? opt.l2_decay_mul : 1.0;

        // computed
        this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
        this.out_sx = 1;
        this.out_sy = 1;
        this.layer_type = "fc";

        // initializations
        double bias = opt.bias_pref;
        this.filters = new Vol[this.out_depth];
        for(int i=0;i<this.out_depth ;i++) { 
            this.filters[i] = new Vol(1, 1, this.num_inputs, null); 
        }
        this.biases = new Vol(1, 1, this.out_depth, bias);
    
    }
    
    public Vol forward(Vol V, boolean is_training) {
      this.in_act = V;
      Vol A = new Vol(1, 1, this.out_depth, 0.0);
      double[] Vw = V.w;
      for(int i=0;i<this.out_depth;i++) {
        double a = 0.0;
        double[] wi = this.filters[i].w;
        for(int d=0;d<this.num_inputs;d++) {
          a += Vw[d] * wi[d]; // for efficiency use Vols directly for now
        }
        a += this.biases.w[i];
        A.w[i] = a;
      }
      this.out_act = A;
      return this.out_act;
    }
    
    public void backward() {
      Vol V = this.in_act;
      V.dw = global.zeros(V.w.length); // zero out the gradient in input Vol
      
      // compute gradient wrt weights and data
      for(int i=0;i<this.out_depth;i++) {
        Vol tfi = this.filters[i];
        double chain_grad = this.out_act.dw[i];
        for(int d=0;d<this.num_inputs;d++) {
          V.dw[d] += tfi.w[d]*chain_grad; // grad wrt input data
          tfi.dw[d] += V.w[d]*chain_grad; // grad wrt params
        }
        this.biases.dw[i] += chain_grad;
      }
    }
    
    public ParamsAndGrads[] getParamsAndGrads() {
      ParamsAndGrads[] response = new ParamsAndGrads[this.out_depth + 1];
      for(int i=0;i<this.out_depth;i++) {
          response[i] = new ParamsAndGrads(this.filters[i].w, this.filters[i].dw, this.l1_decay_mul, this.l2_decay_mul);
      }
      response[this.out_depth] = new ParamsAndGrads(this.biases.w, this.biases.dw, 0.0, 0.0);
      
      return response;
    }
    
}
