package ai.coldbrew.karpathy;

/**
 *
 * @author gola
 */
public class SoftmaxLayer implements ILayer {
    
    int num_inputs;
    int out_depth;
    int out_sx;
    int out_sy;
    String layer_type = "softmax";
    
    Vol in_act;
    Vol out_act;
    double[] es;
    
    Util global = new Util();
    
    public int get_out_sx() { return out_sx; }
    public int get_out_sy() { return out_sy; }
    public int get_out_depth() { return out_depth; }
    public String get_layer_type() { return layer_type; }
    public Vol get_out_act() { return out_act; }
    
    public SoftmaxLayer(Definition opt) {
        
        // computed
        this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
        this.out_depth = this.num_inputs;
        this.out_sx = 1;
        this.out_sy = 1;
        this.layer_type = "softmax";
    }
    
    public Vol forward(Vol V, boolean is_training) {
      this.in_act = V;

      Vol A = new Vol(1, 1, this.out_depth, 0.0);

      // compute max activation
      double[] as = V.w;
      double amax = V.w[0];
      for(int i=1;i<this.out_depth;i++) {
        if(as[i] > amax) amax = as[i];
      }

      // compute exponentials (carefully to not blow up)
      double[] es = global.zeros(this.out_depth);
      double esum = 0.0;
      for(int i=0;i<this.out_depth;i++) {
        double e = Math.exp(as[i] - amax);
        esum += e;
        es[i] = e;
      }

      // normalize and output to sum to one
      for(int i=0;i<this.out_depth;i++) {
        es[i] /= esum;
        A.w[i] = es[i];
      }

      this.es = es; // save these for backprop
      this.out_act = A;
      return this.out_act;
    }
    
    double backward(int y) {

      // compute and accumulate gradient wrt weights and bias of this layer
      Vol x = this.in_act;
      x.dw = global.zeros(x.w.length); // zero out the gradient of input Vol

      for(int i=0;i<this.out_depth;i++) {
        double indicator = i == y ? 1.0 : 0.0;
        double mul = -(indicator - this.es[i]);
        x.dw[i] = mul;
      }

      // loss is the class negative log likelihood
      return -Math.log(this.es[y]);
    }
    
    public void backward() { }
    
    @Override
    public ParamsAndGrads[] getParamsAndGrads() { 
      return new ParamsAndGrads[0];
    }
    
}
