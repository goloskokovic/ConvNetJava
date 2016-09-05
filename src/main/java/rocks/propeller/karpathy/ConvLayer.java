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
public class ConvLayer implements ILayer {
    
    int out_depth;
    int sx; // filter size. Should be odd if possible, it's cleaner.
    int in_depth;
    int in_sx;
    int in_sy;
    Vol[] filters;
    Vol biases;
    Vol in_act;
    Vol out_act;
    
    int sy;
    int stride; // stride at which we apply filters to input volume
    int pad; // amount of 0 padding to add around borders of input volume
    double l1_decay_mul;
    double l2_decay_mul;
    
    int out_sx;
    int out_sy;
    String layer_type = "conv";
    
    Util global = new Util();
    
    public int get_out_sx() { return out_sx; }
    public int get_out_sy() { return out_sy; }
    public int get_out_depth() { return out_depth; }
    public String get_layer_type() { return layer_type; }
    public Vol get_out_act() { return out_act; }
    
    public ConvLayer(Definition opt) {
    
        // required
        this.out_depth = opt.filters;
        this.sx = opt.sx; // filter size. Should be odd if possible, it's cleaner.
        this.in_depth = opt.in_depth;
        this.in_sx = opt.in_sx;
        this.in_sy = opt.in_sy;
        
        // optional
        this.sy = opt.sy != 0 ? opt.sy : this.sx;
        this.stride = opt.stride != 0 ? opt.stride : 1; // stride at which we apply filters to input volume
        this.pad = opt.pad; // amount of 0 padding to add around borders of input volume
        this.l1_decay_mul = opt.l1_decay_mul;
        this.l2_decay_mul = opt.l2_decay_mul != 0.0 ? opt.l2_decay_mul : 1.0;
        
        // computed
        // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
        // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
        // final application.
        this.out_sx = (int)Math.floor((this.in_sx + this.pad * 2 - this.sx) / this.stride + 1);
        this.out_sy = (int)Math.floor((this.in_sy + this.pad * 2 - this.sy) / this.stride + 1);
        this.layer_type = "conv";
        
        // initializations
        double bias = opt.bias_pref;
        this.filters = new Vol[this.out_depth];
        for(int i=0;i<this.out_depth;i++) { this.filters[i] = new Vol(this.sx, this.sy, this.in_depth, null); }
        this.biases = new Vol(1, 1, this.out_depth, bias);
    
    }
    
    public Vol forward(Vol V, boolean is_training) {
      
      this.in_act = V;
      Vol A = new Vol(this.out_sx |0, this.out_sy |0, this.out_depth |0, 0.0);

      int V_sx = V.sx |0;
      int V_sy = V.sy |0;
      int xy_stride = this.stride |0;

      for(int d=0;d<this.out_depth;d++) {
        Vol f = this.filters[d];
        int x = -this.pad |0;
        int y = -this.pad |0;
        for(int ay=0; ay<this.out_sy; y+=xy_stride,ay++) {  // xy_stride
          x = -this.pad |0;
          for(int ax=0; ax<this.out_sx; x+=xy_stride,ax++) {  // xy_stride

            // convolve centered at this particular location
            double a = 0.0;
            for(int fy=0;fy<f.sy;fy++) {
              int oy = y+fy; // coordinates in the original input array coordinates
              for(int fx=0;fx<f.sx;fx++) {
                int ox = x+fx;
                if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx) {
                  for(int fd=0;fd<f.depth;fd++) {
                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                    a += f.w[((f.sx * fy)+fx)*f.depth+fd] * V.w[((V_sx * oy)+ox)*V.depth+fd];
                  }
                }
              }
            }
            a += this.biases.w[d];
            A.set(ax, ay, d, a);
          }
        }
      }
      this.out_act = A;
      
      return this.out_act;
    }
    
    public void backward() {

      Vol V = this.in_act;
      V.dw = global.zeros(V.w.length); // zero out gradient wrt bottom data, we're about to fill it

      int V_sx = V.sx |0;
      int V_sy = V.sy |0;
      int xy_stride = this.stride |0;

      for(int d=0;d<this.out_depth;d++) {
        Vol f = this.filters[d];
        int x = -this.pad |0;
        int y = -this.pad |0;
        for(int ay=0; ay<this.out_sy; y+=xy_stride,ay++) {  // xy_stride
          x = -this.pad |0;
          for(int ax=0; ax<this.out_sx; x+=xy_stride,ax++) {  // xy_stride

            // convolve centered at this particular location
            double chain_grad = this.out_act.get_grad(ax,ay,d); // gradient from above, from chain rule
            for(int fy=0;fy<f.sy;fy++) {
              int oy = y+fy; // coordinates in the original input array coordinates
              for(int fx=0;fx<f.sx;fx++) {
                int ox = x+fx;
                if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx) {
                  for(int fd=0;fd<f.depth;fd++) {
                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                    int ix1 = ((V_sx * oy)+ox)*V.depth+fd;
                    int ix2 = ((f.sx * fy)+fx)*f.depth+fd;
                    f.dw[ix2] += V.w[ix1]*chain_grad;
                    V.dw[ix1] += f.w[ix2]*chain_grad;
                  }
                }
              }
            }
            this.biases.dw[d] += chain_grad;
          }
        }
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
