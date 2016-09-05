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
public class PoolLayer implements ILayer {
    
    int sx;
    int in_depth;
    int in_sx;
    int in_sy;

    int sy;
    int stride;
    int pad;

    int out_depth;
    int out_sx;
    int out_sy;
    String layer_type = "pool";
    
    double[] switchx;
    double[] switchy;
    
    Vol in_act;
    Vol out_act;
    
    Util global = new Util();
    
    public int get_out_sx() { return out_sx; }
    public int get_out_sy() { return out_sy; }
    public int get_out_depth() { return out_depth; }
    public String get_layer_type() { return layer_type; }
    public Vol get_out_act() { return out_act; }

    public PoolLayer(Definition opt) {
        
        // required
        this.sx = opt.sx; // filter size
        this.in_depth = opt.in_depth;
        this.in_sx = opt.in_sx;
        this.in_sy = opt.in_sy;

        // optional
        this.sy = opt.sy != 0 ? opt.sy : this.sx;
        this.stride = opt.stride != 0 ? opt.stride : 2;
        this.pad = opt.pad; // amount of 0 padding to add around borders of input volume

        // computed
        this.out_depth = this.in_depth;
        this.out_sx = (int)Math.floor((this.in_sx + this.pad * 2 - this.sx) / this.stride + 1);
        this.out_sy = (int)Math.floor((this.in_sy + this.pad * 2 - this.sy) / this.stride + 1);
        this.layer_type = "pool";
        
        // store switches for x,y coordinates for where the max comes from, for each output neuron
        this.switchx = global.zeros(this.out_sx*this.out_sy*this.out_depth);
        this.switchy = global.zeros(this.out_sx*this.out_sy*this.out_depth);
    }
    
    public Vol forward(Vol V, boolean is_training) {
      this.in_act = V;

      Vol A = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);
      
      int n=0; // a counter for switches
      for(int d=0;d<this.out_depth;d++) {
        int x = -this.pad;
        int y = -this.pad;
        for(int ax=0; ax<this.out_sx; x+=this.stride,ax++) {
          y = -this.pad;
          for(int ay=0; ay<this.out_sy; y+=this.stride,ay++) {

            // convolve centered at this particular location
            int a = -99999; // hopefully small enough ;\
            int winx=-1,winy=-1;
            for(int fx=0;fx<this.sx;fx++) {
              for(int fy=0;fy<this.sy;fy++) {
                int oy = y+fy;
                int ox = x+fx;
                if(oy>=0 && oy<V.sy && ox>=0 && ox<V.sx) {
                  Double v = V.get(ox, oy, d);
                  // perform max pooling and store pointers to where
                  // the max came from. This will speed up backprop 
                  // and can help make nice visualizations in future
                  if(v > a) { a = v.intValue(); winx=ox; winy=oy;}
                }
              }
            }
            this.switchx[n] = (double)winx;
            this.switchy[n] = (double)winy;
            n++;
            A.set(ax, ay, d, a);
          }
        }
      }
      this.out_act = A;
      return this.out_act;
    }
    
    public void backward() { 
      // pooling layers have no parameters, so simply compute 
      // gradient wrt data here
      Vol V = this.in_act;
      V.dw = global.zeros(V.w.length); // zero out gradient wrt data
      Vol A = this.out_act; // computed in forward pass 

      int n = 0;
      for(int d=0;d<this.out_depth;d++) {
        int x = -this.pad;
        int y = -this.pad;
        for(int ax=0; ax<this.out_sx; x+=this.stride,ax++) {
          y = -this.pad;
          for(int ay=0; ay<this.out_sy; y+=this.stride,ay++) {

            double chain_grad = this.out_act.get_grad(ax,ay,d);
            V.add_grad((int)this.switchx[n], (int)this.switchy[n], d, chain_grad);
            n++;

          }
        }
      }
    }
    
    public ParamsAndGrads[] getParamsAndGrads() { 
      return new ParamsAndGrads[0];
    }
    
}
