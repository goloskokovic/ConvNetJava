package rocks.propeller.karpathy;

/**
 *
 * @author gola
 */
public class Vol {
    
    int sx ;
    int sy;
    int depth;
    double[] w;
    double[] dw;
    
    Util global = new Util();
    
    public Vol(int sx, int sy, int depth, Double c) {
      // we were given dimensions of the vol
      this.sx = sx;
      this.sy = sy;
      this.depth = depth;
      int n = sx*sy*depth;
      this.w = new double[n];
      this.dw = global.zeros(n);
      if(c == null) {
        // weight normalization is done to equalize the output
        // variance of every neuron, otherwise neurons with a lot
        // of incoming connections have outputs of larger variance
        double scale = Math.sqrt(1.0/(sx*sy*depth));
        for(int i=0;i<n;i++) { 
          this.w[i] = global.randn(0.0, scale);
        }
      } 
      else {
        for(int i=0;i<n;i++) { 
          this.w[i] = c;
        }
      }
    }
    
    public Vol(double[] sx) {
      // we were given a list in sx, assume 1D volume and fill it up
      this.sx = 1;
      this.sy = 1;
      this.depth = sx.length;
      // we have to do the following copy because we want to use
      // fast typed arrays, not an ordinary javascript array
      this.w = global.zeros(this.depth);
      this.dw = global.zeros(this.depth);
      for(int i=0;i<this.depth;i++) {
        this.w[i] = sx[i];
      }
    }
    
    double get(int x, int y, int d) { 
      int ix=((this.sx * y)+x)*this.depth+d;
      return this.w[ix];
    }
    
    void set(int x, int y, int d, double v) { 
      int ix=((this.sx * y)+x)*this.depth+d;
      this.w[ix] = v; 
    }
    
    void add(int x, int y, int d, int v) { 
      int ix=((this.sx * y)+x)*this.depth+d;
      this.w[ix] += v; 
    }
    double get_grad(int x, int y, int d) { 
      int ix = ((this.sx * y)+x)*this.depth+d;
      return this.dw[ix]; 
    }
    
    void set_grad(int x, int y, int d, int v) { 
      int ix = ((this.sx * y)+x)*this.depth+d;
      this.dw[ix] = v; 
    }
    
    void add_grad(int x, int y, int d, double v) { 
      int ix = ((this.sx * y)+x)*this.depth+d;
      this.dw[ix] += v; 
    }
    
    Vol cloneAndZero() { return new Vol(this.sx, this.sy, this.depth, 0.0); }
    
 
    Vol cloneVol() {
      Vol v = new Vol(this.sx, this.sy, this.depth, 0.0);
      int n = this.w.length;
      for(int i=0;i<n;i++) { v.w[i] = this.w[i]; }
      return v;
    }
    
    void addFrom(Vol v) { for(int k=0;k<this.w.length;k++) { this.w[k] += v.w[k]; }}
    void addFromScaled(Vol v, double a) { for(int k=0;k<this.w.length;k++) { this.w[k] += a*v.w[k]; }}
    void setConst(double a) { for(int k=0;k<this.w.length;k++) { this.w[k] = a; }}
    
}
