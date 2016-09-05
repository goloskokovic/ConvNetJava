/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rocks.propeller.karpathy;

import java.util.ArrayList;

/**
 *
 * @author gola
 */
public class LossUtil {
    
    // a window stores _size_ number of values
    // and returns averages. Useful for keeping running
    // track of validation or training accuracy during SGD
    ArrayList<Double> v = new ArrayList<Double>();
    int size = 100;
    int minsize = 20;
    double sum = 0;
    
    public LossUtil() {}
    
    public LossUtil(int size, int minsize) {
        this.size = size;
        this.minsize = minsize;
    }
    
    void add(double x) {
      this.v.add(x);
      this.sum += x;
      if(this.v.size()>this.size) {
        double xold = this.v.remove(0);
        this.sum -= xold;
      }
    }
    
    double get_average() {
      if(this.v.size() < this.minsize) return -1;
      else return this.sum/this.v.size();
    }
    
    void reset() {
      this.v.clear();
      this.sum = 0;
    }
    
    // returns string representation of float
    // but truncated to length of d digits
    double f2t() {
        int d = 5;
        double dd = 1.0 * Math.pow(10, d);
        double x = get_average();
        return Math.floor(x*dd)/dd;
    }
    
}
