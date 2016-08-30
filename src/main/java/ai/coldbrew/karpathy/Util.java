/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ai.coldbrew.karpathy;

import java.util.*;

/**
 *
 * @author gola
 */
public class Util {
    
    // Random number utilities
    boolean return_v = false;
    double v_val = 0.0;
    
    double gaussRandom() {
        if(return_v) { 
          return_v = false;
          return v_val; 
        }
        
        double u = 2*Math.random()-1;
        double v = 2*Math.random()-1;
        double r = u*u + v*v;
        if(r == 0 || r > 1) 
            return gaussRandom();
        
        double c = Math.sqrt(-2*Math.log(r)/r);
        v_val = v*c; // cache this
        return_v = true;
        return u*c;
    }
    
    double randf(double a, double b) { return Math.random()*(b-a)+a; }
    int randi(double a, double b) { return (int)Math.floor(Math.random()*(b-a)+a); }
    double randn(double mu, double std){ return mu+gaussRandom()*std; }
    
    // Array utilities
    double[] zeros(int n) {
        if(n == 0) { return new double[0]; }
        
        double[] arr = new double[n];
//        for(int i=0;i<n;i++) { 
//            arr[i]= 0; 
//        }
        
        return arr;        
    }
    
    double[] convertDoubles(List<Double> arr) {
        double[] ret = new double[arr.size()];
        for (int i=0; i < ret.length; i++) {
            ret[i] = arr.get(i);
        }
        return ret;
    }
    
    boolean arrContains(List<Double> arr, double elt) {
        for(int i=0,n=arr.size();i<n;i++) {
            if(arr.get(i) == elt) return true;
        }
        return false;
    }
    
    double[] arrUnique(double[] arr) {
        ArrayList<Double> b = new ArrayList<Double>();
        //var b = [];
        for(int i=0,n=arr.length;i<n;i++) {
            if(!arrContains(b, arr[i])) {
                b.add(arr[i]);
            }
        }
        return convertDoubles(b);
    }
    
    // return max and min of a given non-empty array.
    double[] maxmin(double[] w) {
        if(w.length == 0) { return new double[0]; } // ... ;s
        double maxv = w[0];
        double minv = w[0];
        int maxi = 0;
        int mini = 0;
        int n = w.length;
        for(int i=1;i<n;i++) {
            if(w[i] > maxv) { maxv = w[i]; maxi = i; } 
            if(w[i] < minv) { minv = w[i]; mini = i; } 
        }
        return new double[] { maxi, maxv, mini, minv, maxv-minv };
    }
    
    // create random permutation of numbers, in range [0...n-1]
    int[] randperm(int n) {
        int i = n, j = 0, temp;
        int[] array = new int[n];
        for(int q=0;q<n;q++)
            array[q]=q;
        while (i-- != 0) {
            j = (int)Math.floor(Math.random() * (i+1));
            temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
        return array;
    }
    
    // sample from list lst according to probabilities in list probs
    // the two lists are of same size, and probs adds up to 1
    double weightedSample(double[] lst, double[] probs) {
        double p = randf(0, 1.0);
        double cumprob = 0.0;
        for(int k=0,n=lst.length;k<n;k++) {
            cumprob += probs[k];
            if(p < cumprob) { return lst[k]; }
        }
        return 0.0;
    }
    
    // syntactic sugar function for getting default parameter values
    int getopt(int default_value, int... field_value) {
        if(field_value.length == 1) {
            // case of single string
            return field_value[0];
        } 
        else {
            // assume we are given a list of string instead
            int ret = default_value;
            for(int i=0;i<field_value.length;i++) {
                int f = field_value[i];
                if (f != 0) {
                    ret = f; // overwrite return value
                }
            }
            return ret;
        }
    }
    
    void Assert(boolean condition, String message) {
        if (!condition) {
            //throw new Exception(message); // Fallback
            System.out.println(message);
        }
    }
    
}
