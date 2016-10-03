/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rocks.propeller.karpathy;

import java.util.ArrayList;
import org.json.JSONObject;

/**
 *
 * @author gola
 */
public class Net {
    
    ILayer[] layers;
    
    Util global = new Util();
    JsonUtil jutil = new JsonUtil();
    
    // takes a list of layer definitions and creates the network layer objects
    void makeLayers(String[] layer_defs) {

      // few checks
      //global.Assert(defs.length >= 2, "Error! At least one input layer and one loss layer are required.") throws Exception;
      //global.Assert(defs[0].getString("type") == "input", "Error! First layer must be the input layer, to declare size of inputs");
      
      Definition[] defs = desugar(layer_defs);
      
      // create the layers
      layers = new ILayer[defs.length];
      for(int i=0;i<defs.length;i++) {
        Definition def = defs[i];
        if(i>0) {
          ILayer prev = layers[i-1];
          def.in_sx = (int)prev.get_out_sx();
          def.in_sy = (int)prev.get_out_sy();
          def.in_depth = prev.get_out_depth();
        }

        switch(def.type) {
          case "fc": layers[i] = new FullyConnLayer(def); break;
          //case "lrn": layers[i] = new LocalResponseNormalizationLayer(def); break;
          //case "dropout": layers[i] = new DropoutLayer(def); break;
          case "input": layers[i] = new InputLayer(def); break;
          case "softmax": layers[i] = new SoftmaxLayer(def); break;
          //case "regression": layers[i] = new RegressionLayer(def); break;
          case "conv": layers[i] = new ConvLayer(def); break;
          case "pool": layers[i] = new PoolLayer(def); break;
          case "relu": layers[i] = new ReluLayer(def); break;
          //case "sigmoid": layers[i] = SigmoidLayer(def); break;
          //case "tanh": layers[i] = new TanhLayer(def); break;
          //case "maxout": layers[i] = new MaxoutLayer(def); break;
          //case "svm": layers[i] = new SVMLayer(def); break;
          default: System.out.println("ERROR: UNRECOGNIZED LAYER TYPE: " + def.type);
        }
      }
      
    }
      

    // desugar layer_defs for adding activation, dropout layers etc
      Definition[] desugar(String[] layer_defs) {
          
        JSONObject[] defs = new JSONObject[layer_defs.length];
        for(int i = 0; i<layer_defs.length; i++) { defs[i] = new JSONObject(layer_defs[i]); }
        
        ArrayList<Definition> new_defs = new ArrayList<Definition>();
        for(int i=0;i<defs.length;i++) {
          JSONObject def = defs[i];
          
          if(def.getString("type").equals("softmax") || def.getString("type").equals("svm")) {
            // add an fc layer here, there is no reason the user should
            // have to worry about this and we almost always want to
            new_defs.add(new Definition("fc", def.getInt("num_classes")));
          }

          if(def.getString("type").equals("regression")) {
            // add an fc layer here, there is no reason the user should
            // have to worry about this and we almost always want to
            new_defs.add(new Definition("fc", def.getInt("num_neurons")));
          }
          
          Definition layer = new Definition(def.getString("type"));

          if((def.getString("type").equals("fc") || def.getString("type").equals("conv")) && def.has("bias_pref")){
            layer.bias_pref = 0.0;
            if(def.has("activation") && def.getString("activation").equals("relu")) {
              layer.bias_pref = 0.1; // relus like a bit of positive bias to get gradients early
              // otherwise it's technically possible that a relu unit will never turn on (by chance)
              // and will never get any gradient and never contribute any computation. Dead relu.
            }
          }
          
          if(def.has("out_sx"))
            layer.out_sx = def.getInt("out_sx");
          if(def.has("out_sy"))
            layer.out_sy = def.getInt("out_sy");
          if(def.has("out_depth"))
            layer.out_depth = def.getInt("out_depth");
          
          if(def.has("sx"))
            layer.sx = def.getInt("sx");
          if(def.has("filters"))
            layer.filters = def.getInt("filters");
          if(def.has("stride"))
            layer.stride = def.getInt("stride");
          if(def.has("pad"))
            layer.pad = def.getInt("pad");
          if(def.has("activation"))
            layer.activation = def.getString("activation");
          if(def.has("num_classes"))
            layer.num_neurons = def.getInt("num_classes");
          
          new_defs.add(layer);

          if(def.has("activation")) {
            if(def.getString("activation").equals("relu")) { 
                new_defs.add(new Definition("relu")); 
            }
            else if (def.getString("activation").equals("sigmoid")) { 
                new_defs.add(new Definition("sigmoid")); 
            }
            else if (def.getString("activation").equals("tanh")) { 
                new_defs.add(new Definition("tanh")); 
            }
            else if (def.getString("activation").equals("maxout")) {
              // create maxout activation, and pass along group size, if provided
              int gs = def.has("group_size") ? def.getInt("group_size") : 2;
              new_defs.add(new Definition(gs, "maxout"));
            }
            else { 
                //console.log('ERROR unsupported activation ' + def.activation);
                System.out.println("ERROR unsupported activation " + def.getString("activation"));
            }
          }
          
          if(def.has("drop_prob") && def.getString("type").equals("dropout")) {
              Definition temp = new Definition("dropout");
              temp.drop_prob = def.getString("drop_prob");
              new_defs.add(temp);
          }

        }
        return new_defs.toArray(new Definition[0]);
      }
      
    // forward prop the network. 
    // The trainer class passes is_training = true, but when this function is
    // called from outside (not from the trainer), it defaults to prediction mode
    Vol forward(Vol V, boolean is_training) {
      //if(typeof(is_training) === 'undefined') is_training = false;
      Vol act = this.layers[0].forward(V, is_training);
      for(int i=1;i<this.layers.length;i++) {
        act = this.layers[i].forward(act, is_training);
      }
      return act;
    }

    double getCostLoss(Vol V, int y) {
      this.forward(V, false);
      int N = this.layers.length;
      double loss = ((SoftmaxLayer)this.layers[N-1]).backward(y);
      return loss;
    }
    
    // backprop: compute gradients wrt all parameters
    double backward(int y) {
      int N = this.layers.length;
      double loss = ((SoftmaxLayer)this.layers[N-1]).backward(y); // last layer assumed to be loss layer
      for(int i=N-2;i>=0;i--) { // first layer assumed input
        this.layers[i].backward();
      }
      return loss;
    }
    
    ParamsAndGrads[] getParamsAndGrads() {
      // accumulate parameters and gradients for the entire network
      ArrayList<ParamsAndGrads> response = new ArrayList<ParamsAndGrads>();
      for(int i=0;i<this.layers.length;i++) {
        ParamsAndGrads[] layer_reponse = this.layers[i].getParamsAndGrads();
        for(int j=0;j<layer_reponse.length;j++) {
          response.add(layer_reponse[j]);
        }
      }
      return response.toArray(new ParamsAndGrads[0]);
    }
    
    double getPrediction() {
      // this is a convenience function for returning the argmax
      // prediction, assuming the last layer of the net is a softmax
      ILayer S = this.layers[this.layers.length-1];
      global.Assert(S.get_layer_type().equals("softmax"), "getPrediction function assumes softmax as last layer of the net!");

      double[] p = S.get_out_act().w;
      double maxv = p[0];
      int maxi = 0;
      for(int i=1;i<p.length;i++) {
        if(p[i] > maxv) { maxv = p[i]; maxi = i; }
      }
      return maxi; // return index of the class with highest class probability
    }
       
    void toJSON(String filePath) throws java.lang.Exception { jutil.toJSON(filePath, layers); }
    void fromJSON(String filePath) throws java.lang.Exception { jutil.fromJSON(filePath, layers); }
}
