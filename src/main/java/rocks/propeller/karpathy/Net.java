/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rocks.propeller.karpathy;

import java.util.ArrayList;
import java.util.Iterator;
import org.json.JSONObject;
import org.json.JSONArray;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;

/**
 *
 * @author gola
 */
public class Net {
    
    ILayer[] layers;
    
    Util global = new Util();
    
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
       
    void toJSON(String filePath) throws java.lang.Exception {
    
        JSONObject network = new JSONObject();
        JSONArray jsonLayers = new JSONArray();
        network.put("layers", jsonLayers);        
                
        for(ILayer layer : layers) {
            switch(layer.get_layer_type()) {
                case "conv": {
                    JSONObject jsonLayer = new JSONObject();
                    ConvLayer l = (ConvLayer)layer;
                    jsonLayer.put("sx", l.sx);
                    jsonLayer.put("sy", l.sy);
                    jsonLayer.put("stride", l.stride);
                    jsonLayer.put("in_depth", l.in_depth);
                    jsonLayer.put("out_depth", l.out_depth);
                    jsonLayer.put("out_sx", l.out_sx);
                    jsonLayer.put("out_sy", l.out_sy);
                    jsonLayer.put("layer_type", l.layer_type);
                    jsonLayer.put("l1_decay_mul", l.l1_decay_mul);
                    jsonLayer.put("l2_decay_mul", l.l2_decay_mul);
                    jsonLayer.put("pad", l.pad);
                    
                    JSONArray jsonFilters = new JSONArray();
                    for (Vol filter : l.filters) {
                        JSONObject jsonFilter = new JSONObject();
                        jsonFilter.put("sx", filter.sx);
                        jsonFilter.put("sy", filter.sy);
                        jsonFilter.put("depth", filter.depth);
                    
                        JSONObject jsonW = new JSONObject();
                        for(int wI=0; wI<filter.w.length; wI++) { 
                            jsonW.put((String.valueOf(wI)), filter.w[wI]);
                        }
                        jsonFilter.put("w", jsonW);
                        jsonFilters.put(jsonFilter);
                    }
                                        
                    JSONObject jsonBiases = new JSONObject();
                    jsonBiases.put("sx", l.biases.sx);
                    jsonBiases.put("sy", l.biases.sy);
                    jsonBiases.put("depth", l.biases.depth);
                    
                    JSONObject jsonW = new JSONObject();
                    for(int wI=0; wI<l.biases.w.length; wI++) { 
                        jsonW.put((String.valueOf(wI)), l.biases.w[wI]);
                    }
                    
                    jsonBiases.put("w", jsonW);
                    jsonLayer.put("biases", jsonBiases);
                    jsonLayers.put(jsonLayer);
                                       
                    break;
                }
                case "fc": {
                    JSONObject jsonLayer = new JSONObject();
                    FullyConnLayer l = (FullyConnLayer)layer;
                    jsonLayer.put("out_depth", l.out_depth);
                    jsonLayer.put("out_sx", l.out_sx);
                    jsonLayer.put("out_sy", l.out_sy);
                    jsonLayer.put("num_inputs", l.num_inputs);
                    jsonLayer.put("l1_decay_mul", l.l1_decay_mul);
                    jsonLayer.put("l2_decay_mul", l.l2_decay_mul);                    
                    
                    JSONArray jsonFilters = new JSONArray();
                    for (Vol filter : l.filters) {
                        JSONObject jsonFilter = new JSONObject();
                        jsonFilter.put("sx", filter.sx);
                        jsonFilter.put("sy", filter.sy);
                        jsonFilter.put("depth", filter.depth);
                    
                        JSONObject jsonW = new JSONObject();
                        for(int wI=0; wI<filter.w.length; wI++) { 
                            jsonW.put((String.valueOf(wI)), filter.w[wI]);
                        }
                        jsonFilter.put("w", jsonW);
                        jsonFilters.put(jsonFilter);
                    }
                                        
                    JSONObject jsonBiases = new JSONObject();
                    jsonBiases.put("sx", l.biases.sx);
                    jsonBiases.put("sy", l.biases.sy);
                    jsonBiases.put("depth", l.biases.depth);
                    
                    JSONObject jsonW = new JSONObject();
                    for(int wI=0; wI<l.biases.w.length; wI++) { 
                        jsonW.put((String.valueOf(wI)), l.biases.w[wI]);
                    }
                    
                    jsonBiases.put("w", jsonW);
                    jsonLayer.put("biases", jsonBiases);
                    jsonLayers.put(jsonLayer);
                                       
                    break;
                }
                case "input": { 
                    JSONObject jsonLayer = new JSONObject();
                    InputLayer l = (InputLayer)layer;
                    jsonLayer.put("out_depth", l.out_depth);
                    jsonLayer.put("out_sx", l.out_sx);
                    jsonLayer.put("out_sy", l.out_sy);
                    jsonLayer.put("layer_type", l.layer_type);
                    jsonLayers.put(jsonLayer);
                    break;
                }
                case "softmax": {
                    JSONObject jsonLayer = new JSONObject();
                    SoftmaxLayer l = (SoftmaxLayer)layer;
                    jsonLayer.put("out_depth", l.out_depth);
                    jsonLayer.put("out_sx", l.out_sx);
                    jsonLayer.put("out_sy", l.out_sy);
                    jsonLayer.put("num_inputs", l.num_inputs);
                    
                    break;
                }
                case "pool": {
                    JSONObject jsonLayer = new JSONObject();
                    PoolLayer l = (PoolLayer)layer;
                    jsonLayer.put("sx", l.sx);
                    jsonLayer.put("sy", l.sy);
                    jsonLayer.put("stride", l.stride);
                    jsonLayer.put("in_depth", l.in_depth);
                    jsonLayer.put("out_depth", l.out_depth);
                    jsonLayer.put("out_sx", l.out_sx);
                    jsonLayer.put("out_sy", l.out_sy);
                    jsonLayer.put("pad", l.pad);
                                        
                    break;
                }
                case "relu": {
                    JSONObject jsonLayer = new JSONObject();
                    ReluLayer l = (ReluLayer)layer;
                    jsonLayer.put("out_depth", l.out_depth);
                    jsonLayer.put("out_sx", l.out_sx);
                    jsonLayer.put("out_sy", l.out_sy);
                                        
                    break;
                }
            }
        }
        
        DataSet.crateTextFile(filePath, network.toString());
        
    }
    
    void fromJSON(String filePath) throws java.lang.Exception {
        
        ObjectMapper mapper = new ObjectMapper();
        JsonNode root = mapper.readTree(new java.io.File(filePath));
        JsonNode layersNode = root.path("layers");
        
        if (layersNode.size() != layers.length)
            throw new Exception("Number of layers in JSON has to be equal to the number in the network");
        
        int layerI=0;
        for (Iterator i = layersNode.elements(); i.hasNext();) {
            JsonNode layerNode = (JsonNode)i.next();
            ILayer layer = loadLayer(layerNode, layers[layerI++]);
        }        
    }
    
    ILayer loadLayer(JsonNode layerNode, ILayer layer) throws java.lang.Exception {
        
        String layer_type = layerNode.path("layer_type").asText();
        
        if (!layer.get_layer_type().equals(layer_type))
            throw new Exception("Layers are not of same type"); 
        
        switch(layer_type) {
                case "conv": {
                    ConvLayer l = (ConvLayer)layer;
                    l.out_depth = layerNode.path("out_depth").asInt();
                    l.out_sx = layerNode.path("out_sx").asInt();
                    l.out_sy = layerNode.path("out_sy").asInt();
                    l.sx = layerNode.path("sx").asInt();
                    l.sy = layerNode.path("sy").asInt();
                    l.stride = layerNode.path("stride").asInt();
                    l.in_depth = layerNode.path("in_depth").asInt();
                    l.l1_decay_mul = layerNode.path("l1_decay_mul").asDouble();
                    l.l2_decay_mul = layerNode.path("l2_decay_mul").asDouble();
                    l.pad = layerNode.path("pad").asInt();                                        

                    JsonNode biasesNode = layerNode.path("biases");
                    l.biases = new Vol(
                            biasesNode.path("sx").asInt(), 
                            biasesNode.path("sy").asInt(), 
                            biasesNode.path("depth").asInt(), 
                            0.0);

                    int wI=0;
                    for (Iterator i = biasesNode.path("w").elements(); i.hasNext();) {
                        JsonNode wNode = (JsonNode)i.next();
                        l.biases.w[wI++] = wNode.asDouble();
                    }
                    
                    int filI=0;
                    JsonNode filtersNode = layerNode.path("filters");
                    l.filters = new Vol[filtersNode.size()];
                    for (Iterator i = filtersNode.elements(); i.hasNext();) {
                        JsonNode volNode = (JsonNode)i.next();
                        Vol v = new Vol(
                            volNode.path("sx").asInt(), 
                            volNode.path("sy").asInt(), 
                            volNode.path("depth").asInt(), 
                            0.0);
                        
                        int filwI=0;
                        for (Iterator j = volNode.path("w").elements(); j.hasNext();) {
                            JsonNode wNode = (JsonNode)j.next();
                            v.w[filwI++] = wNode.asDouble();
                        }
                        
                        l.filters[filI++] = v;
                    }
                    
                    break;
                }
                case "fc": {
                    FullyConnLayer l = (FullyConnLayer)layer;
                    l.out_depth = layerNode.path("out_depth").asInt();
                    l.out_sx = layerNode.path("out_sx").asInt();
                    l.out_sy = layerNode.path("out_sy").asInt();
                    l.num_inputs = layerNode.path("num_inputs").asInt();
                    l.l1_decay_mul = layerNode.path("l1_decay_mul").asDouble();
                    l.l2_decay_mul = layerNode.path("l2_decay_mul").asDouble();

                    JsonNode biasesNode = layerNode.path("biases");
                    l.biases = new Vol(
                            biasesNode.path("sx").asInt(), 
                            biasesNode.path("sy").asInt(), 
                            biasesNode.path("depth").asInt(), 
                            0.0);
                    
                    int wI=0;
                    for (Iterator i = biasesNode.path("w").elements(); i.hasNext();) {
                        JsonNode wNode = (JsonNode)i.next();
                        l.biases.w[wI++] = wNode.asDouble();
                    } 

                    int filI=0;
                    JsonNode filtersNode = layerNode.path("filters");
                    l.filters = new Vol[filtersNode.size()];
                    for (Iterator i = filtersNode.elements(); i.hasNext();) {
                        JsonNode volNode = (JsonNode)i.next();
                        Vol v = new Vol(
                            volNode.path("sx").asInt(), 
                            volNode.path("sy").asInt(), 
                            volNode.path("depth").asInt(), 
                            0.0);
                        
                        int filwI=0;
                        for (Iterator j = volNode.path("w").elements(); j.hasNext();) {
                            JsonNode wNode = (JsonNode)j.next();
                            v.w[filwI++] = wNode.asDouble();
                        }
                        
                        l.filters[filI++] = v;
                    }
                                
                    break;                    
                }
                case "input": {  
                    InputLayer l = (InputLayer)layer;
                    l.out_depth = layerNode.path("out_depth").asInt();
                    l.out_sx = layerNode.path("out_sx").asInt();
                    l.out_sy = layerNode.path("out_sy").asInt();
                                        
                    break;
                }
                case "softmax": {
                    SoftmaxLayer l = (SoftmaxLayer)layer;
                    l.out_depth = layerNode.path("out_depth").asInt();
                    l.out_sx = layerNode.path("out_sx").asInt();
                    l.out_sy = layerNode.path("out_sy").asInt();
                    l.num_inputs = layerNode.path("num_inputs").asInt();

                    break;
                }
                case "pool": {
                    PoolLayer l = (PoolLayer)layer;
                    l.out_depth = layerNode.path("out_depth").asInt();
                    l.out_sx = layerNode.path("out_sx").asInt();
                    l.out_sy = layerNode.path("out_sy").asInt();
                    l.sx = layerNode.path("sx").asInt();
                    l.sy = layerNode.path("sy").asInt();
                    l.stride = layerNode.path("stride").asInt();
                    l.in_depth = layerNode.path("in_depth").asInt();
                    l.pad = layerNode.path("pad").asInt();
                    l.switchx = global.zeros(l.out_sx*l.out_sy*l.out_depth); // need to re-init these appropriately
                    l.switchy = global.zeros(l.out_sx*l.out_sy*l.out_depth);
                    
                    break;
                }
                case "relu": {
                    ReluLayer l = (ReluLayer)layer;    
                    l.out_depth = layerNode.path("out_depth").asInt();
                    l.out_sx = layerNode.path("out_sx").asInt();
                    l.out_sy = layerNode.path("out_sy").asInt();
                    
                    break;
                }
            }
        
        return layer;
    }
}
