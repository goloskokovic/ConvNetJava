/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rocks.propeller.karpathy;

import java.util.Iterator;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.json.JSONArray;
import org.json.JSONObject;

/**
 *
 * @author gola
 */
public class JsonUtil {
    
    void toJSON(String filePath, ILayer[] layers) throws java.lang.Exception {
    
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
    
    void fromJSON(String filePath, ILayer[] layers) throws java.lang.Exception {
        
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
                    //l.switchx = global.zeros(l.out_sx*l.out_sy*l.out_depth); // need to re-init these appropriately
                    //l.switchy = global.zeros(l.out_sx*l.out_sy*l.out_depth);
                    
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
