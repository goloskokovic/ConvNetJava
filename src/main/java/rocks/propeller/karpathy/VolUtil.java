/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rocks.propeller.karpathy;

import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;

/**
 *
 * @author gola
 */
public class VolUtil {
    
  Util global = new Util();  
    
  // Volume utilities
  // intended for use with data augmentation
  // crop is the size of output
  // dx,dy are offset wrt incoming volume, of the shift
  // fliplr is boolean on whether we also want to flip left<->right
  Vol augment(Vol V, int crop, Integer dx, Integer dy, Boolean fliplr) {
    // note assumes square outputs of size crop x crop
    if(fliplr == null) fliplr = false;
    if(dx == null) dx = global.randi(0, V.sx - crop);
    if(dy == null) dy = global.randi(0, V.sy - crop);
    
    // randomly sample a crop in the input volume
    Vol W;
    if(crop != V.sx || dx!=0 || dy!=0) {
      W = new Vol(crop, crop, V.depth, 0.0);
      for(int x=0;x<crop;x++) {
        for(int y=0;y<crop;y++) {
          if(x+dx<0 || x+dx>=V.sx || y+dy<0 || y+dy>=V.sy) continue; // oob
          for(int d=0;d<V.depth;d++) {
            W.set( x, y, d, V.get(x+dx, y+dy, d)); // copy data over
          }
        }
      }
    } else {
      W = V;
    }

    if(fliplr) {
      // flip volume horziontally
      Vol W2 = W.cloneAndZero();
      for(int x=0;x<W.sx;x++) {
        for(int y=0;y<W.sy;y++) {
          for(int d=0;d<W.depth;d++) {
           W2.set(x,y,d,W.get(W.sx - x - 1,y,d)); // copy data over
          }
        }
      }
      W = W2; //swap
    }
    return W;
  }
  
  Vol imagePath_to_vol(String imagePath, boolean convert_grayscale) {
      
        BufferedImage image = null;
        try { 
            image = ImageIO.read(new File(imagePath)); 
        } 
        catch (Exception ex) { 
            System.out.println(ex);
        }
        
        return img_to_vol(image, convert_grayscale);
  }
  
  // img is a DOM element that contains a loaded image
  // returns a Vol of size (W, H, 4). 4 is for RGBA
  Vol img_to_vol(BufferedImage img, boolean convert_grayscale) {

    int RGBA = 4, RGB = 3, BW = 1;  
      
    // prepare the input: get pixels and normalize them
    int[] p = getRgbByteArray(img); //getByteArray(img);
    int W = img.getWidth();
    int H = img.getHeight();
    double[] pv = new double[p.length];
    for(int i=0;i<p.length;i++) {
      pv[i] = p[i]/255.0-0.5; // normalize image pixels to [-0.5, 0.5]
    }
    Vol x = new Vol(W, H, RGB, 0.0); //input volume (image)
    x.w = pv;

    if(convert_grayscale) {
      // flatten into depth=1 array
      Vol x1 = new Vol(W, H, BW, 0.0);
      for(int i=0;i<W;i++) {
        for(int j=0;j<H;j++) {
          x1.set(i,j,0,x.get(i,j,0));
        }
      }
      x = x1;
    }

    return x;
  }
  
  int[] getByteArray(BufferedImage img) {
      
    int w = img.getWidth(null);
    int h = img.getHeight(null);
    int[] rgbs = new int[w * h];
    img.getRGB(0, 0, w, h, rgbs, 0, w);
    
    return rgbs;
  }
  
  int[] getRgbByteArray(BufferedImage image) {

        int w = image.getWidth();
        int h = image.getHeight();
        int BLOCK_SIZE = 3;

        int[] pixels = new int[w * h];
        image.getRGB(0, 0, w, h, pixels, 0, w);

        int[] rgbBytes = new int[w * h * BLOCK_SIZE];

        for (int r = 0; r < h; r++) {
            for (int c = 0; c < w; c++) {
                int index = r * w + c;
                int indexRgb = r * w * BLOCK_SIZE + c * BLOCK_SIZE;

                rgbBytes[indexRgb] = (byte)((pixels[index] >> 16) &0xff);
                rgbBytes[indexRgb + 1] = (byte)((pixels[index] >> 8) &0xff);
                rgbBytes[indexRgb + 2] = (byte)(pixels[index] &0xff);
            }
        }

        return rgbBytes;
    }
}
